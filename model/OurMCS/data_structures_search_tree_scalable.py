from data_structures_common_scalable import DoubleDict
from config import FLAGS

from collections import defaultdict
from copy import deepcopy
import networkx as nx

#########################################################################
# Bidomain
#########################################################################
class Bidomain(object):
    def __init__(self, left, right, natts, bid=None):
        self.left = left
        self.right = right
        self.natts = natts
        self.bid = bid

    def __len__(self):
        return len(self.left) * len(self.right)

def get_natts_hash(node):
    if 'fuzzy_matching' in FLAGS.reward_calculator_mode:
        natts = []
    else:
        natts = FLAGS.node_feats_for_mcs
    natts_hash = tuple([node[natt] for natt in natts])
    return natts_hash


def unroll_bidomains(natts2bds):
    bidomains = [bd for bds in natts2bds.values() for bd in bds]
    return bidomains


def get_natts2g2abd_sg_nids(natts2g2nids, natts2bds, nn_map):
    natts2g2abd_sg_nids = defaultdict(dict)
    sg1, sg2 = set(nn_map.keys()), set(nn_map.values())
    for natts, g2nid in natts2g2nids.items():
        left_cum, right_cum = set(), set()
        if natts in natts2bds:
            for bd in natts2bds[natts]:
                left_cum.update(bd.left)
                right_cum.update(bd.right)
        left_cum.update(sg1.intersection(g2nid['g1']))  # TODO: potential bottleneck O(nn_map)
        left_cum.update(sg2.intersection(g2nid['g2']))
        natts2g2abd_sg_nids[natts]['g1'] = left_cum
        natts2g2abd_sg_nids[natts]['g2'] = right_cum
    return natts2g2abd_sg_nids


def assign_bids(natts2bds):
    # to ensure that given the same natts2bds, bid assignment is deterministic => sorted
    bid = 0
    for natts in sorted(natts2bds.keys()):
        for bd in natts2bds[natts]:
            bd.bid = bid
            bid += 1

#########################################################################
# Action Space
#########################################################################
class ActionSpaceData():
    def __init__(self, action_space, natts2bds_unexhausted, action_space_size_unexhausted_unpruned):
        self.action_space = action_space
        self.natts2bds_unexhausted = natts2bds_unexhausted
        self.action_space_size_unexhausted_unpruned = action_space_size_unexhausted_unpruned

    def filter_action_space_data(self, v, w):
        for bds in self.natts2bds_unexhausted.values():
            for bd in bds:
                if v in bd.left and w in bd.right:
                    self.action_space = [[v], [w], [bd.bid]]

#########################################################################
# State Nodes
#########################################################################
class StateNode(object):
    def __init__(self, ins_g1, ins_g2, nn_map, nn_map_neighbors, natts2bds,
                 natts2g2nids, edge_index1, edge_index2, adj_list1, adj_list2,
                 g1, g2, degree_mat, sgw_mat, pca_mat, cur_id, mcsp_vec,
                 MCS_size_UB, explore_n_pairs=None, pruned_actions=None,
                 exhausted_v=None, exhausted_w=None, tree_depth=0, num_steps=0,
                 cum_reward=0):
        self.ins_g1 = ins_g1
        self.ins_g2 = ins_g2
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.adj_list1 = adj_list1
        self.adj_list2 = adj_list2
        self.g1 = g1
        self.g2 = g2
        self.degree_mat = degree_mat
        self.sgw_mat = sgw_mat
        self.pca_mat = pca_mat
        self.x1 = None
        self.x2 = None
        self.nn_map = nn_map
        self.nn_map_neighbors = nn_map_neighbors
        self.natts2bds = natts2bds
        self.natts2g2nids = natts2g2nids
        self.q_vec = None
        self.recompute = False

        # for additional pruning
        self.pruned_actions = DoubleDict() if pruned_actions is None else pruned_actions
        self.exhausted_v = set() if exhausted_v is None else exhausted_v
        self.exhausted_w = set() if exhausted_w is None else exhausted_w

        # for search tree
        self.action_next_list = []
        self.action_prev = None

        self.MCS_size_UB = MCS_size_UB
        self.cum_reward = cum_reward
        self.v_search_tree = 0  # exhausted_q_max_LB

        self.explore_n_pairs = explore_n_pairs  # thresholds number of times we backtrack
        self.nid = None  # for search_tree.nxgraph
        self.tree_depth = tree_depth
        self.num_steps = num_steps
        self.cur_id = cur_id
        self.mcsp_vec = mcsp_vec
        self.last_v = None

    def compute_reward_from(self, start_node, discount):
        reward_stack = []
        cur_node = self
        while cur_node is not start_node:
            action_prev = cur_node.action_prev
            assert action_prev is not None
            reward = action_prev.reward
            cur_node = action_prev.state_prev
            assert cur_node is not None
            reward_stack.append(reward)

        i = 0
        cum_reward = 0
        while len(reward_stack) > 0:
            reward = reward_stack.pop()
            cum_reward += (discount ** i) * reward
        return cum_reward

    def disentangle_paths(self):
        if len(self.action_next_list) == 0:
            state_cur_list = [deepcopy(self)]
        else:
            state_cur_list = []
            for action_next in self.action_next_list:
                state_next_list = action_next.state_next.disentangle_paths()
                for state_next in state_next_list:
                    state_cur = deepcopy(self)
                    action_edge = deepcopy(action_next)
                    action_edge.link_action2state(state_cur, state_next)
                    state_cur_list.append(state_cur)
        return state_cur_list

    def assign_v(self, discount):
        # unrolling the recursive function calls
        state_stack_order = [self]
        state_stack_compute = []
        while len(state_stack_order) != 0:
            state_popped = state_stack_order.pop()
            state_stack_compute.append(state_popped)
            for action_next in state_popped.action_next_list:
                state_stack_order.append(action_next.state_next)

        # recursively compute LB
        for state in state_stack_compute[::-1]:
            for action_next in state.action_next_list:
                v_max_next_state = action_next.state_next.v_search_tree
                q_max_cur_state = \
                    action_next.reward + discount * v_max_next_state
                state.v_search_tree = max(state.v_search_tree, q_max_cur_state)

    def prune_action(self, v, w):  # , remove_nodes):
        self.pruned_actions.add_lr(v, w)
        # if remove_nodes:
        for bd in unroll_bidomains(self.natts2bds):
            left_valid = bd.left - set(self.exhausted_v)
            right_valid = bd.right - set(self.exhausted_w)
            if v in left_valid:
                assert w in right_valid
                if len(right_valid - self.pruned_actions.l2r[v]) == 0:
                    self.exhausted_v.add(v)
                    self.recompute = True
                if len(left_valid - self.pruned_actions.r2l[w]) == 0:
                    self.exhausted_w.add(w)
                    self.recompute = True

        for g2nids in self.natts2g2nids.values():
            if len(g2nids['g2'] - self.pruned_actions.l2r[v]) == 0:
                self.exhausted_v.add(v)
                self.recompute = True
            if len(g2nids['g1'] - self.pruned_actions.r2l[w]) == 0:
                self.exhausted_w.add(w)
                self.recompute = True

    def get_natts2bds_ubd_unexhausted(self):
        natts2bds_unexhausted = defaultdict(list)
        for natts, g2nids in self.natts2g2nids.items():
            left = g2nids['g1'] - self.exhausted_v
            right = g2nids['g2'] - self.exhausted_w
            if len(left) > 0 and len(right) > 0:
                natts2bds_unexhausted[natts].append(
                    Bidomain(left, right, natts)
                )
        return natts2bds_unexhausted

    def get_natts2bds_abd_unexhausted(self):
        natts2bds_unexhausted = defaultdict(list)
        for natts, bds in self.natts2bds.items():
            for bd in bds:
                assert bd.natts == natts
                if len(bd.left.intersection(self.exhausted_v)) > 0 or \
                        len(bd.right.intersection(self.exhausted_w)) > 0:
                    left = bd.left - self.exhausted_v
                    right = bd.right - self.exhausted_w
                    if len(left) > 0 and len(right) > 0:
                        natts2bds_unexhausted[natts].append(
                            Bidomain(left, right, natts)
                        )
                else:
                    assert len(bd.left) > 0 and len(bd.right) > 0
                    natts2bds_unexhausted[natts].append(bd)
        return natts2bds_unexhausted

    def get_natts2bds_unexhausted(self, with_bids=True):
        if len(self.nn_map) == 0:
            natts2bds_unexhausted = self.get_natts2bds_ubd_unexhausted()
        else:
            natts2bds_unexhausted = self.get_natts2bds_abd_unexhausted()
        if with_bids:
            assign_bids(natts2bds_unexhausted)
        return natts2bds_unexhausted

#########################################################################
# Action Edge
#########################################################################
class ActionEdge(object):
    def __init__(self, action, reward,
                 pruned_actions=None, exhausted_v=None, exhausted_w=None):
        self.action = action
        self.reward = reward

        # input to DQN
        self.pruned_actions = DoubleDict() \
            if pruned_actions is None else deepcopy(pruned_actions)
        self.exhausted_v = set() \
            if exhausted_v is None else deepcopy(exhausted_v)
        self.exhausted_w = set() \
            if exhausted_w is None else deepcopy(exhausted_w)

        # for search tree
        self.state_prev = None
        self.state_next = None

    def link_action2state(self, cur_state, next_state):
        self.state_prev = cur_state
        self.state_next = next_state
        cur_state.action_next_list.append(self)
        next_state.action_prev = self

#########################################################################
# Search Tree
#########################################################################
class SearchTree(object):
    def __init__(self, root):
        self.root = root
        self.nodes = {root}
        self.edges = set()
        self.nxgraph = nx.Graph()

        # add root to nxgraph
        self.cur_nid = 0
        root.nid = self.cur_nid
        self.nxgraph.add_node(self.cur_nid)
        self.nid2ActionEdge = {}
        self.cur_nid += 1

        node_stack = [root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            for action_edge in node.action_next_list:
                node_next = action_edge.state_next
                node_next.cur_nid = self.cur_nid
                self.cur_nid += 1
                self.nodes.add(node_next)
                assert action_edge not in self.edges
                self.edges.add(action_edge)
                self.nxgraph.add_node(node_next.cur_nid)
                self.nid2ActionEdge[node_next.cur_nid] = action_edge
                self.nxgraph.add_edge(node.nid, node_next.cur_nid)
                node_stack.append(node_next)

    def link_states(self, cur_state, action_edge, next_state, q_pred, discount):
        action_edge.link_action2state(cur_state, next_state)

        assert cur_state in self.nodes
        self.nodes.add(next_state)
        assert action_edge not in self.edges
        self.edges.add(action_edge)

        # add edge, node to nxgraph
        assert cur_state.nid is not None
        next_state.nid = self.cur_nid
        self.nxgraph.add_node(self.cur_nid)
        self.nid2ActionEdge[self.cur_nid] = action_edge
        self.nxgraph.add_edge(cur_state.nid, self.cur_nid)
        eid = (cur_state.nid, self.cur_nid)
        self.assign_val_to_edge(eid, 'q_pred', q_pred.item())

        # accumulate the reward
        next_state.cum_reward = \
            self.get_next_cum_reward(cur_state, action_edge, discount)
        self.cur_nid += 1

    def get_next_cum_reward(self, cur_state, action_edge, discount):
        next_cum_reward = \
            cur_state.cum_reward + (discount ** cur_state.num_steps) * action_edge.reward
        return next_cum_reward

    def disentangle_paths(self):
        search_tree_list = []
        root_list = self.root.disentangle_paths()
        for root in root_list:
            search_tree_list.append(SearchTree(root))
        return search_tree_list

    def assign_v_search_tree(self, discount):  # , g1, g2):
        self.root.assign_v(discount)

    def associate_q_pred_true_with_node(self):
        for edge in self.edges:
            nid = edge.state_next.nid
            eid = (edge.state_prev.nid, edge.state_next.nid)
            q_pred = self.nxgraph.edges[eid]['q_pred']
            if 'q_true' in self.nxgraph.edges[eid]:
                q_true = self.nxgraph.edges[eid]['q_true']
            else:
                q_true = None
            if 'q_pred_norm' in self.nxgraph.edges[eid]:
                q_pred_norm = self.nxgraph.edges[eid]['q_pred_norm']
            else:
                q_pred_norm = None
            self.assign_val_to_node(nid, 'q_pred', q_pred)
            self.assign_val_to_node(nid, 'q_pred_str', '{:.2f}'.format(q_pred))
            if q_true is not None:
                self.assign_val_to_node(nid, 'q_true', q_true)
            if q_pred_norm is not None:
                self.assign_val_to_node(nid, 'q_pred_norm', q_pred_norm)
                self.assign_val_to_node(nid, 'q_pred_norm_str', '{:.3f}'.format(q_pred_norm))

    def assign_val_to_node(self, nid, key, val):
        self.nxgraph.nodes[nid][key] = val

    def assign_val_to_edge(self, eid, key, val):
        self.nxgraph.edges[eid][key] = val
