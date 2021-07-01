from torch_scatter import scatter_add
import torch.nn as nn
from utils import OurTimer
import torch

from config import FLAGS
from layers_gnn_propagator import GNNPropagator
from layers_util import MLP, create_act
from data_structures_search_tree_scalable import unroll_bidomains, get_natts2g2abd_sg_nids
import math
import torch.nn.functional as F
from collections import defaultdict


class DVN(nn.Module):
    def __init__(self, n_dim, n_layers, learn_embs, layer_AGG_w_MLP, Q_mode, Q_act,
                 reward_calculator, environment):
        super(DVN, self).__init__()
        self.gnn_main = GNNPropagator(
            [n_dim for _ in range(n_layers + 1)],
            n_dim,
            'GAT',
            learn_embs,
            layer_AGG_w_MLP)
        self.gnn_bd = GNNPropagator(
            [n_dim for _ in range(n_layers + 1)],
            n_dim,
            'GATMan',
            learn_embs,
            layer_AGG_w_MLP)
        self.environment = environment
        self.reward_calculator = reward_calculator
        self.stride_num = int(Q_mode)  # should encapsulate into tunable interact ops
        self.use_tunable_interact = FLAGS.interact_ops is not None
        self.compute_gs = len(
            {'gs', 'abds', 'ubds', 'bbds'}.intersection(set(FLAGS.emb_mode_list))) > 0

        if self.use_tunable_interact:
            dim_interact_out = int(FLAGS.interact_ops[0])
            self._init_tunable_interact_weights(n_dim, dim_interact_out)
        else:
            # compute stride dimensions
            assert False

        # compute MLP dimensions
        dim_small_list = [dim_interact_out, dim_interact_out]
        dim_big_list = [n_dim, n_dim]
        dim_dqn_vec = len(FLAGS.emb_mode_list) * dim_interact_out
        dim_dqn_list = self._get_dims(dim_dqn_vec, 1)

        if FLAGS.num_nodes_dqn_max > 0:
            num_bds = len([x for x in FLAGS.emb_mode_list if 'bd' in x])
            prune_in_dim = n_dim + (2+num_bds)*dim_interact_out
            prune_dim_list = self._get_dims(prune_in_dim, 1)
            self.MLP_prune = MLP(prune_in_dim, 1,
                                 num_hidden_lyr=len(prune_dim_list),
                                 activation_type='elu',
                                 hidden_channels=prune_dim_list, bn=False)

        if self.compute_gs:
            self.MLP_g_small = None
            self.MLP_g_big = MLP(n_dim, n_dim,
                                 num_hidden_lyr=len(dim_big_list),
                                 activation_type='elu',
                                 hidden_channels=dim_big_list, bn=False)
        for emb_mode in FLAGS.emb_mode_list:
            if emb_mode == 'gs':
                pass
            elif emb_mode == 'sgs':
                self.MLP_sg_small = None

                if FLAGS.simplified_sg_emb:
                    self.MLP_sg = MLP(n_dim, n_dim,
                                      num_hidden_lyr=len(dim_big_list),
                                      activation_type='elu',
                                      hidden_channels=dim_big_list, bn=False)
                else:
                    self.MLP_sg = MLP(dim_interact_out, dim_interact_out,
                                      num_hidden_lyr=len(dim_small_list),
                                      activation_type='elu',
                                      hidden_channels=dim_small_list, bn=False)
            elif emb_mode == 'abds':
                self.MLP_abds_small, self.MLP_abds_big, self.MLP_abd_big = \
                    self.get_MLP_bds(n_dim, dim_big_list, dim_interact_out)
                self.abds_default = self._create_default_emb(dim_interact_out)
                self.register_parameter(name='abds_default', param=self.abds_default)
            elif emb_mode == 'ubds':
                self.MLP_ubds_small, self.MLP_ubds_big, self.MLP_ubd_big = \
                    self.get_MLP_bds(n_dim, dim_big_list, dim_interact_out)
                self.ubds_default = self._create_default_emb(dim_interact_out)
                self.register_parameter(name='ubds_default', param=self.ubds_default)
            elif emb_mode == 'bbds':
                self.MLP_bbds_small, self.MLP_bbds_big, self.MLP_bbd_big = \
                    self.get_MLP_bds(n_dim, dim_big_list, dim_interact_out)
                self.bbds_default = self._create_default_emb(dim_interact_out)
                self.register_parameter(name='bbds_default', param=self.bbds_default)
            else:
                assert False

        self.MLP_final = MLP(dim_dqn_vec, 1,
                             num_hidden_lyr=len(dim_dqn_list),
                             activation_type='elu',
                             hidden_channels=dim_dqn_list, bn=False)

        self.with_bdgnn = FLAGS.with_bdgnn
        self.gnn_per_action = FLAGS.with_gnn_per_action
        self.Q_activation = Q_act
        self.Q_act = create_act(self.Q_activation.split('+')[0])
        self.act = create_act('elu')
        self.cache_d = {}
        self.emb_mode_list_bd = [x for x in FLAGS.emb_mode_list if 'bd' in x]

        # TODO: rm me!
        self.timer = OurTimer()

    def _create_default_emb(self, dim_interact_out):
        if FLAGS.default_emb == 'learnable':
            rtn = torch.nn.Parameter(
                torch.randn(1, dim_interact_out), requires_grad=True)
            nn.init.xavier_normal_(rtn)
            return torch.nn.Parameter(rtn,
                                      requires_grad=True)  # torch.flatten(rtn).to(FLAGS.device)
        elif FLAGS.default_emb == 'zeros':
            return torch.zeros(dim_interact_out).to(FLAGS.device)
        else:
            raise NotImplementedError()

    def get_MLP_bds(self, n_dim, dim_big_list, dim_interact_out):
        MLP_bds_small = None
        # MLP_bds_small = MLP(n_dim, dim_small_proj,
        #                          num_hidden_lyr=len(dim_small_list),
        #                          activation_type='elu',
        #                          hidden_channels=dim_small_list, bn=False)
        if FLAGS.run_bds_MLP_before_interact:
            MLP_bds_big = MLP(n_dim, n_dim,
                              num_hidden_lyr=len(dim_big_list),
                              activation_type='elu',
                              hidden_channels=dim_big_list, bn=False)
        else:
            dim_temp_list = [dim_interact_out, dim_interact_out]
            MLP_bds_big = MLP(dim_interact_out, dim_interact_out,
                              num_hidden_lyr=len(dim_temp_list),
                              activation_type='elu',
                              hidden_channels=dim_temp_list, bn=False)
        MLP_bd_big = MLP(n_dim, n_dim,
                         num_hidden_lyr=len(dim_big_list),
                         activation_type='elu',
                         hidden_channels=dim_big_list, bn=False)
        return MLP_bds_small, MLP_bds_big, MLP_bd_big

    def _init_tunable_interact_weights(self, in_dim, final_d):
        self.interact_weights = nn.ModuleDict()
        self.conv_keys = defaultdict(list)
        if self.compute_gs and 'gs' not in FLAGS.emb_mode_list:
            inter_name_list = FLAGS.emb_mode_list + ['gs']
        else:
            inter_name_list = FLAGS.emb_mode_list
        for inter_name in inter_name_list:
            D = 0
            for op in FLAGS.interact_ops[1:]:
                m = None
                if op == 'chunked_dots':
                    self.stride = in_dim // self.stride_num
                    D += math.ceil(in_dim / self.stride)  # TODO: check non-integer strides
                elif op == 'add':
                    D += in_dim
                elif op == 'product':
                    D += in_dim
                elif '1dconv+max' in op:
                    _, num_channels = op.split('_')
                    num_channels = int(num_channels)
                    for dim in [3, in_dim]:
                        self.interact_weights[f'{inter_name}_{op}_{dim}'] = \
                            nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=dim,
                                      stride=1)
                        self.conv_keys[inter_name].append(f'{inter_name}_{op}_{dim}')
                    D += num_channels * ((in_dim - 2) + 1)  # kernel size = 1, 3, in_dim
                else:
                    raise ValueError(f'Unknown op {op}')
            dim_comb_MLP = self._get_dims(D, final_d)
            self.interact_weights[f'{inter_name}_comb_MLP'] = MLP(D, final_d,
                                                                  num_hidden_lyr=len(dim_comb_MLP),
                                                                  activation_type='elu',
                                                                  hidden_channels=dim_comb_MLP,
                                                                  bn=False)

    def _tunable_interact(self, emb1, emb2, inter_name):
        interact_vec_list = []
        for op in FLAGS.interact_ops[1:]:
            m = None
            if op == 'chunked_dots':
                interaction_list = []
                batch_size, dim_in = emb1.shape
                for start_idx in range(0, dim_in, self.stride):
                    A, B = emb1[:, start_idx:start_idx + self.stride], emb2[:,
                                                                       start_idx:start_idx + self.stride]
                    interaction_list.append(
                        torch.bmm(A.view(batch_size, 1, self.stride),
                                  B.view(batch_size, self.stride, 1)))
                interact_vec = torch.cat(tuple(interaction_list), dim=1).view(batch_size, -1)
            elif op == 'add':
                interact_vec = emb1 + emb2
            elif op == 'product':
                interact_vec = emb1 * emb2
            elif '1dconv+max' in op:
                interact_vec = self._1dconv_max(emb1, emb2, self.conv_keys[inter_name])
            else:
                raise ValueError(f'Unknown op {op}')
            interact_vec_list.append(interact_vec)
        interact_vec_comb = torch.cat(tuple(interact_vec_list), dim=1)
        return self.interact_weights[f'{inter_name}_comb_MLP'](interact_vec_comb)

    def _1dconv_max(self, emb1, emb2, conv_keys):
        cat_outputs = []
        N = emb1.size(0)
        emb_list = [emb1, emb2]
        for i, input in enumerate(emb_list):
            input = torch.unsqueeze(input, 1)

            outputs = []
            for key in conv_keys:
                output = self.interact_weights[key](input)
                outputs.append(output)

            output = torch.cat(tuple(outputs), 2)
            cat_outputs.append(output)

        cat_outputs = torch.tanh(torch.stack(tuple(cat_outputs)))
        final, _ = torch.max(cat_outputs, 0)
        return final.view(N, -1)

    def get_ra(self, dqn_input, action, state, next_state):
        g1, g2 = dqn_input.state.g1, dqn_input.state.g2
        v, w = action
        ra = self.reward_calculator.compute_reward(v, w, g1, g2, state, next_state)
        return ra

    def get_g_scores(self, x1_in, x2_in, dqn_input):
        edge_index1 = dqn_input.state.edge_index1 # TODO: NOT VALID EDGE INDEX!
        edge_index2 = dqn_input.state.edge_index2
        M, N = x1_in.size(0), x2_in.size(0)
        assert FLAGS.num_nodes_dqn_max > 0


        embs = self.cached_op(
            fn=self.compute_embs,
            args=(x1_in, x2_in, edge_index1, edge_index2),
            key=self.get_pair_key(dqn_input, ext='e')
        )

        # TODO: check if g_embs is combined
        g_embs = self.cached_op(
            fn=self.compute_g,
            args=(embs,),
            key=self.get_pair_key(dqn_input, ext='g')
        )

        sg_embs_raw = self.cached_op(
            fn=self.compute_sg,
            args=(embs, dqn_input),
            key=self.get_state_key(dqn_input, ext='g')
        )


        sgs1 = F.normalize(self.MLP_sg(sg_embs_raw[0]), dim=1, p=2)
        sgs2 = F.normalize(self.MLP_sg(sg_embs_raw[1]), dim=1, p=2)
        sg_embs = self._tunable_interact(sgs1, sgs2, 'sgs')

        # TODO: check if bd_emb is combined
        bd_emb = self.compute_bd(
            embs,
            dqn_input.state,
            g_embs,
            None,
            None
        )


        state_embs = torch.cat((g_embs, sg_embs, bd_emb), dim=-1)
        g1_score = torch.cat((embs[0], self._tile_vec(state_embs, M, state_embs.size(-1))), dim=-1)
        g2_score = torch.cat((embs[1], self._tile_vec(state_embs, N, state_embs.size(-1))), dim=-1)
        g1_score, g2_score = self.MLP_prune(g1_score).view(-1), self.MLP_prune(g2_score).view(-1)
        if self.Q_activation == 'elu+1':
            g_scores = self.Q_act(g1_score) + 1, self.Q_act(g2_score) + 1  # (num_actions, 1)
        else:
            g_scores = self.Q_act(g1_score), self.Q_act(g2_score)  # (num_actions, 1)


        return g_scores

    def __call__(self, x1_in, x2_in, dqn_input):
        s_raw_list, s_raw_q_vec_idx_list = [], []
        v_list, w_list, _ = dqn_input.action_space_data.action_space
        q_vec = torch.zeros(len(v_list)).to(FLAGS.device)
        edge_index1 = dqn_input.valid_edge_index1
        edge_index2 = dqn_input.valid_edge_index2

        # # DEPRECATED:
        # # very tricky code -> bidomain means we must use dqn_input for edges NOT next state...
        # # NOTE: x1 includes the un-coarse subgraph nodes (it just doesn't get propagated)
        # #       valid_indices does not include these subgraph nodes
        # x1_proc, x2_proc, edge_index1, edge_index2 = \
        #     self.collapse_graphs(x1_in, x2_in, dqn_input, action)
        # valid_indices1, valid_indices2 = self.get_valid_indices(x1_proc, x2_proc, next_state)

        # Q = r + DVN(state)

        embs = self.cached_op(
            fn=self.compute_embs,
            args=(x1_in, x2_in, edge_index1, edge_index2),
            key=self.get_pair_key(dqn_input, ext='e')
        )

        g_embs = self.cached_op(
            fn=self.compute_g,
            args=(embs,),
            key=self.get_pair_key(dqn_input, ext='g')
        )

        sg_embs = self.cached_op(
            fn=self.compute_sg,
            args=(embs, dqn_input),
            key=self.get_state_key(dqn_input, ext='g')
        )

        # A = (a1, a2, a3)
        # St+1 = (s1', s2', s3')

        bd_embs = []
        timer = OurTimer()
        env_time = 0
        MLP_time = 0
        temp_time = self.timer.get_duration()
        for q_vec_idx, action in enumerate(zip(v_list, w_list)):
            _, next_state = self.environment(dqn_input.state, action, q_vec_idx) #si' = env(si,ai)
            env_time += self.timer.get_duration() - temp_time
            temp_time = self.timer.get_duration()
            q_vec[q_vec_idx] += \
                self.get_ra(dqn_input, action, dqn_input.state, next_state)

            if self.state_is_leaf_node(next_state):
                continue
            else:
                bd_embs.append(
                    self.compute_bd(
                        embs,
                        next_state,
                        g_embs,
                        q_vec_idx,
                        timer)
                ) # bd_emb(si)

                s_raw_q_vec_idx_list.append(q_vec_idx)
            MLP_time += self.timer.get_duration() - temp_time
            temp_time = self.timer.get_duration()

        if len(s_raw_q_vec_idx_list) > 0:
            g_embs = self.collate_g(g_embs, s_raw_q_vec_idx_list)
            sg_embs = self.collate_sg(sg_embs, embs, v_list, w_list, s_raw_q_vec_idx_list)
            bd_embs = self.collate_bd(bd_embs)

            state_embs = torch.cat((g_embs, sg_embs, bd_embs), dim=-1)

            q_vec_raw = self.MLP_final(state_embs).view(-1)
            if FLAGS.device == 'cpu':
                s_raw_q_vec_idx = torch.LongTensor(s_raw_q_vec_idx_list).view(-1)
            else:
                s_raw_q_vec_idx = torch.cuda.LongTensor(s_raw_q_vec_idx_list).view(-1)
            if self.Q_activation == 'elu+1':
                s_raw_q_vec = self.Q_act(q_vec_raw) + 1  # (num_actions, 1)
            else:
                s_raw_q_vec = self.Q_act(q_vec_raw)  # (num_actions, 1)
            scatter_add(self.reward_calculator.discount * s_raw_q_vec, s_raw_q_vec_idx, out=q_vec) # Q += DVN(state)

        if FLAGS.time_analysis:
            timer = OurTimer()
        else:
            timer = None

        return q_vec

    def cached_op(self, fn, args, key):
        hierarchy, sub_key = key
        if hierarchy not in self.cache_d:
            self.cache_d[hierarchy] = {}

        if sub_key not in self.cache_d[hierarchy]:
            val = fn(*args)
            self.cache_d[hierarchy] = {sub_key: val}  # do not retain old data for memory saving
            # if hierarchy[-1] == 'e':
            #     print('NOT IN HIERARCHY:', hierarchy, sub_key)
        else:
            val = self.cache_d[hierarchy][sub_key]
        return val

    def get_pair_key(self, dqn_input, ext=''):
        hierarchy = f'pair_{ext}'
        # exhausted_v, exhausted_w = \
        #     frozenset(dqn_input.exhausted_v), frozenset(dqn_input.exhausted_w)
        exhausted_v, exhausted_w = frozenset(), frozenset()
        sub_key = \
            (dqn_input.state.cur_id, dqn_input.pair_id, exhausted_v, exhausted_w)
        key = (hierarchy, sub_key)
        return key

    def get_state_key(self, dqn_input, ext=''):
        hierarchy = f'state_{ext}'
        sub_key = (dqn_input.pair_id,
                   frozenset(dqn_input.state.nn_map.keys()),
                   frozenset(dqn_input.state.nn_map.values()))
        key = (hierarchy, sub_key)
        return key

    def compute_embs(self, x1_in, x2_in, edge_index1, edge_index2):
        x1_out, x2_out, _, _, _ = self.gnn_main(x1_in, x2_in, edge_index1, edge_index2)
        embs = x1_out, x2_out
        return embs

    def compute_g(self, embs):
        x1, x2 = embs
        g1 = self.MLP_g_big(torch.sum(x1, dim=0)).view(1, -1)
        g2 = self.MLP_g_big(torch.sum(x2, dim=0)).view(1, -1)

        g1 = F.normalize(g1, dim=1, p=2)
        g2 = F.normalize(g2, dim=1, p=2)

        gs = self._tunable_interact(g1, g2, 'gs')
        return gs

    def compute_sg(self, embs, dqn_input):
        x1, x2 = embs
        sgs1 = torch.sum(x1[list(dqn_input.state.nn_map.keys())], dim=0).view(1, -1)
        sgs2 = torch.sum(x2[list(dqn_input.state.nn_map.values())], dim=0).view(1, -1)
        sgs = (sgs1, sgs2)
        return sgs

    def compute_bd(self, embs, state, gs, q_vec_idx, timer):
        emb_list = []
        x1_in, x2_in = embs
        for emb_mode in self.emb_mode_list_bd:
            # construct bidomain embeddings
            if emb_mode == 'abds':
                bds1_list_raw, bds2_list_raw = [], []
                for bidomain in unroll_bidomains(state.natts2bds):
                    # abd(label = i) = sum(x[abd_j(label = i)])
                    bds1_list_raw.append(torch.sum(x1_in[list(bidomain.left)], dim=0))
                    bds2_list_raw.append(torch.sum(x2_in[list(bidomain.right)], dim=0))
            elif emb_mode == 'ubds':
                natts2g2abd_sg_nids = \
                    get_natts2g2abd_sg_nids(
                        state.natts2g2nids, state.natts2bds, state.nn_map)
                bds1_list_raw, bds2_list_raw = [], []
                for natts, g2nids in state.natts2g2nids.items():
                    # ubd(label = i) = sum(x[g(label = i) - (sum_j abd_j(label = i) + sg(label=i))])
                    if natts in natts2g2abd_sg_nids:
                        g2abd_sg_nids = natts2g2abd_sg_nids[natts]
                        indices_1 = list(set(g2nids['g1']) - set(g2abd_sg_nids['g1']))
                        indices_2 = list(set(g2nids['g2']) - set(g2abd_sg_nids['g2']))
                    else:
                        indices_1 = list(g2nids['g1'])
                        indices_2 = list(g2nids['g2'])
                    bds1_raw = torch.sum(x1_in[indices_1], dim=0)
                    bds2_raw = torch.sum(x2_in[indices_2], dim=0)

                    bds1_list_raw.append(bds1_raw)
                    bds2_list_raw.append(bds2_raw)
            else:
                assert False

            emb = \
                self._get_bds(
                    bds1_list_raw, bds2_list_raw,
                    emb_mode,
                    state, gs, q_vec_idx,
                    timer
                )
            emb_list.append(emb.view(1, -1))

        bds = torch.cat(tuple(emb_list), dim=1)
        return bds

    def collate_g(self, g_embs, s_raw_q_vec_idx_list):
        g_embs_collate = self._tile_vec(g_embs, len(s_raw_q_vec_idx_list), g_embs.size(-1))
        return g_embs_collate

    def collate_sg(self, sg_embs, embs, v_list, w_list, s_raw_q_vec_idx_list):
        n_dim = sg_embs[0].size(-1)
        sgs1 = self._tile_vec(sg_embs[0], len(v_list), n_dim)[s_raw_q_vec_idx_list] + \
               (embs[0][v_list])[s_raw_q_vec_idx_list]
        sgs2 = self._tile_vec(sg_embs[1], len(w_list), n_dim)[s_raw_q_vec_idx_list] + \
               (embs[1][w_list])[s_raw_q_vec_idx_list]
        sgs1 = F.normalize(self.MLP_sg(sgs1), dim=1, p=2)
        sgs2 = F.normalize(self.MLP_sg(sgs2), dim=1, p=2)
        sg_embs_collate = self._tunable_interact(sgs1, sgs2, 'sgs')
        return sg_embs_collate

    def collate_bd(self, bd_embs):
        bd_embs_collate = torch.cat(tuple(bd_embs), dim=0)
        return bd_embs_collate

    def state_is_leaf_node(self, state):
        return len(unroll_bidomains(state.natts2bds)) == 0

    def _get_bds(self, bds1_list_raw, bds2_list_raw, mode, state, gs, q_vec_idx, timer):
        if mode == 'abds':
            default_vec = self.abds_default
        elif mode == 'ubds':
            default_vec = self.ubds_default
        else:
            assert False

        assert len(bds1_list_raw) == len(bds2_list_raw)
        if len(bds1_list_raw) > 0:
            # bds1_list_raw, bds2_list_raw = [], []
            bds1_list_raw = torch.stack(tuple(bds1_list_raw), dim=0)
            bds2_list_raw = torch.stack(tuple(bds2_list_raw), dim=0)

            if FLAGS.no_bd_MLPs:
                # TODO: I put this here so that i can load models with MLP_abd_..., put this init in future
                bds1_list = bds1_list_raw
                bds2_list = bds2_list_raw
            elif mode == 'abds':
                bds1_list = self.MLP_abd_big(bds1_list_raw)
                bds2_list = self.MLP_abd_big(bds2_list_raw)
            elif mode == 'ubds':
                bds1_list = self.MLP_ubd_big(bds1_list_raw)
                bds2_list = self.MLP_ubd_big(bds2_list_raw)
            elif mode == 'bbds':
                bds1_list = self.MLP_bbd_big(bds1_list_raw)
                bds2_list = self.MLP_bbd_big(bds2_list_raw)
            else:
                assert False

            bds_list = self._tunable_interact(
                F.normalize(bds1_list, dim=1, p=2),
                F.normalize(bds2_list, dim=1, p=2),
                mode
            )  # num_bds by D

            if FLAGS.attention_bds:
                # assert gs.shape[0] == 1
                att = torch.matmul(bds_list, gs.t())
                att = F.softmax(att, 0)
                bds_list = bds_list * att

            # else:
            if FLAGS.no_bd_MLPs:
                # TODO: I put this here so that i can load models with MLP_abd_..., put this init in future
                print('@@@@@@@@@@@@@@@@@@@@@')
                bds = torch.sum(bds_list, dim=0).view(1, -1)
            elif mode == 'abds':
                bds = self.MLP_abds_big(torch.sum(bds_list, dim=0)).view(1, -1)
            elif mode == 'ubds':
                bds = self.MLP_ubds_big(torch.sum(bds_list, dim=0)).view(1, -1)
            elif mode == 'bbds':
                bds = self.MLP_bbds_big(torch.sum(bds_list, dim=0)).view(1, -1)
            else:
                assert False

            if FLAGS.normalize_emb:
                bds = F.normalize(bds, dim=1, p=2)

            bds_out = default_vec + bds
        else:
            bds_out = default_vec

        return bds_out

    def collapse_graphs(self, x1_in, x2_in, dqn_input, action):
        v, w = action
        sg1_nid = dqn_input.state.g1.number_of_nodes()
        sg2_nid = dqn_input.state.g2.number_of_nodes()
        x1_in, edge_index1 = self.collapse_graph(
            x1_in,
            sg1_nid,
            dqn_input.valid_edge_index1,
            set(dqn_input.state.nn_map.keys()).union({v}))
        x2_in, edge_index2 = self.collapse_graph(
            x2_in,
            sg2_nid,
            dqn_input.valid_edge_index2,
            set(dqn_input.state.nn_map.values()).union({w}))
        return x1_in, x2_in, edge_index1, edge_index2

    def collapse_graph(self, x_in, sg_nid, edge_index, sg_nodes):
        for nid in sg_nodes:
            edge_index[0][edge_index[0] == nid] = sg_nid
            edge_index[1][edge_index[1] == nid] = sg_nid
            non_redundant_edges = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, non_redundant_edges]
        x_in = torch.cat(
            (x_in, self.local_degree_prof(x_in, sg_nodes).view(1, -1)), dim=0)
        return x_in, edge_index

    def get_valid_indices(self, x1, x2, state):
        masked_nodes_l = set(state.nn_map.keys())
        masked_nodes_r = set(state.nn_map.values())
        valid_indices1 = set(range(x1.size(0))) - masked_nodes_l
        valid_indices2 = set(range(x2.size(0))) - masked_nodes_r
        return valid_indices1, valid_indices2

    def interact_and_encode(self, emb1, emb2, MLP):
        encode_vec = MLP(emb1 + emb2)

        interaction_list = []
        batch_size, dim_in = emb1.shape
        for start_idx in range(0, dim_in, self.stride):
            A, B = emb1[:, start_idx:start_idx + self.stride], emb2[:,
                                                               start_idx:start_idx + self.stride]
            interaction_list.append(
                torch.bmm(A.view(batch_size, 1, self.stride),
                          B.view(batch_size, self.stride, 1)))
        interact_vec = torch.cat(tuple(interaction_list), dim=1).view(batch_size, -1)

        interact_and_encode_vec = torch.cat((interact_vec, encode_vec), dim=1)
        return interact_and_encode_vec

    def _get_dims(self, in_dim, out_dim):
        dims = []
        dim_val = in_dim
        while dim_val > out_dim:
            dim_val = dim_val // 2
            dims.append(dim_val)
        dims = dims[:-1]
        return dims

    def _tile_vec(self, vec, dim0, dim1):
        return torch.cat(dim0 * [vec.view((1, dim1))])
