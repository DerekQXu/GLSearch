import numpy as np

class RewardCalculator():
    def __init__(self, mode, feat_map, calc_bound=None, discount = 1.00):
        if ';' in mode:
            mode_name, mode_args = mode.split(';')
            mode_args = mode_args.split(',')
        else:
            mode_name, mode_args = mode, []
        self.mode = mode_name
        self.discount = discount

        if self.mode == 'vanilla':
            pass
        elif self.mode == 'normalized':
            pass
        elif self.mode == 'edge_count':
            pass
        elif self.mode == 'normalized_edge_count':
            pass
        elif self.mode == 'normalized_edge_count_hybrid':
            beta, = mode_args
            self.beta = float(beta)
        elif mode_name == 'mcsprl':
            assert calc_bound is not None
        elif self.mode == 'weighted_reward':
            node_score_config, = mode_args
            self._init_compute_score(node_score_config)
        elif self.mode == 'fuzzy_matching':
            reward_matrix_config, = mode_args
            self.reward_matrix = self._generate_reward_matrix(len(feat_map), reward_matrix_config)
        else:
            assert False

        # utility functions
        self.feat_map = feat_map
        self.calc_bound = calc_bound

    def compute_reward_batch(self, action_space, g1, g2, state, next_state):
        v_list, w_list, _ = action_space

        r = []
        for v,w in zip(v_list,w_list):
            r.append(self.compute_reward(v,w,g1,g2,state,next_state))
        return np.array(r)

    def compute_reward(self, v, w, g1, g2, state, next_state):
        if self.mode == 'vanilla':
            r = 1.0
        elif self.mode == 'normalized':
            r = 1.0 / min(g1.number_of_nodes(),g2.number_of_nodes())
        elif self.mode == 'edge_count':
            r = self._get_delta_edges(state, next_state, g1, g2)
        elif self.mode == 'normalized_edge_count':
            r_num = self._get_delta_edges(state, next_state, g1, g2)
            r_denom = (g1.number_of_edges() + g2.number_of_edges())/2
            r = r_num / r_denom
        elif self.mode == 'normalized_edge_count_hybrid':
            r_num_edge = self._get_delta_edges(state, next_state, g1, g2)
            r_denom_edge = (g1.number_of_edges() + g2.number_of_edges())/2
            r_edge = r_num_edge / r_denom_edge
            r_node = 1.0 / min(g1.number_of_nodes(),g2.number_of_nodes())
            r = self.beta * r_node + (1-self.beta) * r_edge
        elif self.mode == 'mcsprl':
            r = self.calc_bound(state) - self.calc_bound(next_state)
        elif self.mode == 'weighted_reward':
            v_score = self.compute_score(v, g1)
            w_score = self.compute_score(v, g2)
            r = (v_score + w_score)/2
            assert r < 1.1
        elif self.mode == 'fuzzy_matching':
            # ... TODO: only works with synthetic datasets!
            assert v is not None and w is not None
            v_idx = self.feat_map[g1.nodes[v]['type']]
            w_idx = self.feat_map[g2.nodes[w]['type']]
            r = self.reward_matrix[v_idx,w_idx]
        else:
            assert False

        return r

    ################################################################

    def _get_delta_edges(self, state, next_state, g1, g2):
        sg1_state = list(state.nn_map.keys())
        sg2_state = list(state.nn_map.values())
        sg1_next_state = list(next_state.nn_map.keys())
        sg2_next_state = list(next_state.nn_map.values())
        num_edges_state = \
            g1.subgraph(sg1_state).number_of_edges() + \
            g2.subgraph(sg2_state).number_of_edges()
        num_edges_next_state = \
            g1.subgraph(sg1_next_state).number_of_edges() + \
            g2.subgraph(sg2_next_state).number_of_edges()
        delta_edges = (num_edges_next_state - num_edges_state)/2
        return delta_edges

    def _init_compute_score(self, node_score_config):
        if node_score_config == 'inv_degree':
            self.compute_score = lambda v, g: 1/g.degree[v]
        elif node_score_config == 'act_degree':
            self.compute_score = lambda v, g: np.tanh(g.degree[v]/10)
        else:
            assert False

    def _generate_reward_matrix(self, num_features, reward_matrix_config):
        if reward_matrix_config == 'diag':
            reward_matrix = np.diag([1]*num_features)
        elif reward_matrix_config == 'eye':
            x = np.random.randn(177*num_features)
            y = np.random.randn(177*num_features)
            reward_matrix, _, _ = np.histogram2d(x, y, bins=num_features)
        elif reward_matrix_config == 'soft_diag':
            gaussian = np.random.normal(0, 0.1, 100 * (num_features * 2))
            reward_vec, _ = np.histogram(gaussian, bins=num_features * 2)
            reward_matrix = []
            for i in range(num_features - 1, -1, -1):
                reward_matrix.append(reward_vec[i:i + num_features])
            reward_matrix = np.stack(tuple(reward_matrix))
            reward_matrix = (reward_matrix + np.transpose(reward_matrix)) / (
                        2 * np.max(reward_matrix))
        else:
            assert False
        return reward_matrix
