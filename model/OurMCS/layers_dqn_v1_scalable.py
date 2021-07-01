import torch.nn as nn
import torch

from data_structures_common_scalable import DQNInput
from data_structures_search_tree_scalable import ActionSpaceData
from config import FLAGS
from layers_dvn_scalable import DVN

# TODO: line 63ish!!

class Q_network_v1(nn.Module):
    def __init__(self, encoder_type, embedder_type, interact_type, in_dim, n_dim,
                 n_layers, GNN_mode, learn_embs, layer_AGG_w_MLP, Q_mode, Q_act,
                 reward_calculator=None, environment=None):
        super(Q_network_v1, self).__init__()
        # TODO: figure out a way to not have to pass in as variables
        #  -> just pass encoder_args, embedder_args, interactor_args
        self.encoder_type = encoder_type
        self.embedder_type = embedder_type
        self.interact_type = interact_type
        self.encode = EncoderMLP(in_dim, n_dim)
        if interact_type == 'dvn':
            self.embed = None
            self.interact = None
            self.dvn = DVN(
                n_dim, n_layers, learn_embs, layer_AGG_w_MLP, Q_mode, Q_act, reward_calculator,
                environment)
        else:
            assert False

    def __call__(self, dqn_input, detach_in_chunking_stage=False):
        x1_in, x2_in = self.encode(dqn_input.state.ins_g1), self.encode(dqn_input.state.ins_g2)
        if self.interact_type == 'dvn':
            q_vec = self._chunked_interact(x1_in, x2_in, dqn_input, detach_in_chunking_stage,
                                           self.dvn)
        else:
            assert False
        return q_vec

    def get_g_scores(self, dqn_input):
        # TODO: fix below duplicate code with li 53-57
        x1_in, x2_in = self.encode(dqn_input.state.ins_g1), self.encode(dqn_input.state.ins_g2)
        if self.interact_type != 'dvn':
            assert False # dqn version not supported
            x1_in, x2_in, _, _, _ = self.embed(x1_in, x2_in, dqn_input)
        return self.dvn.get_g_scores(x1_in, x2_in, dqn_input)

    def _chunked_interact(self, x1_in, x2_in, dqn_input, detach_in_chunking_stage, fun):
        # ex. 'Xproduct-x1_x2-sg1_sg2_g1_g2'
        max_chunk_size = FLAGS.max_chunk_size  # 128
        num_actions = len(dqn_input.action_space_data.action_space[0])
        # print(f'in dvn wrapper: {num_actions}')
        if num_actions > max_chunk_size:
            dqn_input_chunk_list = self.get_chunks(dqn_input, max_chunk_size)
            q_vec_chunk_list = []
            for dqn_input_chunk in dqn_input_chunk_list:
                if detach_in_chunking_stage:
                    q_vec_chunk = fun(x1_in, x2_in, dqn_input_chunk).detach()
                else:
                    q_vec_chunk = fun(x1_in, x2_in, dqn_input_chunk)
                q_vec_chunk_list.append(q_vec_chunk)
            q_vec = self.merge_chunks(q_vec_chunk_list)
        else:
            q_vec = fun(x1_in, x2_in, dqn_input)
        return q_vec

    def get_chunks(self, dqn_input, max_chunk_size):
        prev_slice_idx = 0
        chunks = []
        while prev_slice_idx < len(dqn_input.action_space_data.action_space[0]):
            next_slice_idx = prev_slice_idx + max_chunk_size
            chunks.append(self.obtain_chunk(dqn_input, prev_slice_idx, next_slice_idx))
            prev_slice_idx = next_slice_idx
        return chunks

    def obtain_chunk(self, dqn_input, prev_slice_idx, next_slice_idx):
        v1f_idx, v2f_idx, bd_indices = dqn_input.action_space_data.action_space
        action_space_chunk = v1f_idx[prev_slice_idx:next_slice_idx], \
                             v2f_idx[prev_slice_idx:next_slice_idx], \
                             bd_indices[prev_slice_idx:next_slice_idx]
        action_space_data_chunk = ActionSpaceData(
            action_space_chunk,
            dqn_input.action_space_data.natts2bds_unexhausted,
            dqn_input.action_space_data.action_space_size_unexhausted_unpruned
        )
        dqn_input_chunk = DQNInput(dqn_input.state, action_space_data_chunk, dqn_input.restore_bidomains)
        return dqn_input_chunk

    def merge_chunks(self, x_in_chunk_list):
        return torch.cat(x_in_chunk_list, dim=0)  # TODO: dim = -1 or 0?


class EncoderMLP(nn.Module):
    def __init__(self, in_dim, n_dim):
        super(EncoderMLP, self).__init__()
        self.mlp = nn.Linear(in_dim, n_dim)

    def __call__(self, emb):
        return self.mlp(emb)
