from config import FLAGS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from layers_GATConvManual import GATConvManual

#########################################################################
# GNN Method
#########################################################################
class GNNPropagator(nn.Module):
    def __init__(self, dims, out_dim, gnn_type, learn_embs, layer_AGG_w_MLP):
        super(GNNPropagator, self).__init__()
        self.n_layers = len(dims) - 1
        self.GNNs = []
        self.SGMNs = []
        self.MLPs = []
        self.gnn_opts = gnn_type.split('-')
        self.w = nn.Parameter(torch.Tensor(*(1, self.n_layers)).to(FLAGS.device),
                              requires_grad=True)
        nn.init.xavier_normal_(self.w)
        for i in range(self.n_layers):
            if 'GCN' in self.gnn_opts:
                self.GNNs.append(GCNConv(dims[i], dims[i + 1]))
            elif 'GIN' in self.gnn_opts:
                self.GNNs.append(GINConv(dims[i], dims[i + 1]))
            elif 'GAT' in self.gnn_opts:
                self.GNNs.append(GATConv(dims[i], dims[i + 1]))
            elif 'GATMan' in self.gnn_opts:
                self.GNNs.append(GATConvManual(dims[i], dims[i + 1]))
            else:
                assert False
            self.MLPs.append(nn.Linear(dims[i + 1], out_dim))
        self.GNNs = nn.ModuleList(self.GNNs).to(FLAGS.device)
        self.MLPs = nn.ModuleList(self.MLPs).to(FLAGS.device)
        self.act = nn.ReLU()

        if learn_embs:
            self.G = torch.nn.Parameter(
                        torch.randn((1, dims[1]), device=FLAGS.device),
                        requires_grad=True
                    )
            self.S = torch.nn.Parameter(
                        torch.randn((1, dims[1]), device=FLAGS.device),
                        requires_grad=True
                    )
            self.BD = torch.nn.Parameter(
                        torch.randn((1, dims[1]), device=FLAGS.device),
                        requires_grad=True
                    )
        else:
            self.G = torch.zeros((1, dims[1]), device=FLAGS.device)
            self.S = torch.zeros((1, dims[1]), device=FLAGS.device)
            self.BD = torch.zeros((1, dims[1]), device=FLAGS.device)

        self.layer_AGG_w_MLP = layer_AGG_w_MLP

    def __call__(self, X1, X2, edge_index1, edge_index2, nn_map=None, action_space_data=None, alpha1=None, alpha2=None):
        gnn_type = self.gnn_opts[0] # old code
        bypass_GSBD = gnn_type in ['GAT', 'GATMan', 'GCN']
        if not bypass_GSBD:
            n_bisets = 0 if action_space_data is None else len(action_space_data.pruned_bds)
            G = self.G.repeat(2, 1)
            if nn_map is not None and len(nn_map) == 0:
                S = None
            else:
                S = self.S.repeat(2, 1)
            if n_bisets == 0:
                BD = None
            else:
                BD = self.BD.repeat(2*n_bisets,1)

        # apply GNNs
        w = F.normalize(self.w, p=2, dim=0)
        X1s, X2s = [], []
        for i in range(self.n_layers):
            X1, X2 = self.GNNs[i](X1, edge_index1), self.GNNs[i](X2, edge_index2)
            if self.layer_AGG_w_MLP:
                X1, X2 = self.act(X1), self.act(X2)
                X1, X2 = self.MLPs[i](X1), self.MLPs[i](X2)
            X1s.append(X1)
            X2s.append(X2)

        X1s, X2s = torch.stack(X1s, dim=0), torch.stack(X2s, dim=0)
        X1 = torch.mm(w, X1s.view(w.shape[1], -1)).view(X1s.size(1),
                                                        X1s.size(2))  # np.dot(w,x1s)
        X2 = torch.mm(w, X2s.view(w.shape[1], -1)).view(X2s.size(1),
                                                        X2s.size(2))  # np.dot(w,x2s)


        if bypass_GSBD:
            G, S, BD_list = None, None, None
        else:
            # construct G, S, BD
            if G is None:
                G = self.G.repeat(2, 1)
            if S is None:
                S = self.S.repeat(2, 1)
            if BD is None:
                BD_list = [self.BD.repeat(2, 1) for _ in action_space_data.selected_bd_indices]
            else:
                # TODO: adapt this for multiple bidomains!
                BD_list = [BD[2 * bd_idx: 2 * bd_idx + 2] for bd_idx in action_space_data.selected_bd_indices]

        return X1, X2, G, S, BD_list