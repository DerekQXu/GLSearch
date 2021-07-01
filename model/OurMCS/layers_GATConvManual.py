import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from copy import deepcopy

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class GATConvManual(MessagePassing):
    """
    GAT but you specify the attention
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConvManual, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, alpha):
        """"""
        self.alpha = torch.stack(tuple([alpha] * self.heads), dim=1)

        # self._alpha_iter = torch.stack(tuple([alpha] * self.heads), dim=1)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, edge_index[0], num_nodes)
        alpha = softmax(self.alpha, edge_index[0], num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
