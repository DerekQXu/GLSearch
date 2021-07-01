from config import FLAGS
from batch import create_edge_index
from torch_geometric.nn import MessagePassing
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb


def init_linear(m, method='uniform'):
    assert isinstance(m, nn.Linear)
    if method == 'uniform':
        nn.init.xavier_uniform_(m.weight)
    elif method == 'normal':
        nn.init.xavier_normal_(m.weight)
    else:
        raise ValueError("Method cannot be " + method)
    if m.bias is not None:
        m.bias.data.fill_(0.01)


class PCAModel(torch.nn.Module):
    def __init__(self, desc,
                 tao1=1, tao2=1, sinkhorn_iters=5, sinkhorn_eps=1e-6, aff_max=85):
        super(PCAModel, self).__init__()
        # check desc: input -> GConv 1 output -> CrossConv output -> GConv 2 output
        assert (len(desc) == 4)
        # GConv 1
        self.gconv1_1 = GConv(desc[0], desc[1])
        self.gconv1_2 = GConv(desc[0], desc[1])
        # Affinity 1
        self.aff1 = PCAAffinity(desc[1], desc[1], tao=tao1, clamp_max=aff_max)
        # CrossConv
        self.cconv1_1 = CrossConv(desc[1], desc[1], desc[2])
        self.cconv1_2 = CrossConv(desc[1], desc[1], desc[2])
        # GConv 2
        self.gconv2_1 = GConv(desc[2], desc[3])
        self.gconv2_2 = GConv(desc[2], desc[3])
        # Affinity 2
        self.aff2 = PCAAffinity(desc[3], desc[3], tao=tao2, clamp_max=aff_max)
        # sinkhorn: no para, only need one
        self.sinkhorn = Sinkhorn_v3(sinkhorn_iters, eps=sinkhorn_eps)

    def init_weight(self):
        self.gconv1_1.init_weight()
        self.gconv1_2.init_weight()
        self.cconv1_1.init_weight()
        self.cconv1_2.init_weight()
        self.gconv2_1.init_weight()
        self.gconv2_1.init_weight()
        self.aff1.init_weight()
        self.aff2.init_weight()

    def __call__(self, pair, x1, x2):
        edge1 = create_edge_index(pair.g1.get_nxgraph())
        edge2 = create_edge_index(pair.g2.get_nxgraph())
        mne = self._forward_pair(x1, edge1, x2, edge2)
        return mne

    def forward(self, ins, batch_data, model):
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        x1x2t_li = []
        for pair in pair_list:
            x1, x2 = pair.g1.x, pair.g2.x
            edge1 = create_edge_index(pair.g1.get_nxgraph())
            edge2 = create_edge_index(pair.g2.get_nxgraph())
            mne = self._forward_pair(x1, edge1, x2, edge2)
            x1x2t_li.append(mne)
            pair.assign_y_pred_list([mne for _ in range(FLAGS.n_outputs)],
                                    format='torch_{}'.format(FLAGS.device))
        return x1x2t_li

    def _forward_pair(self, x1, edge1, x2, edge2, coef=None):
        x1_1 = self.gconv1_1(x1, edge1)
        x2_1 = self.gconv1_2(x2, edge2)
        # Affinity 1
        M = self.aff1(x1_1, x2_1, coef=coef)
        if torch.any(torch.isnan(M)) or torch.any(torch.isinf(M)):
            pdb.set_trace()
        # Sinkhorn 1
        M_sh = self.sinkhorn(M)
        if torch.any(torch.isnan(M_sh)) or torch.any(torch.isinf(M_sh)):
            pdb.set_trace()
        # print('\n\n1:', file=sys.stderr)
        # print(M, file=sys.stderr)
        # print(M_sh, file=sys.stderr)
        # print('\n\n', file=sys.stderr)   # NOTE: for debug
        # CrossConv
        x1_2 = self.cconv1_1(x1_1, x2_1, M_sh)
        x2_2 = self.cconv1_2(x2_1, x1_1, M_sh.transpose(0, 1))
        # GConv 2
        x1_3 = self.gconv2_1(x1_2, edge1)
        x2_3 = self.gconv2_2(x2_2, edge2)
        # Affinity 2
        M = self.aff2(x1_3, x2_3, coef=coef)
        if torch.any(torch.isnan(M)) or torch.any(torch.isinf(M)):
            pdb.set_trace()
        # Sinkhorn 2
        M_sh = self.sinkhorn(M)
        if torch.any(torch.isnan(M_sh)) or torch.any(torch.isinf(M_sh)):
            pdb.set_trace()
        # print('\n\n2:', file=sys.stderr)
        # print(M, file=sys.stderr)
        # print(M_sh, file=sys.stderr)
        # print('\n\n', file=sys.stderr)   # NOTE: for debug
        return M_sh


class GConv(MessagePassing):
    def __init__(self, in_ch, out_ch, act='ReLU'):
        super(GConv, self).__init__(aggr='mean')
        self.fmsg = nn.Linear(in_ch, out_ch).float()
        self.fnode = nn.Linear(in_ch, out_ch).float()
        if act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError("act cannot be " + str(act))

    def init_weight(self):
        init_linear(self.fmsg)
        init_linear(self.fnode)

    def forward(self, x, edge_index):
        # pdb.set_trace()
        y = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
            pdb.set_trace()
        return y

    def message(self, x_j):
        # pdb.set_trace()
        return self.act(self.fmsg(x_j))

    def update(self, aggr_out, x):
        x = self.act(self.fnode(x))
        return x + aggr_out


class Sinkhorn_v3(nn.Module):
    def __init__(self, iters, eps=1e-6):
        super(Sinkhorn_v3, self).__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, M):
        for iter in range(self.iters):
            M = F.normalize(M, p=1, dim=1, eps=self.eps)
            M = F.normalize(M, p=1, dim=0, eps=self.eps)
        return M


class Sinkhorn_v4(nn.Module):
    def __init__(self, iters, eps=1e-6, converge_eps=1e-2):
        super(Sinkhorn_v4, self).__init__()
        self.iters = iters
        self.eps = eps
        self.converge_eps = eps

    def forward(self, M):
        for iter in range(self.iters):
            M_old = M
            M = F.normalize(M, p=1, dim=1, eps=self.eps)
            M = F.normalize(M, p=1, dim=0, eps=self.eps)
            if torch.sum(torch.abs(M - M_old)) < self.converge_eps:
                break
        # print("Converge at %d" % iter, file=sys.stderr)
        return M


class PCAAffinity(nn.Module):
    def __init__(self, dim1, dim2, tao, clamp_max):
        super(PCAAffinity, self).__init__()
        self.tao = tao
        self.clamp_max = clamp_max
        self.A = nn.Linear(dim2, dim1, bias=False).float()

    def init_weight(self):
        init_linear(self.A, method='normal')

    def forward(self, f1, f2, coef=None):
        # f1: n1 * m1, f2: n2 * m2
        # pdb.set_trace()
        # print("Before affinity", file=sys.stderr)
        # print(f1, f2, file=sys.stderr)
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)
        # print("After norm", file=sys.stderr)
        # print(f1, f2, file=sys.stderr)
        f = self.A(f2)  # n2 * m1
        f = torch.mm(f1, f.transpose(0, 1))
        f /= f1.shape[1] * f2.shape[1]  # normalize (does it matter?)
        f = f / self.tao
        # clip
        '''if (g >= self.clamp_max).sum() >= g.view(-1).size()[0] * 0.3:
            self.tao = float(f.max()) / self.clamp_max
            g = f / self.tao
            print("Increasing tao to %f" % self.tao)
        else:
            f = torch.clamp(g, -float('inf'), self.clamp_max)'''
        if torch.any(f > self.clamp_max):
            print('Clamp')
            f = torch.clamp(f, -float('inf'), self.clamp_max)
        f = torch.exp(f)
        if coef is not None:
            f = f * coef
        # print('A.max = %f' % self.A.weight.max())
        # print('A', file=sys.stderr)
        # print(self.A.weight, file=sys.stderr)
        # print('f', file=sys.stderr)
        # print(f, file=sys.stderr)
        # print('\n\n', file=sys.stderr)
        if torch.any(torch.isnan(f)) or torch.any(torch.isinf(f)):
            pdb.set_trace()
        return f


class CrossConv(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, act='ReLU'):
        super(CrossConv, self).__init__()
        self.in_ch1 = in_ch1
        self.in_ch2 = in_ch2
        self.out_ch = out_ch
        self.fc = nn.Linear(in_ch1 + in_ch2, out_ch).float()
        if act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError("act cannot be " + str(act))

    def init_weight(self):
        init_linear(self.fc)

    def forward(self, f1, f2, S):
        assert (f1.shape[1] == self.in_ch1)  # n1, m1
        assert (f2.shape[1] == self.in_ch2)  # n2, m2
        assert (S.shape[0] == f1.shape[0] and S.shape[1] == f2.shape[0])  # n1, n2
        f2 = torch.mm(S, f2)  # n1, m2
        f = torch.cat((f1, f2), dim=1)  # n1, m1 + m2
        f = self.fc(f)  # n1, out_ch
        f = self.act(f)
        return f


if __name__ == '__main__':
    x = np.array([[0.2, 0.9], [0.9, 0.3], [0.5, 0.5]], dtype=np.float32)
    xt = torch.from_numpy(x)
    my = Sinkhorn_v3(10)(xt)
    print('my')
    print(my, my.sum(0), my.sum(1))
    # his = sinkhorn(x, 10)
    # print('his')
    # print(his, his.sum(axis=0), his.sum(axis=1))
    '''print(GConv(34, 100))
    print(Sinkhorn(5))
    print(PCAAffinity(83, 21, 1))
    print(CrossConv(34, 34, 100))'''
