from utils_our import get_flags_with_prefix_as_list
from torch_geometric.transforms import LocalDegreeProfile
from utils import assert_0_based_nids
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np


class NodeFeatureEncoder(object):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        if node_feat_name is None:
            return
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(sorted(inputs_set))}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        self.oe = OneHotEncoder(categories='auto').fit(
            np.array(sorted(self.feat_idx_dic.values())).reshape(-1, 1))

    def encode(self, g):
        assert_0_based_nids(g)  # must be [0, 1, 2, ..., N - 1]
        if self.node_feat_name is None:
            return np.array([[1] for n in sorted(g.nodes())])  # NOTE: this will no longer be called now?
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in sorted(g.nodes())]  # sort nids just to make sure
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


def encode_node_features(dataset=None, pyg_single_g=None):
    if dataset:
        assert pyg_single_g is None
        input_dim = 0
    else:
        assert pyg_single_g is not None
        input_dim = pyg_single_g.x.shape[1]
    node_feat_encoders = get_flags_with_prefix_as_list('node_fe')
    if 'one_hot' not in node_feat_encoders:
        raise ValueError('Must have one hot node feature encoder!')
    for nfe in node_feat_encoders:
        if nfe == 'one_hot':
            if dataset:
                input_dim = _one_hot_encode(dataset, input_dim)
        elif nfe == 'local_degree_profile':
            input_dim += 5
            if pyg_single_g:
                pyg_single_g = LocalDegreeProfile()(pyg_single_g)
        else:
            raise ValueError('Unknown node feature encoder {}'.format(nfe))
    if input_dim <= 0:
        raise ValueError('Must have at least one node feature encoder '
                         'so that input_dim > 0')
    if dataset:
        return dataset, input_dim
    else:
        return pyg_single_g, input_dim


def _one_hot_encode(dataset, input_dim):
    gs = [g.get_nxgraph() for g in dataset.gs] # TODO: encode image's complete graph

    from config import FLAGS
    natts = FLAGS.node_feats.split(',')
    natts = [] if natts == ['None'] else natts #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # if len(dataset.natts) > 1:
    if len(natts) > 1:
        node_feat_name = None
        raise ValueError('TODO: handle multiple node features')
    # elif len(dataset.natts) == 1:
    #     node_feat_name = dataset.natts[0]
    elif len(natts) == 1:
        node_feat_name = natts[0]
    else:
        #if no node feat return 1
        for g in gs:
            g.init_x = np.ones((nx.number_of_nodes(g), 1))
        return 1
    nfe = NodeFeatureEncoder(gs, node_feat_name)
    for g in gs:
        x = nfe.encode(g)
        g.init_x = x # assign the initial features
    input_dim += nfe.input_dim()
    return input_dim

from config import FLAGS
# TODO: THIS IS DUPLICATED CODE, WE EVENTUALLY SHOULD MERGE IT WITH ONE_HOT_ENCODE!
def obtain_nfe_feat_idx_div(dataset, natts = FLAGS.node_feats_for_mcs):
    gs = [g.get_nxgraph() for g in dataset.gs] # TODO: encode image's complete graph
    natts = [] if (natts == ['None'] or natts == 'None') else natts #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # if len(dataset.natts) > 1:
    if len(natts) > 1:
        node_feat_name = None
        raise ValueError('TODO: handle multiple node features')
    # elif len(dataset.natts) == 1:
    #     node_feat_name = dataset.natts[0]
    elif len(natts) == 1:
        node_feat_name = natts[0]
    else:
        return {}

    nfe = NodeFeatureEncoder(gs, node_feat_name)
    return nfe.feat_idx_dic

if __name__ == '__main__':
    from dataset import load_dataset

    encode_node_features(load_dataset('aids700nef'))
