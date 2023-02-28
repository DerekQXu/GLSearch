from __future__ import annotations

from options import opt
from .src.merged_graph import MergedGraphData
from .src.node_feat import encode_node_features
from .src.pair_processor import preproc_graph_pair
from torch_geometric.data import Data as PyGSingleGraphData
from torch_geometric.utils import to_undirected
import torch
import networkx as nx
from collections import defaultdict
from .src.containers.dataset import Dataset
from .src.containers.graph import Graph
from .src.containers.graph_pair import GraphPair
from utils.stats import generate_stat_line


class BatchData(object):
    """Mini-batch.

    We assume the following sequential model architecture: Merge --> Split.

        Merge: For efficiency, first merge graphs in a batch into a large graph.
            This is only done for the first several `NodeEmbedding` layers.

        Split: For flexibility, split the merged graph into individual pairs.
            The `gen_list_view_by_split` function should be called immediately
            after the last `NodeEmbedding` layer.
    """

    def __init__(self, batch_gids: torch.tensor, dataset: Dataset):
        self.dataset = dataset
        self.merge_data, self.pair_list = self._merge_into_one_graph(batch_gids)

    def __str__(self):
        return generate_stat_line("merge data", self.merge_data) + \
            generate_stat_line("pair list", [str(p) for p in self.pair_list])

    def _merge_into_one_graph(self, batch_gids: torch.tensor) -> (dict, List[GraphPair]):
        """
        Merge graphs in a batch into a single graph.
        :param batch_gids: A tensor of shape (batch_size, 2).
        """
        single_graph_list = []
        pair_list = []
        gids1 = batch_gids[:, 0]
        gids2 = batch_gids[:, 1]
        assert gids1.shape == gids2.shape
        for (gid1, gid2) in zip(gids1, gids2):
            self._preproc_gid_pair(gid1, gid2, single_graph_list, pair_list)
        assert len(pair_list) == gids1.shape[0] == gids2.shape[0]
        return MergedGraphData.from_data_list(single_graph_list), pair_list

    def _preproc_gid_pair(self, gid1: torch.tensor, gid2: torch.tensor, single_graph_list: List[PyGSingleGraphData],
                          pair_list: List[GraphPair]) -> void:
        gid1 = gid1.item()
        gid2 = gid2.item()
        assert gid1 - int(gid1) == 0
        assert gid2 - int(gid2) == 0
        gid1 = int(gid1)
        gid2 = int(gid2)
        g1: Graph = self.dataset.look_up_graph_by_gid(gid1)
        g2: Graph = self.dataset.look_up_graph_by_gid(gid2)
        pair: GraphPair = self.dataset.look_up_pair_by_gids(g1.gid(), g2.gid())
        preproc_g_list = preproc_graph_pair(g1, g2, pair)  # possibly combine
        this_single_graph_list = [_convert_nx_to_pyg_graph(g) for g in preproc_g_list]
        single_graph_list.extend(this_single_graph_list)
        pair.assign_g1_g2(g1, g2)
        pair_list.append(pair)


def create_edge_index(g):
    edge_index = torch.tensor(list(g.edges),
                              device=opt.device).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=g.number_of_nodes())
    return edge_index


def _convert_nx_to_pyg_graph(g: Graph) -> PyGSingleGraphData:
    """converts_a networkx graph to a PyGSingleGraphData."""
    # Reference: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/ppi.py
    if not isinstance(g, nx.Graph):
        raise ValueError('Input graphs must be undirected nx.Graph,'
                         ' NOT {}'.format(type(g)))
    edge_index = create_edge_index(g)
    data = PyGSingleGraphData(
        x=torch.tensor(g.init_x,
                       dtype=torch.float32,
                       # required by concat with LocalDegreeProfile()
                       device=opt.device),
        edge_index=edge_index,
        edge_attr=None,
        y=None)
    data, nf_dim = encode_node_features(pyg_single_g=data)
    assert data.is_undirected()
    assert data.x.shape[1] == nf_dim
    return data
