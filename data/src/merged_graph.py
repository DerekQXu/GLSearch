from __future__ import annotations

from options import opt
from torch_geometric.data import Data as PyGSingleGraphData
import torch
import re

"""
Reference: 
https://github.com/rusty1s/pytorch_geometric/blob/71edd874f6056942c7c1ebdae6854da34f68aeb7/torch_geometric/data/batch.py
"""


class MergedGraphData(PyGSingleGraphData):
    r"""A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    """

    def __init__(self, batch=None, **kwargs):
        super(MergedGraphData, self).__init__(**kwargs)
        self.batch = batch
        self.anchor_info = None

    @staticmethod
    def from_data_list(data_list:List[PyGSingleGraphData]) -> dict:  # merge
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        """

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = MergedGraphData()

        for key in keys:
            batch[key] = []
        batch.batch = []

        indices_list = []

        cumsum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long,
                                          device=opt.device))
            for key in data.keys:
                item = data[key]
                item = item + cumsum if __cumsum__(key, item) else item
                batch[key].append(item)

            indices_list.append((cumsum, cumsum + num_nodes))

            cumsum += num_nodes

        for key in keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key],
                                          device=opt.device)
            else:
                raise ValueError('Unsupported attribute type.')
        batch.batch = torch.cat(batch.batch, dim=-1)

        return {'merge': batch.contiguous(), 'ind_list': indices_list}

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    @staticmethod
    def to_data_list(batch_data, tensor):  # split
        assert type(batch_data) is dict
        assert 'merge' in batch_data and 'ind_list' in batch_data
        rtn = []
        for (start, end) in batch_data['ind_list']:
            # Each data point is a pair in our project.
            # use the `sp` attribute of batch_data['merge'] to further
            # extract a single graph from the graph pair.
            pair_data = tensor[start:end]
            rtn.append(pair_data)
        return rtn

def __cumsum__(key, value):
    # FIXME __cumsum__ is deprecated in our torch version. Source code: https://github.com/JonasSchult/pytorch_geometric_fork/blob/4f655fbe5e79af274143fe8915b635397ffeca2a/torch_geometric/data/data.py#L30
    if bool(re.search('(index|face)', key)):
        print("Warning: __cumsum__ is deprecated in our torch version. This code might not work as expected.")
    return bool(re.search('(index|face)', key))