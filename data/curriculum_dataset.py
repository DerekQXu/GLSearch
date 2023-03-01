from __future__ import annotations

import torch
import random
from .base_dataset import BaseDataset
from .src.containers import Dataset, Graph, GraphPair
from options import opt
from utils.stats import generate_stat_line

seed = random.Random(123)


class CurriculumDataset(BaseDataset):
    dataset: Dataset = None
    gid1gid2_list: torch.Tensor = None
    num_node_features: int = None
    opt = None

    def __init__(self, dataset: Dataset, gid1gid2_list, num_node_features: int):
        super(CurriculumDataset, self).__init__(opt=opt)
        self.dataset = dataset
        self.gid1gid2_list = gid1gid2_list
        self.num_node_features = num_node_features
        self.opt = opt
        self.name = dataset.name

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, index):
        return self.gid1gid2_list[index]

    @staticmethod
    def from_legacy_dataset(legacy_dataset: Dict[str, any]) -> CurriculumDataset:
        """
        Create a new CurriculumDataset from a legacy OurCocktailDataset.
        """
        new_dataset = CurriculumDataset(
            Dataset.from_legacy_dataset(legacy_dataset['dataset']),
            [],
            legacy_dataset['num_node_feat'])
        return new_dataset

    @staticmethod
    def merge(dataset_list: List[CurriculumDataset], num_pairs_list: List[int]) -> CurriculumDataset:
        """
        Merge a list of datasets
        """
        name = dataset_list[0].name + "_cur"  # the name is the name of the first dataset of the curriculum
        # For each dataset extract randomly <num_pairs> pairs and graphs
        gs_list, pairs_list = _get_filtered_pairs_and_gs_list(dataset_list, num_pairs_list)
        gs_cum, pairs_cum = _merge_gs_and_pairs(gs_list, pairs_list)

        # merge the remaining attributes
        num_node_feat = dataset_list[0].num_node_features
        gid1gid2_list = torch.tensor(list(pairs_cum.keys()), device=opt.device)
        dataset = Dataset(name, gs_cum, pairs_cum)
        return CurriculumDataset(dataset, gid1gid2_list, num_node_feat)

    def __str__(self):
        return self.dataset.__str__() + \
            generate_stat_line('Num node features', self.num_node_features)


def _get_filtered_pairs_and_gs_list(dataset_list: List[CurriculumDataset], num_pairs_list: List[int]) -> (
        List[List[Graph]], List[PairDict]):
    """For each dataset of the curriculum, randomly select <num_pairs> pairs and graphs"""
    pairs_list = []
    gs_list = []
    for cur_dataset, num_pairs in zip(dataset_list, num_pairs_list):
        pairs, gs = _filter_pair_list(cur_dataset, num_pairs)
        pairs_list.append(pairs)
        gs_list.append(gs)
    return gs_list, pairs_list


def _filter_pair_list(cur_dataset: CurriculumDataset, num_pairs: int) -> (PairDict, List[Graph]):
    """
    Randomly select <num_pairs> pairs and  graphs from a singsle dataset
    """
    for g in cur_dataset.dataset.graphs:
        g.graph['src'] = cur_dataset.dataset.name
        g.graph['orig_gid'] = g.graph['gid']

    if num_pairs < 0:
        pairs, gs = cur_dataset.dataset.pairs, cur_dataset.dataset.graphs
    else:
        # sample the pairs “randomly”
        pairs_unfiltered, gs_unfiltered = cur_dataset.dataset.pairs, cur_dataset.dataset.graphs
        pair_keys = seed.sample(list(pairs_unfiltered.keys()), num_pairs)

        # only keep the pairs in pair indices
        valid_gids = set()
        pairs_filtered_unoffset = {}
        for key in pair_keys:
            pairs_filtered_unoffset[key] = pairs_unfiltered[key]
            gid1, gid2 = key
            valid_gids.add(gid1)
            valid_gids.add(gid2)
        gs_filtered_unoffset = [g for g in gs_unfiltered if
                                g.graph['gid'] in valid_gids]

        # relabel the gids -> gid = 0, 1, 2, 3, ... for later
        pairs, gs = {}, []
        old_gid2new_gid, _ = _offset_graphs(gs, gs_filtered_unoffset, offset=0)
        _offset_pairs(pairs, pairs_filtered_unoffset, old_gid2new_gid)
    return pairs, gs


def _offset_graphs(gs_cum: List[Graph], gs: List[Graph], offset) -> Tuple[Dict[int, int], int]:
    """Relabel the graph gids -> gid = 0, 1, 2, 3, ... """
    old_gid2new_gid = {}
    for g in gs:
        old_gid2new_gid[g.gid()] = offset
        g.graph['gid'] = offset
        if 'glabel' in g.graph:
            del g.graph['glabel']
        offset += 1
    gs_cum.extend(gs)
    return old_gid2new_gid, offset


def _offset_pairs(pairs_cum, pairs, old_gid2new_gid) -> None:
    for key, val in pairs.items():
        gid1, gid2 = key
        pairs_cum[(old_gid2new_gid[gid1], old_gid2new_gid[gid2])] = val


def _merge_gs_and_pairs(gs_list: List[List[Graph]], pairs_list: List[PairDict]) -> (List[Graph], PairDict):
    """Merge the graphs and pairs from all datasets"""
    offset = 0
    gs_cum, pairs_cum = [], {}
    assert len(gs_list) == len(pairs_list)
    for gs, pairs in zip(gs_list, pairs_list):
        old_gid2new_gid, offset = _offset_graphs(gs_cum, gs, offset)
        _offset_pairs(pairs_cum, pairs, old_gid2new_gid)
    print('merged dataset')
    return gs_cum, pairs_cum
