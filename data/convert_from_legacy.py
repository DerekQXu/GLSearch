import os
import re
from typing import Dict, List, Tuple
from legacy_glsearch_code.utils import load
from legacy_glsearch_code.dataset import OurDataset
from torch.utils.data import Dataset as TorchDataset
import random

klepto_folder = "klepto_files"
def main():
    dataset_list = [
        ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
    ]
    load_train_test_data(dataset_list)

class OurModelData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""

    def __init__(self, dataset:OurDataset, num_node_feat:int):
        self.dataset, self.num_node_feat = dataset, num_node_feat
        gid_pairs = list(self.dataset.pairs.keys())
        self.gid1gid2_list = torch.tensor(
            sorted(gid_pairs),
            device=FLAGS.device)  # takes a while to move to GPU

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, idx):
        return self.gid1gid2_list[idx]

    def get_pairs_as_list(self):
        return [self.dataset.look_up_pair_by_gids(gid1.item(), gid2.item())
                for (gid1, gid2) in self.gid1gid2_list]

    def truncate_large_graphs(self):
        gid_pairs = list(self.dataset.pairs.keys())
        if FLAGS.filter_large_size < 1:
            raise ValueError('Cannot filter graphs of size {} < 1'.format(
                FLAGS.filter_large_size))
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1)
            g2 = self.dataset.look_up_graph_by_gid(gid2)
            if g1.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size and \
                    g2.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

    def select_specific_for_debugging(self):
        gid_pairs = list(self.dataset.pairs.keys())
        gids_selected = FLAGS.select_node_pair[:-1].split('_')
        assert (len(gids_selected) == 2)
        gid1_selected, gid2_selected = int(gids_selected[0]), int(gids_selected[1])
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1).get_nxgraph()
            g2 = self.dataset.look_up_graph_by_gid(gid2).get_nxgraph()
            if g1.graph['gid'] == gid1_selected and g2.graph['gid'] == gid2_selected:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        FLAGS.select_node_pair = None  # for test
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

class OurCocktailData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""
    def __init__(self, ourModelDataset_list: List[OurModelData], num_pairs_list:int):
        self.seed = random.Random(123)

        gs_list, natts_list, eatts_list, pairs_list, tvt_list, \
        align_metric_list, node_ordering_list, glabel_list = \
            self._get_OurModelDataset_contents(ourModelDataset_list, num_pairs_list)

        gs_cum, pairs_cum = self._merge_gs_and_pairs(gs_list, pairs_list)

        # TODO: this implementation assumes all datasets share the following
        name, natts, eatts, tvt, align_metric, node_ordering, glabel = \
            self._get_dataset_metadata(
                natts_list, eatts_list, tvt_list, align_metric_list, node_ordering_list,
                glabel_list)

        self.num_node_feat = ourModelDataset_list[0].num_node_feat
        self.gid1gid2_list = torch.tensor(list(pairs_cum.keys()), device=FLAGS.device)

        loaded_dict = None
        self.dataset = OurDataset(name, gs_cum, natts, eatts, pairs_cum, tvt, align_metric,
                                  node_ordering, glabel, loaded_dict)

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, idx):
        return self.gid1gid2_list[idx]

    def _get_OurModelDataset_contents(self, ourModelDataset_list, num_pairs_list):
        pairs_list, gs_list = self._get_filtered_pairs_and_gs_list(ourModelDataset_list,
                                                                   num_pairs_list)
        # name = 'cocktail:' + ';'.join(FLAGS.dataset)
        natts_list = [ourModelDataset.dataset.natts
                      for ourModelDataset in ourModelDataset_list]
        eatts_list = [ourModelDataset.dataset.eatts
                      for ourModelDataset in ourModelDataset_list]
        tvt_list = [ourModelDataset.dataset.tvt
                    for ourModelDataset in ourModelDataset_list]
        align_metric_list = [ourModelDataset.dataset.align_metric
                             for ourModelDataset in ourModelDataset_list]
        node_ordering_list = [ourModelDataset.dataset.node_ordering
                              for ourModelDataset in ourModelDataset_list]
        glabel_list = [ourModelDataset.dataset.glabel
                       for ourModelDataset in ourModelDataset_list]
        return gs_list, natts_list, eatts_list, pairs_list, tvt_list, \
               align_metric_list, node_ordering_list, glabel_list

    def _get_filtered_pairs_and_gs_list(self, ourModelDataset_list, num_pairs_list):
        pairs_list = []
        gs_list = []
        for ourModelDataset, num_pairs in zip(ourModelDataset_list, num_pairs_list):
            pairs, gs = self._filter_pair_list(ourModelDataset, num_pairs)
            pairs_list.append(pairs)
            gs_list.append(gs)
        return pairs_list, gs_list

    def _filter_pair_list(self, ourModelDataset, num_pairs):
        for g in ourModelDataset.dataset.gs:
            g.nxgraph.graph['src'] = ourModelDataset.dataset.name
            g.nxgraph.graph['orig_gid'] = g.nxgraph.graph['gid']

        if num_pairs < 0:
            pairs, gs = ourModelDataset.dataset.pairs, ourModelDataset.dataset.gs
        else:
            # sample the pairs "randomly"
            pairs_unfiltered, gs_unfiltered = ourModelDataset.dataset.pairs, ourModelDataset.dataset.gs
            pair_keys = self.seed.sample(list(pairs_unfiltered.keys()), num_pairs)

            # only keep the pairs in pair indices
            valid_gids = set()
            pairs_filtered_unoffset = {}
            for key in pair_keys:
                pairs_filtered_unoffset[key] = pairs_unfiltered[key]
                gid1, gid2 = key
                valid_gids.add(gid1)
                valid_gids.add(gid2)
            gs_filtered_unoffset = [g for g in gs_unfiltered if
                                    g.nxgraph.graph['gid'] in valid_gids]

            # relabel the gids -> gid = 0, 1, 2, 3, ... for later
            pairs, gs = {}, []
            old_gid2new_gid, _ = self._offset_graphs(gs, gs_filtered_unoffset, offset=0)
            self._offset_pairs(pairs, pairs_filtered_unoffset, old_gid2new_gid)
        return pairs, gs

    def _merge_gs_and_pairs(self, gs_list, pairs_list):
        offset = 0
        gs_cum, pairs_cum = [], {}
        assert len(gs_list) == len(pairs_list)
        for gs, pairs in zip(gs_list, pairs_list):
            old_gid2new_gid, offset = self._offset_graphs(gs_cum, gs, offset)
            self._offset_pairs(pairs_cum, pairs, old_gid2new_gid)
        print('merged dataset')
        return gs_cum, pairs_cum

    def _offset_graphs(self, gs_cum, gs, offset):
        old_gid2new_gid = {}
        for g in gs:
            old_gid2new_gid[g.gid()] = offset
            g.nxgraph.graph['gid'] = offset
            if 'glabel' in g.nxgraph.graph:
                del g.nxgraph.graph['glabel']
            offset += 1
        gs_cum.extend(gs)
        return old_gid2new_gid, offset

    def _offset_pairs(self, pairs_cum, pairs, old_gid2new_gid):
        for key, val in pairs.items():
            gid1, gid2 = key
            pairs_cum[(old_gid2new_gid[gid1], old_gid2new_gid[gid2])] = val

    def _get_dataset_metadata(self, natts_list, eatts_list, tvt_list,
                              align_metric_list, node_ordering_list, glabel_list):
        name = 'cocktail_temp'
        natts = self._merge_metadata(natts_list)
        eatts = self._check_content_same_and_return_first_element(eatts_list)
        tvt = self._check_content_same_and_return_first_element(tvt_list)
        align_metric = self._check_content_same_and_return_first_element(align_metric_list)
        node_ordering = self._check_content_same_and_return_first_element(node_ordering_list)
        # glabel = self._check_content_same_and_return_first_element(glabel_list)
        glabel = None
        return name, natts, eatts, tvt, align_metric, node_ordering, glabel

    def _merge_metadata(self, metadata_list):
        metadata_merged = set()
        for metadata in metadata_list:
            for elt in metadata:
                metadata_merged.add(elt)
        return list(metadata_merged)

    def _check_content_same_and_return_first_element(self, li):
        assert len(li) > 0
        elt_first = li[0]
        for k, elt in enumerate(li):
            # assert elt == elt_first
            pass
        return elt_first

    ######################################################################################

    def get_pairs_as_list(self):
        return [self.dataset.look_up_pair_by_gids(gid1.item(), gid2.item())
                for (gid1, gid2) in self.gid1gid2_list]

    def truncate_large_graphs(self):
        gid_pairs = list(self.dataset.pairs.keys())
        if FLAGS.filter_large_size < 1:
            raise ValueError('Cannot filter graphs of size {} < 1'.format(
                FLAGS.filter_large_size))
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1)
            g2 = self.dataset.look_up_graph_by_gid(gid2)
            if g1.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size and \
                    g2.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

    def select_specific_for_debugging(self):
        gid_pairs = list(self.dataset.pairs.keys())
        gids_selected = FLAGS.select_node_pair[:-1].split('_')
        assert (len(gids_selected) == 2)
        gid1_selected, gid2_selected = int(gids_selected[0]), int(gids_selected[1])
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1).get_nxgraph()
            g2 = self.dataset.look_up_graph_by_gid(gid2).get_nxgraph()
            if g1.graph['gid'] == gid1_selected and g2.graph['gid'] == gid2_selected:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        FLAGS.select_node_pair = None  # for test
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU



def load_train_test_data(dataset_list: List[OurCocktailData]) -> Tuple[List[OurCocktailData], List[OurCocktailData]]:
    """
    :param dataset_list: List of curriculums. Each curriculum is a list of (dataset_name, num_pairs) tuples. Each tuple represents a dataset of graph, and the number of graphs to use from that dataset
    """
    train_data = []
    test_data = []
    for curriculum in dataset_list:
        dataset_name_list = curriculum[0]
        assert type(curriculum[1]) is int
        train_data_list, test_data_list, num_pairs_list  = [], [], []
        for single_dataset_tuple in dataset_name_list:
            dataset_name, num_pairs = single_dataset_tuple
            train_data_elt, test_data_elt = load_single_train_test_data(dataset_name)
            train_data_list.append(train_data_elt)
            test_data_list.append(test_data_elt)
            num_pairs_list.append(num_pairs)
        train_data.append(OurCocktailData(train_data_list, num_pairs_list))
        test_data.append(OurCocktailData(test_data_list, num_pairs_list))

    return train_data, test_data


def load_single_train_test_data(dataset_name: str) -> Tuple[OurModelData, OurModelData]:

    # search file in folder
    full_name = None
    for k in os.listdir(klepto_folder):
        if re.match(dataset_name, k):
            full_name = k
            break
    if full_name is None:
        raise ValueError('No file found for dataset {}'.format(dataset_name))

    # laod klepto file
    tp = os.path.join(klepto_folder, full_name)
    tp = os.path.abspath(tp)
    print('Loading data from {}'.format(tp))
    rtn = load(tp)
    if rtn is not None:
        if len(rtn) == 0:
            # FIXME reset:
            # raise ValueError('Weird empty loaded dict')
            print('Weird empty loaded dict')
            return [],[]
        train_data, test_data = rtn['train_data'], rtn['test_data']
    else:
        raise ValueError('No data loaded')

    train_data.dataset.print_stats()
    test_data.dataset.print_stats()
    return train_data, test_data





if __name__ == '__main__':
    main()