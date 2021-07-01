from load_data import load_dataset
from node_feat import encode_node_features
from config import FLAGS
from torch.utils.data import Dataset as TorchDataset
import torch
from utils_our import get_flags_with_prefix_as_list
from utils import get_save_path, save, load
from os.path import join
from warnings import warn
from dataset import OurDataset
import random

class OurModelData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""

    def __init__(self, dataset, num_node_feat):
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

    def __init__(self, ourModelDataset_list, num_pairs_list):
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


def load_train_test_data():
    if FLAGS.dataset == 'cocktail':
        train_data = []
        test_data = []
        for curriculum in FLAGS.dataset_list:
            dataset_name_list = curriculum[0]
            assert type(curriculum[1]) is int
            train_data_list, test_data_list, num_pairs_list  = [], [], []
            for idx in range(len(dataset_name_list)):
                dataset_name, num_pairs = dataset_name_list[idx]
                train_data_elt, test_data_elt = \
                    load_single_train_test_data(dataset_name,
                                                FLAGS.align_metric,
                                                FLAGS.node_ordering)
                train_data_list.append(train_data_elt)
                test_data_list.append(test_data_elt)
                num_pairs_list.append(num_pairs)
            train_data.append(OurCocktailData(train_data_list, num_pairs_list))
            test_data.append(OurCocktailData(test_data_list, num_pairs_list))
    else:
        train_data, test_data = load_single_train_test_data(FLAGS.dataset,
                                                            FLAGS.align_metric,
                                                            FLAGS.node_ordering)
        train_data = [train_data]
        test_data = [test_data]

    return train_data, test_data


def load_single_train_test_data(dataset_name, align_metric, node_ordering):
    dir = join(get_save_path(), 'OurModelData')

    sfn = '{}_train_test_{}_{}_{}'.format(
        dataset_name, align_metric, node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')))
    '''
    sfn = '{}_train_test_{}_{}_{}{}{}'.format(
        FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')),
        _none_empty_else_underscore(FLAGS.filter_large_size),
        _none_empty_else_underscore(FLAGS.select_node_pair))
    '''
    tp = join(dir, sfn)
    # version option
    if dataset_name in ['aids700nef', 'linux', 'imdbmulti', 'alchemy']:
        if FLAGS.dataset_version != None:
            tp += '_{}'.format(FLAGS.dataset_version)
    tp += '_{}'.format(FLAGS.node_feats)
    rtn = load(tp)
    if rtn is not None:
        if len(rtn) == 0:
            raise ValueError('Weird empty loaded dict')
        train_data, test_data = rtn['train_data'], rtn['test_data']
    else:
        train_data, test_data = _load_train_test_data_helper(dataset_name, align_metric,
                                                             node_ordering)
        save({'train_data': train_data, 'test_data': test_data}, tp)
    if FLAGS.validation:
        all_spare_ratio = 1 - FLAGS.throw_away
        train_val_ratio = 0.6 * all_spare_ratio
        dataset = train_data.dataset
        dataset.tvt = 'all'
        if all_spare_ratio != 1:
            dataset_train, dataset_test, _ = dataset.tvt_split(
                [train_val_ratio, all_spare_ratio], ['train', 'validation', 'spare'],
                split_by=FLAGS.split_by)
        else:
            dataset_train, dataset_test = dataset.tvt_split(
                [train_val_ratio], ['train', 'validation'], split_by=FLAGS.split_by)
        assert train_data.num_node_feat == test_data.num_node_feat
        train_data = OurModelData(dataset_train, train_data.num_node_feat)
        test_data = OurModelData(dataset_test, test_data.num_node_feat)

    if FLAGS.filter_large_size is not None:
        print('truncating graphs...')
        train_data.truncate_large_graphs()
        test_data.truncate_large_graphs()

    if FLAGS.select_node_pair is not None:
        print('selecting node pair...')
        if FLAGS.select_node_pair[-1] == 'r':
            train_data.select_specific_for_debugging()
            test_data = train_data
        elif FLAGS.select_node_pair[-1] == 'e':
            test_data.select_specific_for_debugging()
            train_data = test_data
        else:
            assert False

    train_data.dataset.print_stats()
    test_data.dataset.print_stats()
    return train_data, test_data


def _none_empty_else_underscore(v):
    if v is None:
        return ''
    return '_{}'.format(v)


def _load_train_test_data_helper(dataset_name, align_metric, node_ordering):
    # TODO: possible tvt options list as well for combined cocktail datasets
    from dataset_config import get_dataset_conf
    from utils import format_str_list
    _, _, tvt_options, *_ = get_dataset_conf(dataset_name)
    tvt_options = format_str_list(tvt_options)
    # if 'test' not in tvt_options:
    #     tvt_options
    if tvt_options == 'all':
        dataset = load_dataset(dataset_name, 'all', align_metric,
                               node_ordering)
        dataset.print_stats()
        # Node feature encoding must be done at the entire dataset level.
        print('Encoding node features')
        dataset, num_node_feat = encode_node_features(dataset=dataset)
        print('Splitting dataset into train test')
        dataset_train, dataset_test = dataset.tvt_split(
            [FLAGS.train_test_ratio], ['train', 'test'], split_by=FLAGS.split_by)
    elif tvt_options == 'test':
        dataset_test = load_dataset(dataset_name, 'test', align_metric,
                                    node_ordering)
        dataset_test, num_node_feat = \
            encode_node_features(dataset=dataset_test)
        dataset_train = dataset_test  # just reused the test set in training
    elif tvt_options == 'train,test':
        dataset_test = load_dataset(dataset_name, 'test', align_metric,
                                    node_ordering)
        dataset_train = load_dataset(dataset_name, 'train', align_metric,
                                     node_ordering)
        dataset_train, num_node_feat_train = \
            encode_node_features(dataset=dataset_train)
        dataset_test, num_node_feat_test = \
            encode_node_features(dataset=dataset_test)
        if num_node_feat_train != num_node_feat_test:
            raise ValueError('num_node_feat_train != num_node_feat_test '
                             '{] != {}'.
                             format(num_node_feat_train, num_node_feat_test))
        num_node_feat = num_node_feat_train
    else:
        print(FLAGS.tvt_options)
        raise NotImplementedError()
    dataset_train.print_stats()
    dataset_test.print_stats()
    train_data = OurModelData(dataset_train, num_node_feat)
    test_data = OurModelData(dataset_test, num_node_feat)
    return train_data, test_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.sampler import SubsetRandomSampler
    from batch import BatchData

    # print(len(load_dataset(FLAGS.dataset).gs))
    data = OurModelData()
    print(len(data))
    # print('model_data.num_features', data.num_features)
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(dataset_size * 0.2)
    random.Random(123).shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader = DataLoader(data, batch_size=3, shuffle=True)
    print(len(loader.dataset))
    for i, batch_gids in enumerate(loader):
        print(i, batch_gids)
        batch_data = BatchData(batch_gids, data.dataset)
        print(batch_data)
        # print(i, batch_data, batch_data.num_graphs, len(loader.dataset))
        # print(batch_data.sp)
