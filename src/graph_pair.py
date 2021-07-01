from dist_sim_converter import normalize_ds_score, dist_to_sim, sim_to_dist
import numpy as np
from scipy import sparse
from graph import HierarchicalGraph


# from scipy.sparse import csc_matrix TODO: may use this to optimize


class GraphPair(object):
    def __init__(self, y_true_dict_list=None, ds_true=None, g1=None, g2=None,
                 check_1_1=False, running_time=None, edge_types=None,
                 has_true_matching=True):
        if running_time is not None:
            self.running_time = running_time
        if y_true_dict_list is not None:
            self._check_y_true_dict_list(y_true_dict_list)
        if ds_true is not None:
            if type(ds_true) is not int or ds_true < 0:
                raise ValueError('ds_true must be a non-negative int {}'.
                                 format(ds_true))
            self.__ds_true = ds_true
        if has_true_matching:
            if y_true_dict_list is None and ds_true is None:
                raise ValueError('Must provide at least one of alignment and score')
        self.y_true_dict_list = y_true_dict_list
        self.y_true_mat_list = None
        if g1:  # do not assign g1 to self (so we can save disk space)
            self._check_nids('row', g1)
        if g2:
            self._check_nids('col', g2)
        if check_1_1:
            self._check_one_to_one(g1, g2)
        if edge_types is not None:
            self.edge_types = edge_types

    def has_y_true_dict_list(self):
        return self.y_true_dict_list is not None

    def get_pair_id(self):
        self._check_g1_g2()
        return (self.g1.gid(), self.g2.gid())

    def assign_y_true_dict_list(self, y_true_dict_list):
        self._check_y_true_dict_list(y_true_dict_list)
        self.y_true_dict_list = y_true_dict_list
        self.y_true_mat_list = None

    def _check_y_true_dict_list(self, y_true_dict_list):
        if type(y_true_dict_list) is not list or not y_true_dict_list:
            raise ValueError('y_dict_list must be a non-empty list')
        for y_dict in y_true_dict_list:
            if type(y_dict) is not dict:
                raise ValueError('y_dict_list must be a list of dicts')

    def assign_g1_g2(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        if type(g1) is HierarchicalGraph or type(g2) is HierarchicalGraph:
            self.m = sum([g.get_nxgraph().number_of_nodes() for g in g1.get_nxgraph()])
            self.n = sum([g.get_nxgraph().number_of_nodes() for g in g2.get_nxgraph()])
        else:
            self.m = self.g1.get_nxgraph().number_of_nodes()
            self.n = self.g2.get_nxgraph().number_of_nodes()
        if hasattr(self, 'y_pred_mat_list'):
            self._check_shape(self.y_pred_mat_list)

    def get_g1_g2(self, need_networkx=False):
        self._check_g1_g2()
        if need_networkx:
            return self.g1.get_nxgraph(), self.g2.get_nxgraph()
        else:
            return self.g1, self.g2

    def _check_g1_g2(self):
        if not hasattr(self, 'g1') or not hasattr(self, 'g2'):
            raise ValueError('Must call assign_g1_g2 before calling this')

    def get_y_true_list_mat_view(self, format):
        self._check_g1_g2()
        if not self.y_true_mat_list:
            y_list_mat = [self._convert_nn_map_dict_to_matching_mat(nn_map, True)
                          for nn_map in self.y_true_dict_list]
            for y_list_match_mat, y_dict in zip(
                    y_list_mat, self.y_true_dict_list):
                for node_g1, node_g2 in y_dict.items():
                    if node_g2 != -1:
                        y_list_match_mat[node_g1][node_g2] = 1
            # self.y_list_mat_view = list(map(csc_matrix, y_list_mat))
            self.y_true_mat_list = y_list_mat
        assert self.y_true_mat_list and \
               len(self.y_true_mat_list) == len(self.y_true_dict_list)
        return [_convert_to(x, format) for x in self.y_true_mat_list]

    def _convert_nn_map_dict_to_matching_mat(self, nn_map, check_1_1):
        mat = np.zeros((self.m, self.n))
        assert type(nn_map) is dict
        if check_1_1:
            if len(nn_map.keys()) != len(set(nn_map.values())):
                raise ValueError('nn_map not 1-1 {}'.format(nn_map))
        for node_g1, node_g2 in nn_map.items():
            if node_g2 != -1:
                mat[node_g1][node_g2] = 1
        return mat

    def get_y_true_list_dict_view(self):
        assert self.y_true_dict_list
        return self.y_true_dict_list

    # This function can be called multiple times. Will extend.
    def assign_y_pred_list(self, y_pred_mat_list, format, timestamps=None):
        if 'dict' in format:  # convert nn_map dict to mat
            y_pred_mat_list = [self._convert_nn_map_dict_to_matching_mat(d, True)
                               for d in y_pred_mat_list]
        if not hasattr(self, 'y_pred_mat_list'):
            self.y_pred_mat_list = []
        self.y_pred_mat_list.extend(self._assign_pred_list_helper(
            y_pred_mat_list, format))
        if timestamps is not None:
            assert type(timestamps) is list and len(timestamps) == len(y_pred_mat_list)
            if not hasattr(self, 'y_pred_mat_timestamp_list'):
                self.y_pred_mat_timestamp_list = []
            self.y_pred_mat_timestamp_list.extend(timestamps)

    # This function can be called multiple times. Will extend.
    def assign_tree_pred_list(self, tree_pred_mat_list, format, timestamps=None):
        if 'dict' in format:  # convert nn_map dict to mat
            tree_pred_mat_list = [self._convert_nn_map_dict_to_matching_mat(d, True)
                                  for d in tree_pred_mat_list]
        if not hasattr(self, 'tree_pred_mat_list'):
            self.tree_pred_mat_list = []
        self.tree_pred_mat_list.extend(self._assign_pred_list_helper(
            tree_pred_mat_list, format))
        if timestamps is not None:
            assert type(timestamps) is list and len(timestamps) == len(tree_pred_mat_list)
            if not hasattr(self, 'tree_pred_mat_timestamp_list'):
                self.tree_pred_mat_timestamp_list = []
            self.tree_pred_mat_timestamp_list.extend(timestamps)

    def _assign_pred_list_helper(self, pred_mat_list, format):
        pred_mat_list_final = []
        self._check_g1_g2()
        if type(pred_mat_list) is not list or not pred_mat_list:
            raise ValueError('Must pass a non-empty list')
        self._check_shape(pred_mat_list)
        for pred_mat in pred_mat_list:
            pred_mat = _convert_to(pred_mat, format)
            # x = type(pred_mat)
            pred_mat_list_final.append(pred_mat)
        return pred_mat_list_final

    def _check_shape(self, y_pred_mat_list):
        for y_pred_mat in y_pred_mat_list:
            if y_pred_mat.shape != (self.m, self.n):
                raise ValueError('Shape mismatch! y_pred_mat shape {}; '
                                 'm, n: {} by {}'.
                                 format(y_pred_mat.shape, self.m, self.n))

    def get_y_pred_list_mat_view(self, format):
        return self._get_pred_list_mat_view_helper(format, 'y')

    def get_tree_pred_list_mat_view(self, format):
        return self._get_pred_list_mat_view_helper(format, 'tree')

    def _get_pred_list_mat_view_helper(self, format, tag):
        if tag == 'y':
            check = 'y_pred_mat_list'
        elif tag == 'tree':
            check = 'tree_pred_mat_list'
        else:
            assert False
        if not hasattr(self, check):
            raise ValueError('Must call assign_{} before calling this'.format(
                check))
        timestamp_list = None
        if tag == 'y':
            mat_list = self.y_pred_mat_list
            if hasattr(self, 'y_pred_mat_timestamp_list'):
                timestamp_list = self.y_pred_mat_timestamp_list
        elif tag == 'tree':
            mat_list = self.tree_pred_mat_list
            if hasattr(self, 'tree_pred_mat_timestamp_list'):
                timestamp_list = self.tree_pred_mat_timestamp_list
        else:
            assert False
        assert mat_list
        rtn = [_convert_to(y_pred_mat, format) for y_pred_mat in mat_list]
        assert rtn
        if timestamp_list is not None:
            assert len(timestamp_list) == len(rtn)
            rtn = [(x, timestamp_list[i]) for i, x in enumerate(rtn)] # return timestamp as a second entry
        return rtn

    def get_edge_types(self):
        return self.edge_types

    def get_ds_true(self, ds_norm, dos_true, dos_pred, ds_kernel):
        """
        May need to perform normalization + dist <--> sim transformation.
        According to dos_pred!
        """
        if not hasattr(self, '_GraphPair__ds_true'):
            raise ValueError('This graph pair does not have ds_true')
        self._check_g1_g2()
        rtn = self.__ds_true
        if ds_norm:
            rtn = normalize_ds_score(rtn, self.g1, self.g2)
        if dos_true != dos_pred:
            if dos_true == 'dist':
                assert dos_pred == 'sim'
                rtn = dist_to_sim(rtn, ds_kernel)
            elif dos_true == 'sim':
                assert dos_pred == 'dist'
                rtn = sim_to_dist(rtn, ds_kernel)
            elif dos_true == 'random':
                pass  # doesn't matter
            else:
                assert False
        return rtn

    def get_ds_true_link_pred(self):
        return self._GraphPair__ds_true

    def assign_ds_pred(self, ds_pred):
        # if type(ds_pred) is not float:
        #     raise ValueError('Must pass a float')
        ds_pred = _convert_to(ds_pred, format='numpy')
        self.ds_pred = ds_pred

    def assign_link_pred(self, link_pred):
        if link_pred.shape[0] == 1:
            self.link_pred = link_pred.item()
        else:
            self.link_pred = link_pred.detach().cpu().numpy()

    def get_link_pred(self):
        if not hasattr(self, 'link_pred'):
            raise ValueError('Must call assign_link_pred before calling this')
        return self.link_pred

    def has_alignment_true_pred(self):
        return hasattr(self, 'y_true_dict_list') and self.has_alignment_pred()

    def has_ds_score_true_pred(self):
        return hasattr(self, '_GraphPair__ds_true') and hasattr(self, 'ds_pred')

    def has_alignment_pred(self):
        return hasattr(self, 'y_pred_mat_list')

    def add_match_eval_result(self, match_eval_result):
        # if hasattr(self, 'match_eval_result'):
        #     raise NotImplementedError()  # TODO: may do eval multiple times, e.g. val, test
        self.match_eval_result = match_eval_result

    def get_xs(self):
        self._check_g1_g2()
        if not hasattr(self.g1, 'x') or not hasattr(self.g2, 'x'):
            raise ValueError('Has not assigned x to the graph pair\'s g1 and g2')
        return _convert_to(self.g1.x, 'numpy'), _convert_to(self.g2.x, 'numpy')

    def assign_pred_time(self, duration):
        self.duration = duration

    def get_pred_time(self):
        return self.duration

    def get_true_time(self):
        if hasattr(self, 'running_time'):
            return self.running_time
        assert False

    def _check_nids(self, rc, g):
        assert g is not None
        if g is not None and self.y_true_dict_list:
            for y_dict in self.y_true_dict_list:
                if rc == 'r':
                    nids = y_dict.keys()
                elif rc == 'c':
                    nids = y_dict.values()
                else:
                    assert False
                for id in nids:
                    nn = g.get_nxgraph().number_of_nodes()
                    if id < 0 or id >= nn:
                        raise ValueError('Wrong nid {}; Total #nodes {}'.
                                         format(id, nn))

    def _check_one_to_one(self, g1, g2):
        if g1 is None or g2 is None:
            raise ValueError('Cannot check the one-to-one constraint '
                             'if g1 or g2 is None {} {}'.format(g1, g2))
        # TODO: check one-to-one
        raise NotImplementedError()

    def shrink_space_for_save(self): # TODO: check what causes 300 MB
        import torch
        self.__dict__.pop('g1', None)
        self.__dict__.pop('g2', None)
        self.__dict__.pop('m', None)
        self.__dict__.pop('n', None)
        self.__dict__.pop('y_true_dict_list', None)
        self.__dict__.pop('y_true_mat_list', None)
        if hasattr(self, 'link_pred'):
            if type(self.link_pred) == torch.Tensor:
                self.link_pred = self.link_pred.item()
            self.link_true = self._GraphPair__ds_true
        self.__dict__.pop('_GraphPair__ds_true', None)


def _convert_to(x, format):
    import torch
    if type(x) != sparse.csr.csr_matrix and format == 'sparse':
        if type(x) == torch.Tensor:
            return sparse.csr_matrix(x.detach().cpu().numpy())
        elif type(x) == np.ndarray:
            return sparse.csr_matrix(x)
        else:
            assert False
    if type(x) != np.ndarray and format == 'numpy':
        if type(x) == torch.Tensor:
            return x.detach().cpu().numpy()
        elif type(x) == sparse.csr.csr_matrix:
            return x.toarray()
        else:
            assert False
    elif type(x) != torch.Tensor and 'torch' in format:
        device = format.split('_')[1]
        if type(x) == np.ndarray:
            return torch.tensor(x, device=device)
        elif type(x) == sparse.csr.csr_matrix:
            return torch.tensor(x.toarray(), device=device)
        else:
            assert False
    else:
        return x
