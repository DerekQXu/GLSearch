from typing import Dict, List, Tuple
from .graph import Graph

class GraphPair:
    true_dict_list: List[Dict[int, int]]
    g1:Graph
    g2:Graph
    duration:float

    # FIXME if we need attributes m and n, they are just the number_of_nodes() of g1 and g2, respectively.

    def __init__(self, true_dict_list: List[Dict[int, int]] = None):
        self.true_dict_list = true_dict_list

    def __str__(self):
        return f"GraphPair({self.g1.gid()},{self.g2.gid()})"

    @staticmethod
    def from_legacy_pair(legacy_pair: Dict[str,any]):
        """
        Create a new GraphPair from a legacy GraphPair.
        """
        new_pair = GraphPair(legacy_pair['y_true_dict_list'])
        return new_pair

    def assign_g1_g2(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        # TODO is this ever true?
        if hasattr(self, 'y_pred_mat_list'):
            print("GraphPair.assign_g1_g2(): y_pred_mat_list exists, code portion is useful!")
            self._check_shape(self.y_pred_mat_list)

    # TODO is this used?
    def _check_shape(self, y_pred_mat_list):
        for y_pred_mat in y_pred_mat_list:
            if y_pred_mat.shape != (self.m, self.n):
                raise ValueError('Shape mismatch! y_pred_mat shape {}; '
                                 'm, n: {} by {}'.
                                 format(y_pred_mat.shape, self.m, self.n))

    def assign_pred_time(self, duration):
        self.duration = duration

    def get_pred_time(self):
        return self.duration