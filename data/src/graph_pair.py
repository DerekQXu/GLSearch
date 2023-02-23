from typing import Dict, List, Tuple

class GraphPair:
    true_dict_list: List[Dict[int, int]]

    def __init__(self, true_dict_list: List[Dict[int, int]] = None):
        self.true_dict_list = true_dict_list

    @staticmethod
    def from_legacy_pair(legacy_pair: Dict[str,any]):
        """
        Create a new GraphPair from a legacy GraphPair.
        """
        new_pair = GraphPair(legacy_pair['y_true_dict_list'])
        return new_pair