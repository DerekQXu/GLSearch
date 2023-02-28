from .graph import Graph
from .graph_pair import GraphPair
from typing import Dict, List, Tuple
from collections import OrderedDict
from utils.stats import generate_stat_line

PairDict = Dict[Tuple[int, int], GraphPair]


class Dataset:
    name: str
    graphs: List[Graph]
    pairs: Dict[Tuple[int, int], GraphPair]
    gs_map: Dict[int, int]
    id_map: Dict[int, int]
    type_dataset: str

    def __init__(self, name: str = None, graphs: List[Graph] = None, pairs: PairDict = None,
                 gs_map: Dict[int, int] = None, id_map: Dict[int, int] = None):
        self.name = name
        self.graphs = graphs
        # TODO gs_map and id_map are generated automatically, we can probably avoid to bring them from the legacy dataset
        self.gs_map = self._gen_gs_map()  # a dict that maps gid to id, id used for enumerating the dataset
        # TODO check if id_map is really needed
        self.id_map = self._gen_id_map()  # a dict that maps id to gid
        self.pairs = pairs  # a dict that maps (gid1, gid2) to GraphPair
        # Perform some validity checks, to make sure the dataset is valid
        self._check_invariants()

    @staticmethod
    def from_legacy_dataset(legacy_dataset: Dict[str, any]):
        """
        Create a new Dataset from a legacy OurDataset (in exported JSON format).
        """
        legacy_pairs = legacy_dataset['pairs']
        pairs = {key: GraphPair.from_legacy_pair(legacy_pairs[key]) for key in legacy_pairs}
        graph_list = [Graph(legacy_dataset['gs'][i]) for i in range(len(legacy_dataset['gs']))]
        #TODO possibly import glabel and see if it is used anywhere
        new_dataset = Dataset(
            legacy_dataset['name'],
            graph_list,
            pairs,
            legacy_dataset['gs_map'],
            legacy_dataset['id_map'])
        return new_dataset

    def __str__(self):
        return generate_stat_line('Dataset', self.name) + \
            generate_stat_line('num_graphs', len(self.graphs)) + \
            generate_stat_line('num_pairs', len(self.pairs))

    def _gen_gs_map(self):
        rtn = {}
        for i, g in enumerate(self.graphs):
            rtn[g.gid()] = i
        return rtn

    def _gen_id_map(self):
        assert (hasattr(self, "gs_map"))
        return {id: gid for gid, id in self.gs_map.items()}

    def print_stats(self):
        print(self)
        print('WARNING: deprecated: the easiest way to inspect this class is to directly call print(dataset)')
        print('To show more info, edit the function Dataset.__str__')

    def _check_invariants(self):
        _assert_nonempty_str(self.name)
        assert self.graphs and type(self.graphs) is list, type(self.graphs)
        assert self.gs_map and type(self.gs_map) is dict
        assert len(self.graphs) == len(self.gs_map)
        self._check_pairs()
        self._check_types()

    def _check_pairs(self):
        assert type(self.pairs) is dict  # may have zero pairs
        for (gid1, gid2), pair in self.pairs.items():
            assert gid1 in self.gs_map and gid2 in self.gs_map, \
                '{} {}'.format(gid1, gid2)
            assert isinstance(pair, GraphPair)

    def _check_types(self):
        for g in self.graphs:
            assert isinstance(g, Graph)

    def look_up_graph_by_gid(self, gid:int) -> Graph:
        _id = self.gs_map.get(gid)
        if _id is None:
            raise ValueError('Cannot find graph w/ gid {} out of {} graphs'.format(
                gid, len(self.gs_map)))
        assert 0 <= _id < len(self.graphs)
        return self.graphs[_id]

    def look_up_pair_by_gids(self, gid1:int, gid2:int) -> GraphPair:
        pair = self.pairs.get((gid1, gid2))
        if pair is None:
            pair = self.pairs.get((gid2, gid1))
            if not pair:
                raise ValueError('Cannot find ({},{}) out of {} pairs'.format(
                    gid1, gid2, len(self.pairs)))
        return pair

def _assert_nonempty_str(s):
    assert s is None or (s and type(s) is str)



