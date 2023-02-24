from data.src.graph_pair import GraphPair
from data.src.graph import Graph
from typing import Dict, List, Tuple

PairDict = Dict[Tuple[int, int], GraphPair]
class Dataset:
    name:str
    graphs: List[Graph]
    pairs: Dict[Tuple[int, int], GraphPair]
    gs_map: Dict[int, int]
    id_map: Dict[int, int]
    type_dataset:str

    def __init__(self, name: str = None, graphs: List[Graph] = None, pairs: PairDict = None, gs_map: Dict[int, int] = None, id_map: Dict[int, int] = None):
        self.name = name
        self.graphs = graphs
        self.pairs = pairs
        self.gs_map = gs_map
        self.id_map = id_map

    @staticmethod
    def from_legacy_dataset(legacy_dataset: Dict[str, any]):
        """
        Create a new Dataset from a legacy OurDataset (in exported JSON format).
        """
        legacy_pairs = legacy_dataset['pairs']
        pairs = {key: GraphPair.from_legacy_pair(legacy_pairs[key]) for key in legacy_pairs}
        graph_list = [Graph(legacy_dataset['gs'][i]) for i in range(len(legacy_dataset['gs']))]

        new_dataset = Dataset(
            legacy_dataset['name'],
            graph_list,
            pairs,
            legacy_dataset['gs_map'],
            legacy_dataset['id_map'])
        return new_dataset
