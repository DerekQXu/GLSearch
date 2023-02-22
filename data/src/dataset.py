from graph_pair import GraphPair
from networkx.classes.graph import Graph
from typing import Dict, List, Tuple
from legacy_glsearch_code.dataset import OurDataset

class Dataset:
    name:string
    graphs: List[Graph]
    pairs: Dict[Tuple[int, int], GraphPair]
    gs_map: Dict[int, int]
    id_map: Dict[int, int]

    def __init__(self, name: string = None, graphs: List[Graph] = None, pairs: Dict[Tuple[int, int], GraphPair] = None, gs_map: Dict[int, int] = None, id_map: Dict[int, int] = None):
        self.name = name
        self.graphs = graphs
        self.pairs = pairs
        self.gs_map = gs_map
        self.id_map = id_map

    @staticmethod
    def from_legacy_dataset(legacy_dataset: OurDataset) -> Dataset:
        """
        Create a new Dataset from a legacy OurDataset.
        """
        pairs = {key: GraphPair.from_legacy_pair(legacy_dataset.pairs[key]) for key in legacy_dataset.pairs}

        new_dataset = Dataset(
            legacy_dataset.name,
            legacy_dataset.graphs,
            pairs,
            legacy_dataset.gs_map,
            legacy_dataset.id_map)
        return new_dataset
