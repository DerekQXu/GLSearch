from __future__ import annotations

import torch
from base_dataset import BaseDataset

from src.dataset import Dataset


class CurriculumDataset(BaseDataset):
    dataset: Dataset = None
    gid1gid2_list: torch.Tensor = None
    num_node_features: int = None

    def __init__(self, dataset: Dataset, num_node_features: int):
        super(CurriculumDataset, self).__init__(opt=None)
        self.dataset = dataset
        # TODO retrieve this from dataset
        # self.gid1gid2_list = gid1gid2_list
        self.num_node_features = num_node_features
        pass

    def __len__(self):
        # TODO implement this
        pass

    def __getitem__(self, index):
        # TODO implement this
        pass

    @staticmethod
    def from_legacy_dataset(legacy_dataset: Dict[str, any]) -> CurriculumDataset:
        """
        Create a new CurriculumDataset from a legacy OurCocktailDataset.
        """
        new_dataset = CurriculumDataset(
            Dataset.from_legacy_dataset(legacy_dataset['dataset']),
            legacy_dataset['num_node_feat'])
        return new_dataset

    def merge(self, other: CurriculumDataset):
        """
        Merge this dataset with another dataset.
        """
        # TODO implement this
        pass
