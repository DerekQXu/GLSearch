from __future__ import annotations

import os
import re
from torch.utils.data import Dataset as TorchDataset
import random
import pickle
from .curriculum_dataset import CurriculumDataset
from options import opt


def load_dataset_list(dataset_list: List[Tuple[List[Tuple[str, int]]]] or str) -> List[CurriculumDataset]:
    """
    :param dataset_list: List of curriculums. Each curriculum is a list of (dataset_name, num_pairs) tuples. Each tuple represents a dataset of graph, and the number of graphs to use from that dataset
    """
    data = []
    # For each curriculum
    for curriculum in dataset_list:
        dataset_name_list = curriculum[0]
        assert type(curriculum[1]) is int
        cur_datasets = []
        num_pairs_list = []
        # Load and merge all the datasets within the curriculum
        for single_dataset_tuple in dataset_name_list:
            dataset_name, num_pairs = single_dataset_tuple
            dataset = load_single_dataset(dataset_name)
            cur_datasets.append(dataset)
            num_pairs_list.append(num_pairs)

        data.append(CurriculumDataset.merge(cur_datasets, num_pairs_list))
    return data


def load_single_dataset(dataset_name: str) -> CurriculumDataset:
    """
    :param dataset_name: Base name of the dataset file
    """
    # append phase to the name
    dataset_name = dataset_name + '_' + opt.phase

    # search file in folder
    full_name = None
    for k in os.listdir(opt.data_folder):
        if re.match(dataset_name, k):
            full_name = k
            break
    if full_name is None:
        raise ValueError('No file found for dataset {}'.format(dataset_name))

    path = os.path.join(opt.data_folder, full_name)
    print('Loading data from {}'.format(path))

    with open(path, 'rb') as f:
        json_data = pickle.load(f)
        data = CurriculumDataset.from_legacy_dataset(json_data)

    if data is None:
        raise ValueError('Could not open dataset {}'.format(dataset_name))
    return data
