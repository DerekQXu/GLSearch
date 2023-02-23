from __future__ import annotations

import os
import re
from torch.utils.data import Dataset as TorchDataset
import random
import pickle
from curriculum_dataset import CurriculumDataset

def load_train_test_data(dataset_list: List[Tuple[List[Tuple[str, int]]]] or str) -> Tuple[List[CurriculumDataset], List[CurriculumDataset]]:
    """
    :param dataset_list: List of curriculums. Each curriculum is a list of (dataset_name, num_pairs) tuples. Each tuple represents a dataset of graph, and the number of graphs to use from that dataset
    """
    train_data = []
    test_data = []
    # For each curriculum
    for curriculum in dataset_list:
        dataset_name_list = curriculum[0]
        assert type(curriculum[1]) is int
        train_data_cur = None
        test_data_cur = None
        num_pairs_list = []
        # Load and merge all the datasets within the curriculum
        for single_dataset_tuple in dataset_name_list:
            dataset_name, num_pairs = single_dataset_tuple
            train_data_elt, test_data_elt = load_single_train_test_data(dataset_name)
            # Merge the datasets with the ones already loaded in previous iterations (if any)
            train_data_cur = train_data_elt if train_data_cur is None else train_data_cur.merge(train_data_elt)
            test_data_cur = test_data_elt if test_data_cur is None else test_data_cur.merge(test_data_elt)
            num_pairs_list.append(num_pairs)
        # TODO find how to use the num_pairs_list
        train_data.append(train_data_cur)
        test_data.append(test_data_cur)

    return train_data, test_data


def load_single_train_test_data(dataset_name: str) -> Tuple[CurriculumDataset, CurriculumDataset]:

    # search file in folder
    full_name = None
    for k in os.listdir(data_folder):
        if re.match(dataset_name, k):
            full_name = k
            break
    if full_name is None:
        raise ValueError('No file found for dataset {}'.format(dataset_name))

    path = os.path.join(data_folder, full_name)
    print('Loading data from {}'.format(path))

    with open(path, 'rb') as f:
        data = pickle.load(f)
        train_data = CurriculumDataset.from_legacy_dataset(data['train_data'])
        test_data = CurriculumDataset.from_legacy_dataset(data['test_data'])

    return train_data, test_data


if __name__ == '__main__':
    data_folder = "dataset_files"
    _dataset_list = [
        ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
    ]
    load_train_test_data(_dataset_list)