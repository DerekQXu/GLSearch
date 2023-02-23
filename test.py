from options import opt
from data.load_datasets import load_dataset_list

def test():
    # Load datasets as list of CurriculumDataset objects.
    datasets = load_dataset_list(opt.dataset_list)
    print(datasets)

if __name__ == '__main__':
    test()