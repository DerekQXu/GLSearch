from options import opt
from data.load_datasets import load_dataset_list

def train():
    datasets = load_dataset_list(opt.dataset_list)
    print(datasets)

if __name__ == '__main__':

    train()