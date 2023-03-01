from options import opt
from torch.utils.data import DataLoader
from data import BatchData, load_dataset_list

def train():
    datasets = load_dataset_list(opt.dataset_list)
    for cur_dataset in datasets:
        print(str(cur_dataset))
        """data_loader = DataLoader(cur_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle_input)
        for i, data in enumerate(data_loader):
            batch_data = BatchData(data, cur_dataset.dataset)
            print(str(batch_data))
"""

if __name__ == '__main__':

    train()