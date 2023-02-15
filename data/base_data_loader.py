from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from base_dataset import BaseDataset

class BaseDataLoader(ABC):
    def __init__(self, opt, dataset: BaseDataset):
        super(BaseDataLoader, self).__init__()
        self.opt = opt
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=opt.batchSize, num_workers=int(opt.nThreads))
        
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __iter__(self):
        pass
    

