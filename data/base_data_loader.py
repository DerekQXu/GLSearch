from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    def __init__(self, opt, dataset: Dataset):
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
    

