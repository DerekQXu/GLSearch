from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
    def __init__(self, opt):
        super(BaseDataset, self).__init__()
        self.opt = opt
    
    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass
