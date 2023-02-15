from torch.nn import Module
from abc import ABC, abstractmethod

class BaseModel(Module, ABC):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt

    @abstractmethod
    def forward(self):
        pass
