from base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()

    def parse(self):
        super().parse()
        
        self.opt.isTrain = 'test'
        
        return self.opt

