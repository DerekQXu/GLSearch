from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()


    def parse(self):
        super().parse()
        
        self.opt.phase = 'train'
        
        return self.opt
