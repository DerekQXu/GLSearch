from base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize(self)
        
        self.parser.add_argument('--train', required=True, type=str, help='train parameter')
        
    def parse(self):
        super().parse()
        
        self.opt.isTrain = True
        
        return self.opt
