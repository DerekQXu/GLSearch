from base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize(self)

        self.parser.add_argument('--test', required=True, type=str, help='test parameter')
        
    def parse(self):
        super().parse()
        
        self.opt.isTrain = False
        
        return self.opt

