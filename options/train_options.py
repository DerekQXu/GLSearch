from options.base_options import BaseOptions
import os

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        # TODO add help hints
        self.parser.add_argument('--device', type=str, default='cpu', help='')
        self.parser.add_argument('--data_folder', type=str, default=os.path.join("data", "dataset_files"), help='')
        dataset_list = [
            ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
        ]
        self.parser.add_argument('--dataset_list', type=list, default=dataset_list, help='')
        self.parser.add_argument('--shuffle_input', type=bool, default=False, help='')
        self.parser.add_argument('--batch_size', type=int, default=1, help='')
        # feature encoders
        self.parser.add_argument('--node_fe_1', type=str, default='one_hot', help='')
        self.parser.add_argument('--node_fe_2', type=str, default='local_degree_profile', help='')

    def parse(self):
        super().parse()
        
        self.opt.phase = 'train'
        
        return self.opt
