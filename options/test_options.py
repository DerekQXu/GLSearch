from base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Specify the device to use. Can be "cpu" or "cuda".')
        self.parser.add_argument('--data_folder', type=str, default=os.path.join("data", "dataset_files"),
                                 help='path to the folder containing the dataset files (in pickle json format)')
        dataset_list = [
            ([('aids700nef', 30),
              ('linux', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=16,ed=5,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=14,ed=0.14,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=18,ed=0.2|2,gen_type=WS', -1),
              ], 2500),
            ([('ptc', 30),
              ('imdbmulti', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=32,ed=4,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=30,ed=0.12,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=34,ed=0.2|2,gen_type=WS', -1),
              ], 2500),
            ([('mutag', 30),
              ('redditmulti10k', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=48,ed=4,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=46,ed=0.1,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=50,ed=0.2|4,gen_type=WS', -1),
              ], 2500),
            ([('webeasy', 30),
              ('mcsplain-connected', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=64,ed=3,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=62,ed=0.08,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=66,ed=0.2|4,gen_type=WS', -1),
              ], 2500)
        ]
        self.parser.add_argument('--dataset_list', type=list, default=dataset_list, help='list of datasets to train on')
        self.parser.add_argument('--shuffle_input', type=bool, default=False, help='')
        self.parser.add_argument('--batch_size', type=int, default=1, help='')
        # feature encoders
        self.parser.add_argument('--node_fe_1', type=str, default='one_hot', help='first node feature encoder')
        self.parser.add_argument('--node_fe_2', type=str, default='local_degree_profile',
                                 help='second node feature encoder')

    def parse(self):
        super().parse()
        
        self.opt.isTrain = 'test'
        
        return self.opt

