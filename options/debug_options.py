import os

"""Debug options for the project."""

class DebugOptions:
    def __init__(self):
        self.phase = 'train'
        self.device = 'cuda'
        self.data_folder = os.path.join("data", "dataset_files")
        self.dataset_list = [
            ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
        ]
        self.shuffle_input = False
        self.batch_size = 1
        # feature encoders
        self.node_fe_1 = 'one_hot'
        self.node_fe_2 = 'local_degree_profile'
