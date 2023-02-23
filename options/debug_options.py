import os

"""Debug options for the project."""

class DebugOptions:
    phase = 'train'
    device = 'cpu'
    data_folder = os.path.join("data","dataset_files")
    dataset_list = [
            ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
        ]
