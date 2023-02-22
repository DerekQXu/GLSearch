import os
import pickle
import networkx as nx

def load_single_dataset(path):
    with open(path, "rb") as f:
        graph = pickle.load(f)
        print("Number of nodes: ", graph.number_of_nodes())
        print(graph)




if __name__ == '__main__':
    folder = os.path.join("data","dataset_files")
    pickles = os.listdir(folder)
    print(pickles)
    import re
    for p in pickles:
        if re.match(".*roadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2.*", p):
            load_single_dataset(os.path.join(folder, p))
