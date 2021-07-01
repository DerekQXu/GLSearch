import sys
import pandas as pd
import os
import copy
from ast import literal_eval
dir_path = os.path.dirname(os.path.realpath(__file__))

def preprocess_node_mapping(dataset_csv_path):
    if not os.path.exists(dataset_csv_path):
        print("{} ....is not a valid path".format(dataset_csv_path))
        return
    dir_path = os.path.dirname(dataset_csv_path)
    res_folder_path = os.path.join(dir_path, "preprocessed")
    res_file_name = os.path.basename(os.path.splitext(dataset_csv_path)[0])
    res_path = os.path.join(res_folder_path, res_file_name + "_preprocessed_node_mapping")

    if not os.path.isdir(res_folder_path):
        os.mkdir(res_folder_path)

    for index, chunk in enumerate(pd.read_csv('{}'.format(dataset_csv_path), sep=',', chunksize=1)):
        if index % 500 == 0:
            print(index)
        edge_mappings = chunk['node_mapping'][index]
        node_mappings = []
        try:
            edge_mappings = literal_eval(edge_mappings)
        except ValueError as e:
            chunk.to_csv('{}.csv'.format(os.path.join(res_folder_path, res_file_name + "_bad_mappings")), header=False, mode='a', chunksize=1)
            continue
        chunk['edge_mapping'] = [copy.deepcopy(edge_mappings)]

        for edge_mapping in edge_mappings:
            node_mapping = {}
            # one node with one node mapping
            if edge_mapping == {}:
                pass
            # two nodes with two nodes mapping
            elif len(edge_mapping) == 1:
                nodes_i, nodes_j = edge_mapping.popitem()
                node_mapping[nodes_i[0]] = nodes_j[0]
                node_mapping[nodes_i[1]] = nodes_j[1]

            # O.W.
            else:
                for nodes_i, nodes_j in edge_mapping.items():  # key, value
                    for node_i in nodes_i: # nodes in first edge
                        if node_mapping.get(node_i) is None:   #no node_mapping yet
                            node_mapping[node_i] = list(nodes_j)
                        else:
                            # if node_i == '6':
                            #     kk = node_mapping[node_i]
                            #     a = set(node_mapping[node_i])
                            #     b = set(nodes_j)
                            #     c = b.intersection(a)
                            #     d = list(c)
                            node_mapping[node_i] = list(set(nodes_j).intersection(set(node_mapping[node_i])))[0] if type(
                                node_mapping[node_i]) == list else list(set(nodes_j).intersection(
                                set([node_mapping[node_i]])))[0]

                # <class 'dict'>: {'3': '6', '5': ['6', '8'], '0': ['7', '6'], '4': '2', '7': ['0', '2']}
                for n_1, n_2 in node_mapping.items():
                    if type(n_2) == list:
                        n_1_p, n_2_p = None, None
                        for nodes_i, nodes_j in edge_mapping.items():
                            if n_1 in nodes_i:
                                l = list(nodes_i)
                                l.remove(n_1)
                                n_1_p = l[0]
                                assert (type(node_mapping[n_1_p]) != list)
                                n_2_p = node_mapping[n_1_p]
                        assert (n_1_p is not None and n_2_p is not None)
                        n_2 = list(n_2)
                        n_2.remove(n_2_p)
                        assert (len(n_2) == 1)
                        node_mapping[n_1] = n_2[0]
                        assert (type(node_mapping[n_1]) != list)

            node_mappings.append(node_mapping)
        chunk['node_mapping'] = str(node_mappings)

        # chunk.to_csv('{}.csv'.format(dataset_name[:-5]), header=False, mode='a', chunksize=1)
        if index == 0:
            chunk.to_csv('{}.csv'.format(res_path), header=chunk.columns, mode='w', chunksize=1)
        else:
            chunk.to_csv('{}.csv'.format(res_path), header=False, mode='a', chunksize=1)

if __name__ == '__main__':
    if 'python' in sys.argv[0]:
        dataset_csv_path = sys.argv[2]
    else:
        dataset_csv_path = sys.argv[1]
    preprocess_node_mapping(dataset_csv_path)
