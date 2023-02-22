from utils import append_ext_to_filepath
from collections import OrderedDict
from random import randint
from os.path import join

""" Graph labels. """


def add_glabel_to_each_graph(graphs, dirname, glabel_type):
    glabels_dict = None
    if glabel_type in ['discrete', 'continuous']:
        glabels_dict = _load_glabels_from_txt(join(dirname, 'glabels.txt'),
                                              glabel_type)
    seen = set()  # check every graph id is seen only once
    for g in graphs:
        gid = g.get_nxgraph().graph['gid']
        assert gid not in seen, '{} seen but {}'.format(seen, gid)
        seen.add(gid)
        if glabel_type is None:
            continue
        if glabel_type in ['discrete', 'continuous']:
            glabel = glabels_dict[gid]
        elif glabel_type == 'random':
            glabel = randint(0, 9)  # randomly assign a graph label from {0, .., 9}
        else:
            assert False
        g.get_nxgraph().graph['glabel'] = glabel  # important; this line actually stores the label
    return graphs


def save_glabels_as_txt(filepath, glabels):
    filepath = append_ext_to_filepath('.txt', filepath)
    with open(filepath, 'w') as f:
        for id, glabel in OrderedDict(glabels).items():
            f.write('{}\t{}\n'.format(id, glabel))


def _load_glabels_from_txt(filepath, glabel_type):
    filepath = append_ext_to_filepath('.txt', filepath)
    rtn = {}
    int_map = {}
    seen_glabels = set()
    with open(filepath) as f:
        for line in f:
            ls = line.rstrip().split()
            assert (len(ls) == 2)
            gid = int(ls[0])
            if glabel_type == 'discrete':
                glabel, int_map = _parse_as_int_glabel(ls[1], int_map)
            elif glabel_type == 'continuous':
                glabel = float(ls[1])
            else:
                assert False, 'Should not call this function if no glabel ' \
                              'or random glabels'
            rtn[gid] = glabel
            seen_glabels.add(glabel)
    if 0 not in seen_glabels:  # check 0-based graph labels
        raise RuntimeError('{} has no glabel 0; {}'.format(filepath, seen_glabels))
    return rtn


def _parse_as_int_glabel(glabel_str, int_map):
    try:
        glabel = int(glabel_str)
    except ValueError:
        label_string = glabel_str
        glabel = int_map.get(label_string)
        if glabel is None:
            glabel = len(int_map)  # guarantee 0-based
            int_map[label_string] = glabel  # increase the size of int_map by 1
    return glabel, int_map
