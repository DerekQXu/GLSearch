from load_classic_data import load_debug_data, load_classic_data
from load_syn_data import load_syn_data
from load_mccreesh_data import load_mccreesh_data
from load_cv_data import load_cv_dataset
from load_old_data import load_old_data
from load_kiwi_data import load_kiwi_data
from load_interaction_data import load_interaction_data
from load_pdb_data_mcs import load_pdb_data_mcs
from load_gexf_data import load_duogexf_data, load_isogexf_data, load_ccgexf_data, load_ssgexf_data
try:
    from load_pdb_data import load_pdb_data
except:
    print("did not load pdb data")
from load_circuit_graph import load_circuit_graph


def get_dataset_conf(name):
    name = name if '[' != name[0] else eval(name)[0]

    if 'debug' in name:
        eatts = []
        if 'BA' in name:
            natts = []
            tvt_options = ['train', 'test']
        else:
            natts = ['type']
            tvt_options = ['all']
        align_metric_options = ['random']
        loader = load_debug_data
        dataset_type = 'OurDataset'
        glabel = 'random'
    elif 'syn' in name:
        eatts = []
        natts = ['type']
        tvt_options = ['train', 'test']
        align_metric_options = ['random', 'mcs']
        loader = load_syn_data
        dataset_type = 'OurDataset'
        glabel = 'random'
    elif 'isogexf' in name:
        natts = []  # 'label'
        eatts = []
        tvt_options = ['test']
        align_metric_options = ['mcs']
        loader = load_isogexf_data
        glabel = None
        dataset_type = 'OurDataset'
    elif 'duogexf' in name:
        eatts = []
        natts = ['type']  # 'label'
        tvt_options = ['test']
        align_metric_options = ['mcs']
        loader = load_duogexf_data
        glabel = None
        dataset_type = 'OurDataset'
    elif 'ssgexf' in name:
        eatts = []
        natts = ['type']  # 'label'
        tvt_options = ['test']
        align_metric_options = ['mcs']
        loader = load_ssgexf_data
        glabel = None
        dataset_type = 'OurDataset'
    elif 'ccgexf' in name:
        eatts = []
        natts = ['type']  # 'label'
        tvt_options = ['test']
        align_metric_options = ['mcs']
        loader = load_ccgexf_data
        glabel = None
        dataset_type = 'OurDataset'
    elif name in ['aids700nef', 'linux', 'imdbmulti', 'mutag', 'alchemy',
                  'ptc', 'nci1', 'nci109', 'webeasy', 'redditmulti10k', 'DD']:
        if name in ['aids700nef', 'ptc', 'nci1', 'nci109', 'webeasy', 'mutag', 'DD']:
            natts = ['type']
        elif name in ['alchemy']:
            natts = ['a_type']
        else:
            natts = []
        eatts = []
        tvt_options = ['all']
        align_metric_options = ['mcs']
        loader = load_classic_data
        dataset_type = 'OurDataset'
        if name in ['imdbmulti', 'DD']:
            glabel = 'discrete'
        elif name in ['alchemy']:
            glabel = 'continuous'
        else:
            glabel = None
    elif name in ['mcs33ve', 'mcs33ve-connected', 'mcsplain', 'mcsplain-connected', 'sip']:
        if name in ['mcsplain', 'mcsplain-connected', 'sip']:
            natts = []
            eatts = []
        else:
            natts = ['feature']
            eatts = ['feature']
        tvt_options = ['all']
        align_metric_options = ['mcs']
        loader = load_mccreesh_data
        dataset_type = 'OurDataset'
        glabel = None
    # elif name in ['icpr2016_x', 'icpr2016_y]
    elif name in ['CUB']:
        natts = []
        eatts = []
        tvt_options = ['train', 'test']
        align_metric_options = ['vsm']
        loader = load_cv_dataset
        dataset_type = 'OurDataset'
        glabel = None
    elif name in ['aids700nef_old_small', 'aids700nef_old', 'linux_old',
                  'imdbmulti_old', 'ptc_old']:
        if name in ['debug_old', 'aids700nef_old', 'ptc_old']:
            natts = ['type']
            eatts = []
        else:
            natts = []
            eatts = []
        tvt_options = ['train', 'test']
        align_metric_options = ['ged', 'mcs']
        loader = load_old_data
        dataset_type = 'OurOldDataset'
        glabel = None
    elif 'kiwi_loop' in name:  # kiwi_loop:model=BA,ng=100,nn_mean=30,nn_std=5,ed_mean=0.5,ed_std=0.2
        natts = []
        eatts = []
        tvt_options = ['all']
        align_metric_options = [None]
        loader = load_kiwi_data
        dataset_type = 'OurDataset'
        glabel = 'discrete'
    elif name in ['ddi_snap_drugbank', 'ddi_small_drugbank', 'ddi_decagon', 'ddi_decagon_combination', 'ddi_drugcombo']:
        natts = ['atom_type', 'aromatic', 'acceptor', 'donor', 'hybridization', 'num_h']
        eatts = []
        tvt_options = ['all']
        align_metric_options = ['interaction']
        loader = load_interaction_data
        glabel = None
        dataset_type = 'BiLevelDataset'
    elif 'ppi_snap_pdb' in name:
        natts = ['type', 'x', 'y', 'z', 'ss_type']  # 'label'
        eatts = ['dist']
        tvt_options = ['all']
        align_metric_options = ['interaction']
        loader = load_pdb_data
        glabel = None
        dataset_type = 'OurStringPdbDataset'
    elif 'pdb_mcs' in name:
        natts = ['type', 'x', 'y', 'z', 'ss_type']
        eatts = ['dist', 'id']
        tvt_options = ['train', 'test']
        align_metric_options = ['random']
        glabel = 'random'
        loader = load_pdb_data_mcs
        dataset_type = 'OurMCSPdbDataset'
    elif 'circuit' in name:
        natts = ['is_device', 'name', 'type', 'port']  # 'label'
        eatts = []
        tvt_options = ['test']
        align_metric_options = ['mcs']
        loader = load_circuit_graph
        glabel = None
        dataset_type = 'OurDataset'
    elif name == 'cocktail':
        eatts = []
        natts = ['type']
        tvt_options = ['train', 'test']
        align_metric_options = ['mcs']
        loader = load_syn_data
        dataset_type = 'OurDataset'
        glabel = 'random'
    else:
        raise ValueError('Unknown dataset {}'.format(name))
    check_tvt_align_lists(tvt_options, align_metric_options)
    return natts, eatts, tvt_options, align_metric_options, loader, \
           dataset_type, glabel


def check_tvt_align_lists(tvt_options, align_metric_options):
    for tvt in tvt_options:
        check_tvt(tvt)
    for align_metric in align_metric_options:
        check_align(align_metric)


def check_tvt(tvt):
    if tvt not in ['train', 'val', 'test', 'all']:
        raise ValueError('Unknown tvt specifier {}'.format(tvt))


def check_align(align_metric):
    if align_metric not in ['ged', 'mcs', 'vsm', 'random', 'interaction', None]:
        raise ValueError('Unknown graph alignment metric {}'.
                         format(align_metric))
    if align_metric is None:
        return 'noalign'
    else:
        return align_metric


def check_node_ordering(node_ordering):
    if node_ordering is None:
        return 'noordering'
    elif node_ordering == 'bfs':
        return node_ordering
    else:
        raise ValueError('Unknown node ordering {}'.format(node_ordering))
