from .utils import get_flags_with_prefix_as_list
from torch_geometric.transforms import LocalDegreeProfile



# FIXME (maybe): removed dataset, check if it's actually needed
def encode_node_features(pyg_single_g):
    assert pyg_single_g is not None
    input_dim = pyg_single_g.x.shape[1]
    node_feat_encoders = get_flags_with_prefix_as_list('node_fe')
    # FIXME (maybe): check if this is actually needed (not needed if "dataset" is not used, right?)
    if 'one_hot' not in node_feat_encoders:
        raise ValueError('Must have one hot node feature encoder!')
    for nfe in node_feat_encoders:
        if nfe == 'one_hot':
            NotImplementedError('One hot node feature encoder not implemented! implement from GLSearch')
        elif nfe == 'local_degree_profile':
            input_dim += 5
            if pyg_single_g:
                pyg_single_g = LocalDegreeProfile()(pyg_single_g)
        else:
            raise ValueError('Unknown node feature encoder {}'.format(nfe))
    if input_dim <= 0:
        raise ValueError('Must have at least one node feature encoder '
                         'so that input_dim > 0')

    return pyg_single_g, input_dim





