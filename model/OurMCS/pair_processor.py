def preproc_graph_pair(g1, g2, pair):
    return [g1,
            g2]  # TODO: supersource, corasening, ... (store sp in pair for split)


def _supersource(g):
    raise NotImplementedError()


def _form_pair(g1, g2):
    raise NotImplementedError()


def postproc_graph_pairs_assign_node_embeds(node_embed_list, node_embed_name,
                                            pair_list):
    assert len(node_embed_list) == 2 * len(
        pair_list)  # TODO: pairwise supersource...
    for i, pair in enumerate(pair_list):
        # Update node embeddings.
        pair.g1.__setattr__(node_embed_name, node_embed_list[2 * i])
        pair.g2.__setattr__(node_embed_name, node_embed_list[2 * i + 1])
