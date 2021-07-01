import networkx as nx

# BFS ordering
def get_bfs(g):
    start_node = 0
    neighbors = dict(nx.bfs_successors(g, source=start_node, depth_limit=len(g)))
    queue = [start_node]
    non2bfs = {}
    bfs2non = {}
    old_idx = 0
    new_idx = 0

    while queue:
        old_idx = node_in = queue.pop(0)
        non2bfs[old_idx] = new_idx
        bfs2non[new_idx] = old_idx
        new_idx += 1
        if node_in in neighbors.keys():
            for node_out in neighbors[node_in]:
                queue.append(node_out)
    return non2bfs, bfs2non