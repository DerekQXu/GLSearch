import networkx as nx
import warnings

class Graph(nx.Graph):
    """
    A networkx graph with some additional attributes.
    """
    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)
        if 'gid' not in self.graph or type(self.graph['gid']) is not int \
                or self.graph['gid'] < 0:
            raise ValueError('Graph ID must be non-negative integers {}'.
                             format(self.graph.get('gid')))
        if not nx.is_connected(self):
            raise ValueError('Graph {} must be connected'.
                             format(self.graph['gid']))
        origin_graph = args[0]
        self.init_x = origin_graph.init_x

    def gid(self):
        return self.graph['gid']

    def get_nxgraph(self):
        # FIXME deprecated
        warnings.warn("Graph.get_nxgraph() is deprecated, use Graph directly instead", DeprecationWarning)
        return self

class HierarchicalGraph(Graph):
    def __init__(self):
        super(HierarchicalGraph, self).__init__()
        # FIXME if we need to implement this, we need to integrate the implementation of the following methods:
        #   GraphPair.assign_g1_g2()
        raise NotImplementedError()

