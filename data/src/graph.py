import networkx as nx

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

    def gid(self):
        return self.graph['gid']

