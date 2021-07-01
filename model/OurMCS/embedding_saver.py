class EmbeddingSaver(object):
    def __init__(self):
        self.d = {}

    def add(self, gid1, gid2, exhausted_v, exhausted_w, x_embs):
        if len(self.d) > 2:
            self.clear()
        self.d[self._get_key(gid1, gid2, exhausted_v, exhausted_w)] = x_embs

    def get(self, gid1, gid2, exhausted_v, exhausted_w):
        key = self._get_key(gid1, gid2, exhausted_v, exhausted_w)
        if key in self.d:
            return self.d[key]
        else:
            return None

    def clear(self):
        self.d = {}

    def _get_key(self, gid1, gid2, exhausted_v, exhausted_w):
        return gid1, gid2, frozenset(exhausted_v), frozenset(exhausted_w)

EMBEDDING_SAVER = EmbeddingSaver()  # can be used by `from embedding_saver import EMBEDDING_SAVER`
