from collections import deque, Counter, defaultdict
import numpy as np
from config import FLAGS
from data_structures_search_tree_scalable import unroll_bidomains

#########################################################################
# Experience Buffer
#########################################################################
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def extend(self, experiences):
        # print('EXTEND SIZE: ', len(experiences))
        self.buffer.extend(experiences)

    def empty(self):
        self.buffer.clear()

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size),
                                   replace=False)
        return [self.buffer[idx] for idx in indices]


#########################################################################
# Bin Buffer
#########################################################################
# maybe clean this up a bit later
# Bin Buffer has SAME INTERFACE as experience buffer but samples evenly by
# hashing function i.e. nearest rounded down int of (q_max_UB + q_max_LB)/2
class BinBuffer:
    def __init__(self, capacity, sample_strat, biased=None):
        self.bins = defaultdict(list)
        self.history = []
        self.sample_strat = sample_strat
        self.biased = biased
        self.replace = self.biased != 'full'
        self.capacity = capacity

    def __len__(self):
        return len(self.history)

    def filter(self, experiences):
        good_experiences = [experience for experience in experiences
                            if not self._is_trivial(experience)]
        return good_experiences[-self.capacity:]

    def extend(self, experiences):
        # remove badly explored paths (regret); filter by capacity
        experiences = self.filter(experiences)  # TODO: check

        # make space in buffer for samples
        # get the bins we want to pop() and number of times to pop()
        n_rm = max(0,
                   len(self.history) + len(experiences) - self.capacity)  # number of elements to rm
        bin_dict = self.list2dict(self.history[:n_rm])  # bin_idx -> num_samples to remove
        for bin_key, n_rm_bin in bin_dict.items():
            # pop() from bin
            self.bins[bin_key] = self.bins[bin_key][n_rm_bin:]
            if len(self.bins[bin_key]) == 0:
                self.bins.pop(bin_key, None)
        # pop() from history
        self.history = self.history[n_rm:]

        # add samples to buffer
        for experience in experiences:
            bin_key = self.bin_hash(experience)
            if FLAGS.no_trivial_pairs:
                if self._is_trivial(experience):
                    continue  # TODO: make this code prettier in the future
            self.bins[bin_key].append(experience)
            self.history.append (bin_key)
        # print('len history:',len(self.history))

    def _is_trivial(self, experience):
        return len(unroll_bidomains(experience.edge.state_next.natts2bds)) == 0

    def sample(self, n_sample):
        experiences = []  # cumulative sample list
        # sample bins
        bin_keys = list(self.bins.keys())
        if self.biased == 'bias':
            p = np.array(bin_keys)  # can switch distribution later...
            p = p / np.sum(p)  # normalize p
            bin_keys_samples = np.random.choice(bin_keys, n_sample, p=p, replace=True)
        elif self.biased == 'full':
            bin_keys_samples = []
            while len(bin_keys_samples) < n_sample:
                bin_key = np.random.choice(bin_keys, 1, replace=True)[0]
                bin_keys_samples.extend(
                    [bin_key for _ in range(len(self.bins[bin_key]))]
                )
            bin_keys_samples = bin_keys_samples[:n_sample]
        elif self.biased is None:
            bin_keys_samples = np.random.choice(bin_keys, n_sample, replace=True)
        else:
            assert False

        # sample from each bin
        bin_dict = self.list2dict(bin_keys_samples)  # bin_idx -> num_samples to sample
        for bin_key, n_sample_bin in bin_dict.items():
            bin = self.bins[bin_key]
            if self.replace:
                experiences_bin = [bin[i] for i in
                                   np.random.choice(
                                       len(bin), n_sample_bin, replace=True)]
            else:
                experiences_bin = []
                while n_sample_bin > len(bin):
                    experiences_bin.extend(list(bin))
                    n_sample_bin -= len(bin)
                experiences_bin.extend(
                    list(
                        np.random.choice(
                            list(bin), n_sample_bin, replace=False)
                    )
                )
            # append bin samples to cumulative sample list
            experiences.extend(experiences_bin)
        return experiences

    def sample_all(self):
        all_edgs = []
        for bin_val in self.bins.values():
            all_edgs.extend(bin_val)
        return all_edgs

    def bin_hash(self, experience):
        if self.sample_strat == 'q_max':
            bin_key = int(experience.edge.state_next.v_search_tree)  # must be int to hash
        elif self.sample_strat == 'sg':
            bin_key = int(len(experience.edge.state_prev.nn_map))
        elif self.sample_strat is None:
            bin_key = 1
        else:
            assert False
        return bin_key

    def empty(self):
        # empty everything
        self.bins = defaultdict(list)
        self.history = []

    def list2dict(self, l):
        # fast list2dict implementation (basically wc, word count)
        # [aaa, bbb, bbb, ccc, aaa, abc, bbb]
        #   -> {aaa:2, bbb:3, ccc:1, abc:1}
        return dict(Counter(l))
