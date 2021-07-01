from config import FLAGS
from layers_factory import create_layers
import torch.nn as nn
from time import time
from utils_our import get_branch_names, get_flag


class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()

        self.train_data = data
        assert type(data) is list and len(data) >= 1
        self.num_node_feat = data[0].num_node_feat

        # Create layers.
        # for _ in range(n_outputs): TODO: finish real multiple-choice learning
        self.layers = create_layers(self, 'layer', FLAGS.layer_num)
        assert (len(self.layers) > 0)
        self._print_layers(None, self.layers)

        # Create layers for branches
        # (except for the main branch which has been created above).
        bnames = get_branch_names()
        if bnames:
            for bname in bnames:
                blayers = create_layers(
                    self, bname,
                    get_flag('{}_layer_num'.format(bname), check=True))
                setattr(self, bname, blayers)  # so that print(model) can print
                self._print_layers(bname, getattr(self, bname))

        # Map layer object to output.
        # Some layers may not register its output to this dictionary.
        # Currently only NodeEmbedding layers store its output.
        # This dictionary does NOT get cleaned up after each iteration.
        self.layer_output = {}

        # # Initialization.
        # for m in self.modules():
        #     # if isinstance(m, GraphConv): # TODO: check
        #     m.weight.data = init.xavier_uniform(
        #         m.weight.data, gain=nn.init.calculate_gain('relu'))
        #     if m.bias is not None:
        #         m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, cur_id, iter, batch_data):
        t = time()
        # total_loss = 0.0
        # Go through each layer except the last one.
        # acts = [self._get_ins(self.layers[0])]
        md = batch_data.merge_data['merge']
        acts = [md.x]
        # print('Forwarding...')
        for k, layer in enumerate(self.layers):
            ln = layer.__class__.__name__
            # print('\t{}'.format(ln))
            if ln == "GraphConvolutionCollector":
                gcn_num = layer.gcn_num
                ins = acts[-gcn_num:]
            else:
                ins = acts[-1]
            outs = layer(ins, batch_data, self, iter=iter, cur_id=cur_id)
            acts.append(outs)
        # x = data.batch.numpy()
        # print(x)
        total_loss = acts[-1]
        self._forward_for_branches(acts, total_loss, batch_data)
        if not self.training:  # TODO: check; only record time for testing pairs
            for pair in batch_data.pair_list:
                # Divide by the batch size and the running time is not precisely per-pair based.
                pair.assign_pred_time((time() - t) * 1000 / FLAGS.batch_size)  # msec
        return total_loss

    def _forward_for_branches(self, acts, total_loss, batch_data):
        bnames = get_branch_names()
        if not bnames:  # no other branch besides the main branch (i.e. layers)
            return total_loss
        for bname in bnames:
            blayers = getattr(self, bname)
            ins = acts[get_flag('{}_start'.format(bname))]
            outs = None
            assert len(blayers) >= 1
            for layer in blayers:
                outs = layer(ins, batch_data, self)
                ins = outs
            total_loss += get_flag('{}_loss_alpha'.format(bname)) * outs
        return total_loss

    def store_layer_output(self, layer, output):
        self.layer_output[layer] = output

    def get_layer_output(self, layer):
        return self.layer_output[layer]  # may get KeyError/ValueError

    def _print_layers(self, branch_name, layers):
        print('Created {} layers{}: {}'.format(
            len(layers),
            '' if branch_name is None else ' for branch {}'.format(branch_name),
            ', '.join(l.__class__.__name__ for l in layers)))
