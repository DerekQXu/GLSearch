from config import FLAGS
from utils import get_ts, create_dir_if_not_exists, save
from utils_our import get_our_dir, get_model_info_as_str, \
    get_model_info_as_command, extract_config_code, plot_scatter_line, plot_dist
from tensorboardX import SummaryWriter
from collections import OrderedDict
from pprint import pprint
from os.path import join
import torch
import networkx as nx
import numpy as np

class Saver(object):
    def __init__(self):
        model_str = self._get_model_str()
        self.logdir = join(
            get_our_dir(),
            'logs',
            '{}_{}'.format(model_str, get_ts()))
        create_dir_if_not_exists(self.logdir)
        self.writer = SummaryWriter(self.logdir)
        self.model_info_f = self._open('model_info.txt')
        self.plotdir = join(self.logdir, 'plot')
        self.objdir = join(self.logdir, 'obj')
        self._log_model_info()
        self._save_conf_code()
        print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

    def close(self):
        self.writer.close()
        if hasattr(self, 'log_f'):
            self.log_f.close()
        if hasattr(self, 'results_f'):
            self.results_f.close()

    def get_log_dir(self):
        return self.logdir

    def get_plot_dir(self):
        create_dir_if_not_exists(self.plotdir)
        return self.plotdir

    def get_obj_dir(self):
        create_dir_if_not_exists(self.objdir)
        return self.objdir

    def log_list_of_lists_to_csv(self, lol, fn, delimiter=','):
        import csv
        fp = open(join(self.logdir, fn), 'a+')
        csv_writer = csv.writer(fp, delimiter=delimiter)
        for l in lol:
            csv_writer.writerow(l)
        fp.close()

    def log_model_architecture(self, model):
        self.model_info_f.write('{}\n'.format(model))
        self.model_info_f.close()  # TODO: check if future if we write more to it

    def log_info(self, s):
        print(s)
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write('{}\n'.format(s))

    def log_info_new_file(self, s, fn):
        # print(s)
        log_f = open(join(self.logdir, fn), 'a')
        log_f.write('{}\n'.format(s))
        log_f.close()

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(extract_config_code())
        p = join(self.get_log_dir(), 'FLAGS')
        save({'FLAGS': FLAGS}, p, print_msg=False)

    # def save_flags(self, fn):
    #     p = join(self.get_log_dir(), fn)
    #     save({'FLAGS': FLAGS}, p, print_msg=False)

    def save_trained_model(self, trained_model, ext=''):
        model_dir = join(self.logdir, 'models')
        create_dir_if_not_exists(model_dir)
        p = join(model_dir, 'trained_model{}.pt'.format(ext))
        torch.save(trained_model.state_dict(), p)
        print('Trained model saved to {}'.format(p))


    def log_bd_stats(self, bd_stats_dict):
        for pair_id, bd_stats_list in bd_stats_dict.items():
            for iteration, bd_stats in enumerate(bd_stats_list):
                bd_stats_connected_arr = self.flatten_list_tuple_into_np_arr(
                    bd_stats.bd_stats_connected)
                bd_stats_unconnected_arr = self.flatten_list_tuple_into_np_arr(
                    bd_stats.bd_stats_unconnected)
                if len(bd_stats_connected_arr) > 1:
                    plot_dist(
                        bd_stats_connected_arr,f'{pair_id}_{iteration}_conn',self.get_plot_dir(), saver=self)
                if len(bd_stats_unconnected_arr) > 1:
                    plot_dist(
                        bd_stats_unconnected_arr,f'{pair_id}_{iteration}_unconn', self.get_plot_dir(), saver=self)

    def flatten_list_tuple_into_np_arr(self, li):
        li_flatten = []
        for elt in li:
            for elt_elt in elt:
                li_flatten.append(elt_elt)
        return np.array(li_flatten)

    def log_scatter_mcs(self, cur_id, iter, result_d):
        sp = join(self.get_obj_dir(), f'{iter}_val')
        save(result_d, sp)
        self._save_to_result_file(f'iter {iter} val result', to_print=True)
        for label, data_dict in result_d.items():
            if FLAGS.val_debug:
                g1, g2 = data_dict['g1'], data_dict['g2']
                nx.write_gexf(g1, join(self.get_obj_dir(), f'{cur_id}_{g1.graph["gid"]}.gexf'))
                nx.write_gexf(g2, join(self.get_obj_dir(), f'{cur_id}_{g2.graph["gid"]}.gexf'))
            plot_scatter_line(data_dict['result'], label, self.get_plot_dir())

        for label, data_dict in result_d.items():
            for model_name, data_dict_elt in data_dict['result'].items():
                incumbent_size_list = data_dict_elt['incumbent_data']
                runtime = data_dict_elt['runtime']  
                self._save_to_result_file(f'  num_iters={len(incumbent_size_list)}, '
                                          f'runtime={runtime}, '
                                          f'mcs={incumbent_size_list[-1]}, '
                                          f'method={model_name}', to_print=True)

    def save_graph_as_gexf(self, g, fn):
        nx.write_gexf(g, join(self.get_obj_dir(), fn))

    def save_eval_result_dict(self, result_dict, label):
        self._save_to_result_file(label)
        self._save_to_result_file(result_dict)

    def save_pairs_with_results(self, test_data, train_data, info):
        p = join(self.get_log_dir(), '{}_pairs'.format(info))
        save({'test_data_pairs':
                  self._shrink_space_pairs(test_data.dataset.pairs),
              # 'train_data_pairs':
              # self._shrink_space_pairs(train_data.dataset.pairs)
              },
             p, print_msg=False)

    def save_ranking_mat(self, true_m, pred_m, info):
        p = join(self.get_log_dir(), '{}_ranking_mats'.format(info))
        save({'true_m': true_m.__dict__, 'pred_m': pred_m.__dict__},
             p, print_msg=False)

    def save_global_eval_result_dict(self, global_result_dict):
        p = join(self.get_log_dir(), 'global_result_dict')
        save(global_result_dict, p, print_msg=False)

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    # def clean_up_saved_models(self, best_iter):
    #     for file in glob('{}/models/*'.format(self.get_log_dir())):
    #         if str(best_iter) not in file:
    #             system('rm -rf {}'.format(file))

    def save_exception_msg(self, msg):
        with self._open('exception.txt') as f:
            f.write('{}\n'.format(msg))

    def _get_model_str(self):
        li = []
        key_flags = [FLAGS.model, FLAGS.dataset]
        for f in key_flags:
            li.append(str(f))
        return '_'.join(li)

    def _log_model_info(self):
        s = get_model_info_as_str()
        c = get_model_info_as_command()
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.write(c)
        self.model_info_f.write('\n\n')
        self.writer.add_text('model_info_str', s)
        self.writer.add_text('model_info_command', c)

    def log_new_FLAGS_to_model_info(self):
        self.model_info_f.write('----- new model info after loading\n')
        self._log_model_info()

    def _save_to_result_file(self, obj, name=None, to_print=False):
        if not hasattr(self, 'results_f'):
            self.results_f = self._open('results.txt')
        if type(obj) is dict or type(obj) is OrderedDict:
            # self.f.write('{}:\n'.format(name))
            # for key, value in obj.items():
            #     self.f.write('\t{}: {}\n'.format(key, value))
            pprint(obj, stream=self.results_f)
        elif type(obj) is str:
            if to_print:
                print(obj)
            self.results_f.write('{}\n'.format(obj))
        else:
            self.results_f.write('{}: {}\n'.format(name, obj))
        self.results_f.flush()

    def _shrink_space_pairs(self, pairs):
        for _, pair in pairs.items():
            # print(pair.__dict__)
            pair.shrink_space_for_save()
            # pass
            # print(pair.__dict__)
            # exit(-1)
        return pairs


saver = Saver()  # can be used by `from saver import saver`
