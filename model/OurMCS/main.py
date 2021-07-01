#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import copy
import matplotlib.pyplot as plt
# from matplotlib_venn import venn2, venn3

from config import FLAGS
from model import Model
from data_model import load_train_test_data
from train import run
from utils_our import get_model_info_as_str, check_flags, \
    convert_long_time_to_str, load_replace_flags
from utils import slack_notify, get_ts, OurTimer
from saver import saver
from time import time
from os.path import basename
import torch
import traceback

import networkx as nx

def main():
    assert FLAGS.tvt_strategy == 'holdout'
    tm = OurTimer()
    # we always use the first data returned
    # load_model determines whether we are training or testing.
    train_test_data, _ = load_train_test_data()
    saver.log_info('Data loading: {}'.format(tm.time_and_clear()))

    if FLAGS.load_model is not None:
        print('Loading model: {}', format(FLAGS.load_model))
        load_replace_flags()
        saver.log_new_FLAGS_to_model_info()

    print('creating model...')
    model = Model(train_test_data).to(FLAGS.device)

    if FLAGS.load_model is not None:
        print('Loading model: {}', format(FLAGS.load_model))
        ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)
        model.load_state_dict(ld)
        print('Model loaded')
    else:
        print(model)

    saver.log_info('Model created: {}'.format(tm.time_and_clear()))
    trained_model = run(train_test_data, saver, model)
    saver.log_info('Model trained: {}'.format(tm.time_and_clear()))
    if FLAGS.save_model:
        saver.save_trained_model(trained_model)

    if FLAGS.batched_logging and saver.curriculum_info is not None:
        saver.log_info(f'============ batched-log ============')
        import numpy as np
        for cur_id, methods_data in saver.curriculum_info.items():
            for method, method_data in methods_data.items():
                sols = method_data['final_mcs_list']
                if len(sols) > 0:
                    sol = np.mean(np.array(sols))
                    saver.log_info(f'{cur_id}-{method}:\t{sol}')
        saver.log_info(f'=====================================')

    overall_time = convert_long_time_to_str(time() - t)
    print(overall_time)
    print(saver.get_log_dir())
    print(basename(saver.get_log_dir()))
    saver.save_overall_time(overall_time)
    saver.close()


if __name__ == '__main__':
    t = time()
    print(get_model_info_as_str())
    check_flags()
    try:
        main()
    except:
        traceback.print_exc()
        s = '\n'.join(traceback.format_exc(limit=-1).split('\n')[1:])
        saver.save_exception_msg(traceback.format_exc())
        # slack_notify('{}@{}: {} on {} {} error \n{}'.
        #              format(FLAGS.user, FLAGS.hostname, FLAGS.model,
        #                     FLAGS.dataset, get_ts(), s))
    else:
        pass
        # slack_notify('{}@{}: {} on {} {} complete'.
        #              format(FLAGS.user, FLAGS.hostname, FLAGS.model,
        #                     FLAGS.dataset, get_ts()))