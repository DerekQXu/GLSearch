from config import FLAGS
from utils_our import get_flag
from utils import OurTimer
from torch.utils.data import DataLoader
from batch import BatchData
import torch
import torch.nn as nn

def run(data, saver, model):
    assert FLAGS.model == 'MCSRL_backtrack'
    assert type(data) is list and len(data) >= 1
    saver.log_model_architecture(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    num_iters_total = 0
    num_iters_total_limit = 0

    for curriculum_id, curriculum_data in enumerate(data):
        num_iters_total_limit += FLAGS.dataset_list[curriculum_id][1]
        stop = False
        while not stop:
            loss, num_iters_total, stop = _run_epoch(
                curriculum_id, num_iters_total, curriculum_data,
                model, optimizer, saver, num_iters_total_limit)
    return model


def _run_epoch(cur_id, num_iters_total, data, model, optimizer, saver,
                 num_iters_total_limit):
    epoch_timer = OurTimer()
    iter_timer = OurTimer()
    data_loader = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle_input)
    total_loss = 0
    num_iters = 0
    stop = False

    for iter, batch_gids in enumerate(data_loader):
        if num_iters_total % int(FLAGS.periodic_save) == 0:
            if FLAGS.save_model:
                saver.save_trained_model(model, ext='_num_iter_{}'.format(num_iters_total))

        if num_iters == FLAGS.only_iters_for_debug:
            stop = True
            break
        if num_iters_total_limit is not None and \
                num_iters_total == num_iters_total_limit:
            stop = True
            break

        batch_data = BatchData(batch_gids, data.dataset)
        loss = _run_iter(cur_id,
            num_iters_total, num_iters_total, batch_data, model,
            optimizer, saver, iter_timer)
        # print(iter, batch_data, batch_data.num_graphs, len(loader.dataset))
        total_loss += loss
        num_iters += 1
        num_iters_total += 1
    if not stop and num_iters_total > 0:
        saver.log_info('Curriculum: {:03d} , Loss: {:.7f} ({} iters)\t\t{}'.format(
            cur_id, total_loss / num_iters, num_iters,
            epoch_timer.time_and_clear()))
    return total_loss, num_iters_total, stop


def _run_iter(cur_id, iter, num_iters_total, batch_data, model,
                optimizer, saver, iter_timer):
    model.train()
    model.zero_grad()
    loss = model(cur_id, iter, batch_data)
    if FLAGS.clipping_val > 0:
        nn.utils.clip_grad_norm_(model.parameters(), FLAGS.clipping_val)
    if loss is None:
        return 0.0
    if FLAGS.retain_graph:
        loss.backward(retain_graph=True)
    else:
        loss.backward()
    optimizer.step()
    saver.writer.add_scalar('loss/loss', loss, num_iters_total)
    loss = loss.item()
    if iter == 0 or (iter + 1) % FLAGS.print_every_iters == 0:
        saver.log_info('\tIter: {:03d} ({}), Loss: {:.7f}\t\t{}'.format(
            iter + 1, num_iters_total + 1, loss,
            iter_timer.time_and_clear()))
    return loss