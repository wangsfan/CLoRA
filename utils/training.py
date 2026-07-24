# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega,
# Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

import sys
import json
import numpy as np
import copy
import math
import os
from argparse import Namespace
from typing import Iterable, Optional
import logging
import torch
from tqdm.auto import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from models.utils.future_model import FutureModel

from utils import disable_logging
from utils.globals import GLOBALS
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint, can_save_and_exit
from utils.loggers import log_extra_metrics, Logger
from utils.schedulers import get_scheduler
from utils.stats import track_system_stats


def _to_device(name: str, x, device):
    if isinstance(x, torch.Tensor):
        if 'label' in name.lower() or 'target' in name.lower():
            return x.to(device, dtype=torch.long)
        return x.to(device)
    return x


def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       args: Namespace,
                       epoch: int,
                       pbar: tqdm,
                       system_tracker=None,
                       scheduler=None) -> int:

    global GLOBALS
    train_iter = iter(train_loader)
    i = 0

    while True:
        if GLOBALS['SHOULD_STOP']:
            logging.info("Training stopped by signal handler.")
            sys.exit(0)

        try:
            data = next(train_iter)
        except StopIteration:
            break

        if args.debug_mode and i > model.get_debug_iters():
            break
        if args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
            break

        inputs, labels, not_aug_inputs = data[0], data[1], data[2]
        inputs = inputs.to(model.device)
        labels = labels.to(model.device, dtype=torch.long)
        not_aug_inputs = not_aug_inputs.to(model.device)

        extra_fields = {
            train_loader.dataset.extra_return_fields[k]:
                _to_device(train_loader.dataset.extra_return_fields[k],
                           data[3 + k], model.device)
            for k in range(len(data) - 3)
        }

        loss = model.meta_observe(inputs, labels, not_aug_inputs,
                                  epoch=epoch, **extra_fields)

        assert not math.isnan(loss)

        if scheduler is not None and args.scheduler_mode == 'iter':
            scheduler.step()

        if args.code_optimization == 0 and 'cuda' in str(args.device):
            torch.cuda.synchronize()

        system_tracker()
        i += 1

        pbar.set_postfix({'loss': loss,
                          'lr': model.opt.param_groups[0]['lr']},
                         refresh=False)
        pbar.update()

    if scheduler is not None and args.scheduler_mode == 'epoch':
        scheduler.step()


@can_save_and_exit
def train(model: ContinualModel,
          dataset: ContinualDataset,
          args: Optional[Namespace] = None) -> None:

    # ==========================================================
    # Save path & txt logger
    # ==========================================================
    save_path = os.path.join(os.getcwd(), 'results',
                             f'{model.NAME}-{dataset.NAME}')
    os.makedirs(save_path, exist_ok=True)

    task_metric_txt = os.path.join(save_path,
                                   'per_task_acc_forgetting.txt')

    # write args only once
    if not os.path.exists(task_metric_txt):
        with open(task_metric_txt, 'w') as f:
            f.write('=' * 50 + '\n')
            f.write('Experiment Arguments\n')
            f.write('=' * 50 + '\n')
            for arg in sorted(vars(args)):
                f.write(f'{arg:<25}: {getattr(args, arg)}\n')
            f.write('-' * 50 + '\n')
            f.write('Task\tAvgAcc(Class-IL)\tForgetting(Class-IL)\n')

    # ==========================================================
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Main process PID: {os.getpid()}")

    GLOBALS['SHOULD_STOP'] = False

    if args is None:
        assert 'MAMMOTH_ARGS' in os.environ
        args = Namespace(**json.loads(os.environ['MAMMOTH_ARGS']))

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING,
                        dataset.NAME, model.NAME)

    model.net.to(model.device)
    torch.cuda.empty_cache()

    is_fwd_enabled = True
    can_compute_fwd_beforetask = True
    random_results_class, random_results_task = [], []

    with track_system_stats(logger, device=args.device) as system_tracker:

        results, results_mask_classes = [], []

        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        eval_dataset = dataset if not args.eval_future else get_dataset(args)

        for cur_task in range(start_task, end_task):

            model.net.train()
            train_loader, _ = dataset.get_data_loaders()

            if not issubclass(dataset.__class__, GCLDataset):
                assert issubclass(train_loader.dataset.__class__,
                                  MammothDatasetWrapper)

            # ---- Forward Transfer (before task)
            if can_compute_fwd_beforetask and args.enable_other_metrics:
                try:
                    r_cls, r_tsk = dataset.evaluate(model, dataset, last=True)
                    random_results_class.append(r_cls)
                    random_results_task.append(r_tsk)
                except Exception:
                    can_compute_fwd_beforetask = False

            model.meta_begin_task(dataset)

            # ---- Training
            if not args.inference_only and args.n_epochs > 0:

                scheduler = get_scheduler(model, args, reload_optim=True)

                epoch = 0
                n_iterations = model.args.n_epochs * len(train_loader)

                train_pbar = tqdm(total=n_iterations,
                                  disable=args.non_verbose)

                while epoch < model.args.n_epochs:

                    model.meta_begin_epoch(epoch, dataset)

                    train_single_epoch(
                        model, train_loader, args,
                        epoch, train_pbar,
                        system_tracker, scheduler
                    )

                    model.meta_end_epoch(epoch, dataset)
                    epoch += 1

                train_pbar.close()

            model.meta_end_task(dataset)

            # ==================================================
            # Evaluation
            # ==================================================
            accs = eval_dataset.evaluate(model, eval_dataset)
            results.append(accs[0])
            results_mask_classes.append(accs[1])

            avg_acc_class_il = float(np.mean(accs[0]))

            # =========================================
            # Print a_{j,i}
            # =========================================
            print(f"\nTask {cur_task + 1} Accuracy Matrix a_j_i:")

            for j, acc in enumerate(accs[0]):
                print(f"a_{{{j + 1},{cur_task + 1}}} = {acc:.4f}")

            print()

            # ---- Forgetting (class-il)
            from utils.metrics import forgetting
            forgetting = forgetting(results)

            # ==================================================
            # Write per-task metrics to txt
            # ==================================================
            with open(task_metric_txt, 'a') as f:

                acc_matrix_str = ', '.join(
                    [f'a_{{{j + 1},{cur_task + 1}}}={acc:.4f}'
                     for j, acc in enumerate(accs[0])]
                )

                f.write(f"{cur_task + 1}\t"
                        f"{acc_matrix_str}\t"
                        f"{avg_acc_class_il:.4f}\t"
                        f"{forgetting:.4f}\n")

            # ---- Logger & checkpoint
            eval_dataset.log(args, logger, accs, cur_task, dataset.SETTING)

            if args.savecheck:
                save_mammoth_checkpoint(
                    cur_task, end_task, args,
                    model,
                    results=[results,
                             results_mask_classes,
                             logger.dump()]
                )

        # ======================================================
        # Final metrics
        # ======================================================
        if args.enable_other_metrics:
            bwt, bwt_mask = logger.add_bwt(results, results_mask_classes)
            log_extra_metrics(args, bwt, bwt_mask,
                              'Backward Transfer', cur_task)

            fgt, fgt_mask = logger.add_forgetting(results,
                                                  results_mask_classes)
            log_extra_metrics(args, fgt, fgt_mask,
                              'Forgetting', cur_task)

        system_tracker.print_stats()
