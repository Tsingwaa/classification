"""TRAINING
"""
import argparse
import math
import os
import warnings
from os.path import join

import numpy as np
import torch
import yaml
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
# from pudb import set_trace
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import get_cm_with_labels


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Validater(BaseTrainer):
    def __init__(self, local_rank, config):
        """ Base validater for all experiments.  """

        #######################################################################
        # Device setting
        #######################################################################
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True
        self.local_rank = local_rank

        #######################################################################
        # Experiment setting
        #######################################################################
        self.exp_config = config['experiment']
        self.exp_name = self.exp_config['name']
        self.user_root = os.environ['HOME']
        self.exp_root = join(self.user_root, 'Experiments')

        self.resume = self.exp_config['resume']
        if '/' in self.exp_config['resume_fpath']:
            self.resume_fpath = self.exp_config['resume_fpath']
        else:
            self.resume_fpath = join(self.exp_root, self.exp_name,
                                     self.exp_config['resume_fpath'])

        self.checkpoint, resume_log = self.resume_checkpoint(self.resume_fpath)
        self.test_epoch = self.checkpoint['epoch']

        self.save_dir = join(self.exp_root, self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        if 'best' in self.resume_fpath:
            self.log_fname = f'eval_best_epoch{self.test_epoch}.log'
        else:
            self.log_fname = f'eval_epoch{self.test_epoch}.log'
        self.log_fpath = join(self.save_dir, self.log_fname)
        self.logger = self.init_logger(self.log_fpath)
        self.log(resume_log)

        exp_init_log = '===> Evaluate experiment "{}" @epoch{}\n'\
            'Log filepath: "{}"\n'.format(self.exp_name,
                                          self.test_epoch,
                                          self.log_fpath,)
        self.logger.info(exp_init_log)

        self._set_configs(config)

    def evaluate(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        val_transform = self.init_transform(self.val_transform_name,
                                            **self.val_transform_params)
        valset = self.init_dataset(self.valset_name,
                                   transform=val_transform,
                                   **self.valset_params)
        num_classes = valset.cls_num
        valloader = DataLoaderX(
            valset,
            batch_size=self.val_batchsize,
            shuffle=False,
            num_workers=self.val_workers,
            pin_memory=True,
            drop_last=False,
        )

        #######################################################################
        # Initialize Network
        #######################################################################
        model = self.init_model(self.network_name, **self.network_params)

        #######################################################################
        # Start evaluating
        #######################################################################
        val_pbar = tqdm(total=len(valloader), desc='Evaluate')

        model.eval()

        all_labels, all_probs, all_preds = [], [], []
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(valloader):
                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
                batch_probs = model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]

                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_probs.extend(batch_probs.cpu().numpy().tolist())
                all_preds.extend(batch_preds.cpu().numpy().tolist())

                val_pbar.update()

        val_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        val_recalls = metrics.recall_score(all_labels, all_preds, average=None)

        head_classes = math.floor(num_classes / 3)
        tail_classes = head_classes
        val_group_recalls = [
            np.around(np.mean(val_recalls[:head_classes]), decimals=4),
            np.around(np.mean(val_recalls[head_classes:num_classes -
                                          tail_classes]),
                      decimals=4),
            np.around(np.mean(val_recalls[num_classes - tail_classes:]),
                      decimals=4)
        ]

        val_pbar.set_postfix_str(f"MR:{val_mr:.2%} "
                                 f"Head:{val_group_recalls[0]:.2%} "
                                 f"Mid:{val_group_recalls[1]:.2%} "
                                 f"Tail:{val_group_recalls[2]:.2%}")
        val_pbar.close()
        classification_report = metrics.classification_report(
            all_labels, all_preds, target_names=valset.classes)

        self.log("===> Classification Report:\n" + classification_report)

        if num_classes <= 20:
            cm_df = get_cm_with_labels(all_labels, all_preds, valset.classes)
            self.log('===> Confusion Matrix:\n' + cm_df.to_string() + '\n')

        # save true_list and pred_list
        np.save(
            join(self.save_dir, f"val_epoch{self.test_epoch}_label_list.npy"),
            np.array(all_labels))
        np.save(
            join(self.save_dir, f"val_epoch{self.test_epoch}_pred_list.npy"),
            np.array(all_preds))
        np.save(
            join(self.save_dir, f"val_epoch{self.test_epoch}_prob_list.npy"),
            np.array(all_probs))

        self.log(f"Results are saved at '{self.save_dir}'.\n"
                 f"===> Ended Evaluation of experiment '{self.exp_name}'"
                 f" @epoch{self.test_epoch}.\n"
                 f"*********************************************************"
                 f"*********************************************************")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        help='Local Rank for\
                        distributed training. if single-GPU, default: -1')
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()
    return args


def main(args):
    warnings.filterwarnings('ignore')
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validater = Validater(local_rank=args.local_rank, config=config)
    validater.evaluate()


if __name__ == '__main__':
    args = parse_args()
    main(args)
