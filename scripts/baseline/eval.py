"""TRAINING
"""
import os
import pudb
import logging
import warnings
import argparse
import torch
import torch.nn.functional as F
from os.path import join
from torch.utils.data import DataLoader
from sklearn import metrics
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import yaml
import json
from prefetch_generator import BackgroundGenerator
from importlib import import_module
from model.network.builder import build_network
from data_loader.dataset.builder import build_dataset
from base.base_trainer import BaseTrainer
from utils import AccAverageMeter


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


class Validater(BaseTrainer):
    def __init__(self, config=None):
        """ Base validater for all experiments.  """

        #######################################################################
        # Device setting
        #######################################################################
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True

        #######################################################################
        # Experiment setting
        #######################################################################
        self.experiment_config = config['experiment']
        self.exp_name = self.experiment_config['name']
        self.user_root = self.user_roots[os.environ['DEVICE']]
        self.resume = self.experiment_config['resume']
        self.resume_fpath = join(
            self.user_root, 'Experiments',
            self.experiment_config['resume_fpath']
        )

        self.checkpoint = self.resume_checkpoint()
        self.test_epoch = self.checkpoint['epoch']

        self.save_dir = join(self.user_root, 'Experiments', self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_fpath = join(self.save_dir,
                              f'eval_epoch{self.test_epoch}.log')
        print(f'Evaluate log fpath: "{self.log_fpath}"')

        logging.basicConfig(
                filename=self.log_fpath,
                filemode='w',
                format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]:'
                ' %(message)s',
                level=logging.INFO
        )

        exp_init_log = '\nEvaluate experiment @epoch{}: {}\nSave dir: "{}"\n'\
            ''.format(
                self.test_epoch,
                self.exp_name,
                self.save_dir,
            )
        self.logging_print(exp_init_log)

        #######################################################################
        # Dataset setting
        #######################################################################
        self.eval_transform_config = config['eval_transform']
        self.evalset_config = config['eval_dataset']

        #######################################################################
        # Dataloader setting
        #######################################################################
        self.evalloader_config = config['evalloader']
        self.evalloader_name = self.evalloader_config['name']
        self.evalloader_param = self.evalloader_config['param']
        self.eval_batch_size = self.evalloader_param['batch_size']
        self.eval_num_workers = self.evalloader_param['num_workers']

        #######################################################################
        # Network setting
        #######################################################################
        self.network_config = config['network']
        self.network_name = self.network_config['name']
        self.network_param = self.network_config['param']

    def evaluate(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        eval_transform = self.init_transform(self.eval_transform_config)
        evalset = self.init_dataset(self.evalset_config, eval_transform)
        self.evalloader = DataLoaderX(
            evalset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
            pin_memory=False,
            drop_last=False,
        )

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model()

        #######################################################################
        # Start evaluating
        #######################################################################
        eval_pbar = tqdm(
            total=len(self.evalloader),
            desc='Evaluate'
        )

        self.model.eval()

        all_labels = []
        all_probs = []
        all_preds = []
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.evalloader):
                batch_imgs = batch_imgs.cuda()
                batch_labels = batch_labels.cuda()
                batch_probs = self.model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]

                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_probs.extend(batch_probs.cpu().numpy().tolist())
                all_preds.extend(batch_preds.cpu().numpy().tolist())

                eval_pbar.update()

        eval_acc = metrics.accuracy_score(all_labels, all_preds)
        eval_mr = metrics.recall_score(all_labels, all_preds,
                                       average='macro')
        eval_ap = metrics.precision_score(all_labels, all_preds,
                                          average='macro')

        target_names = list(evalset.class_to_idx.keys())  # Specify for CIFAR10
        labels = list(range(10))
        classify_report = metrics.classification_report(
            all_labels, all_preds, labels=labels, target_names=target_names
        )

        eval_pbar.set_postfix_str(
            'Acc:{:.0%} MR:{:.0%} AP:{:.0%}'.format(eval_acc, eval_mr, eval_ap)
        )
        eval_pbar.close()

        self.logging_print(classify_report)

        # save true_list and pred_list
        np.save(
            join(self.save_dir, f"eval_epoch{self.test_epoch}_label_list.npy"),
            np.array(all_labels)
        )
        np.save(
            join(self.save_dir, f"eval_epoch{self.test_epoch}_pred_list.npy"),
            np.array(all_preds)
        )
        np.save(
            join(self.save_dir, f"eval_epoch{self.test_epoch}_prob_list.npy"),
            np.array(all_probs)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fpath", type=str)
    args = parser.parse_args()
    return args


def main(args):
    warnings.filterwarnings('ignore')
    with open(args.config_fpath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validater = Validater(config=config)
    validater.evaluate()


if __name__ == '__main__':
    args = parse_args()
    main(args)
