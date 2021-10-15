"""TRAINING
"""
import os
import warnings
import argparse
import torch
import yaml
import numpy as np
# import pandas as pd
# from pudb import set_trace
from os.path import join
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from base.base_trainer import BaseTrainer
from utils.utils import get_cm_with_labels


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

        if '/' in self.experiment_config['resume_fpath']:
            self.resume_fpath = self.experiment_config['resume_fpath']
        else:
            self.resume_fpath = join(
                self.user_root, 'Experiments', self.exp_name,
                self.experiment_config['resume_fpath']
            )

        self.checkpoint, resume_log = self.resume_checkpoint()
        self.test_epoch = self.checkpoint['epoch']

        self.save_dir = join(self.user_root, 'Experiments', self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        if 'best' in self.resume_fpath:
            self.log_fname = f'eval_best_epoch{self.test_epoch}.log'
        else:
            self.log_fname = f'eval_epoch{self.test_epoch}.log'
        self.log_fpath = join(self.save_dir, self.log_fname)
        self.logger = self.init_logger(self.log_fpath)
        self.logger.info(resume_log)

        exp_init_log = '===> Evaluate experiment "{}" @epoch{}\n'\
            'Log filepath: "{}"\n'.format(
                self.exp_name,
                self.test_epoch,
                self.log_fpath,
            )
        self.logger.info(exp_init_log)

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
        eval_pbar.set_postfix_str(
            'Acc:{:.2%} MR:{:.2%} AP:{:.2%}'.format(eval_acc, eval_mr, eval_ap)
        )
        eval_pbar.close()

        classification_report = metrics.classification_report(
            all_labels, all_preds, target_names=evalset.classes
        )
        self.logger.info(
            "===> Classification Report:\n" + classification_report
        )

        cm_df = get_cm_with_labels(all_labels, all_preds, evalset.classes)
        self.logger.info(
            '===> Confusion Matrix:\n' + cm_df.to_string() + '\n'
        )

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

        save_log = f"Results are saved at '{self.save_dir}'.\n"\
            f"===> Ended Evaluation of experiment '{self.exp_name}'"\
            f" @epoch{self.test_epoch}.\n"\
            f"*********************************************************"\
            f"*********************************************************\n\n"
        self.logger.info(save_log)


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
