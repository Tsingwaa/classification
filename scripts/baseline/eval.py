"""TRAINING
"""
import os
import warnings
import argparse
import torch
import yaml
import numpy as np
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
        self.experiment_config = config['experiment']
        self.exp_name = self.experiment_config['name']
        self.user_root = os.environ['HOME']
        self.resume = self.experiment_config['resume']

        if '/' in self.experiment_config['resume_fpath']:
            self.resume_fpath = self.experiment_config['resume_fpath']
        else:
            self.resume_fpath = join(
                self.user_root, 'Experiments', self.exp_name,
                self.experiment_config['resume_fpath'])

        self.checkpoint, resume_log = self.resume_checkpoint()
        self.test_epoch = self.checkpoint['epoch']

        self.save_dir = join(self.user_root, 'Experiments', self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        if 'best' in self.resume_fpath:
            self.log_fname = f'val_best_epoch{self.test_epoch}.log'
        else:
            self.log_fname = f'val_epoch{self.test_epoch}.log'
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
        self.val_transform_config = config['val_transform']
        self.valset_config = config['val_dataset']

        #######################################################################
        # Dataloader setting
        #######################################################################
        self.valloader_config = config['valloader']
        self.valloader_name = self.valloader_config['name']
        self.valloader_param = self.valloader_config['param']
        self.val_batch_size = self.valloader_param['batch_size']
        self.val_num_workers = self.valloader_param['num_workers']

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
        val_transform = self.init_transform(self.val_transform_config)
        valset = self.init_dataset(self.valset_config, val_transform)
        self.valloader = DataLoaderX(
            valset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=False,
        )

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model()

        #######################################################################
        # Start evaluating
        #######################################################################
        val_pbar = tqdm(total=len(self.valloader), desc='Evaluate')

        self.model.eval()

        all_labels = []
        all_probs = []
        all_preds = []
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.valloader):
                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
                batch_probs = self.model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]

                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_probs.extend(batch_probs.cpu().numpy().tolist())
                all_preds.extend(batch_preds.cpu().numpy().tolist())

                val_pbar.update()

        val_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        val_pbar.set_postfix_str('MR:{:.2%}'.format(val_mr))
        val_pbar.close()

        classification_report = metrics.classification_report(
            all_labels, all_preds, target_names=valset.classes
        )

        self.logger.info(
            "===> Classification Report:\n" + classification_report
        )

        if len(valset.classes) <= 20:
            cm_df = get_cm_with_labels(all_labels, all_preds, valset.classes)
            self.logger.info(
                '===> Confusion Matrix:\n' + cm_df.to_string() + '\n'
            )

        # save true_list and pred_list
        np.save(
            join(self.save_dir, f"val_epoch{self.test_epoch}_label_list.npy"),
            np.array(all_labels)
        )
        np.save(
            join(self.save_dir, f"val_epoch{self.test_epoch}_pred_list.npy"),
            np.array(all_preds)
        )
        np.save(
            join(self.save_dir, f"val_epoch{self.test_epoch}_prob_list.npy"),
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
    parser.add_argument('--local_rank', type=int, help='Local Rank for\
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
