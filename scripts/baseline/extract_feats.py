"""TRAINING
"""
import os
import h5py
import warnings
import argparse
import torch
import yaml
import numpy as np
# from pudb import set_trace
from os.path import join
from torch.utils.data import DataLoader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from base.base_trainer import BaseTrainer


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


class Extractor(BaseTrainer):
    def __init__(self, local_rank, config):
        """Extractor to extract feature"""

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
        self.extract_phase = self.experiment_config['extract_phase']

        if '/' in self.experiment_config['resume_fpath']:
            self.resume_fpath = self.experiment_config['resume_fpath']
        else:
            self.resume_fpath = join(
                self.user_root, 'Experiments', self.exp_name,
                self.experiment_config['resume_fpath'])

        self.feat_fpath = join(self.user_root, 'Experiments', self.exp_name,
                               'features-labels.h5')

        self.checkpoint, resume_log = self.resume_checkpoint()

        self.save_dir = join(self.user_root, 'Experiments', self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        #######################################################################
        # Dataloader setting
        #######################################################################
        if self.extract_phase == 'val':
            self.transform_config = config['val_transform']
            self.dataset_config = config['val_dataset']
            self.dataloader_config = config['valloader']
        else:
            self.transform_config = config['train_transform']
            self.dataset_config = config['train_dataset']
            self.dataloader_config = config['trainloader']

        self.dataloader_param = self.dataloader_config['param']
        self.batch_size = self.dataloader_param['batch_size']
        self.num_workers = self.dataloader_param['num_workers']

        #######################################################################
        # Network setting
        #######################################################################
        self.network_config = config['network']
        self.network_name = self.network_config['name']
        self.network_param = self.network_config['param']

    def extract(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        print('\n')
        transform = self.init_transform(self.transform_config, log_file=False)
        dataset = self.init_dataset(self.dataset_config, transform,
                                    log_file=False)
        self.dataloader = DataLoaderX(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(log_file=False)

        #######################################################################
        # Start evaluating
        #######################################################################
        print('\n\n')
        pbar = tqdm(total=len(self.dataloader), desc='Extracting')

        all_feats = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.dataloader):
                # collect labels
                all_labels.extend(batch_labels)

                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_feats = self.model(batch_imgs, embedding=True)

                # collect features
                all_feats.extend(batch_feats.cpu().numpy().tolist())

                pbar.update()
        pbar.close()

        # save feature and labels
        if not os.path.exists(self.feat_fpath):  # h5不能重新写入
            os.remove(self.feat_fpath)

        with h5py.File(self.feat_fpath, 'w') as f:
            f['features'] = np.array(all_feats)
            f['labels'] = all_labels

        print(f'\nFeatures-labels file is saved at "{self.feat_fpath}"\n')


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
    extractor = Extractor(local_rank=args.local_rank, config=config)
    extractor.extract()


if __name__ == '__main__':
    args = parse_args()
    main(args)
