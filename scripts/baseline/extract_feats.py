"""TRAINING
"""
import os
import shutil
import h5py
import warnings
import argparse
import yaml
import torch
import torch.nn.functional as F
# from pudb import set_trace
from os.path import join
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        self.exp_root = join(self.user_root, 'Experiments')
        self.resume = self.experiment_config['resume']
        self.total_epochs = self.experiment_config['total_epochs']
        # 默认取验证集的特征进行降维可视化
        print(f'===> Starting extracting feature of {self.exp_name}...')

        if '/' in self.experiment_config['resume_fpath']:
            self.resume_fpath = self.experiment_config['resume_fpath']
        else:
            self.resume_fpath = join(self.exp_root, self.exp_name,
                                     self.experiment_config['resume_fpath'])

        embedding_dir = join(self.exp_root, 'Tensorboard', self.exp_name,
                             'Embedding')
        if os.path.exists(embedding_dir):
            shutil.rmtree(embedding_dir)

        self.writer = SummaryWriter(log_dir=embedding_dir)

        self.checkpoint, resume_log = self.resume_checkpoint()

        self.save_dir = join(self.user_root, 'Experiments', self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self._set_configs(config)

    def extract(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        train_transform = self.init_transform(self.train_transform_config,
                                              log_file=False)
        trainset = self.init_dataset(self.trainset_config,
                                     train_transform,
                                     log_file=False)
        self.reform_mean = torch.tensor(trainset.mean).view(1, 3, 1, 1)
        self.reform_std = torch.tensor(trainset.std).view(1, 3, 1, 1)
        self.trainloader = DataLoaderX(
            trainset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.train_num_workers,
            pin_memory=True,
            drop_last=False,
        )

        val_transform = self.init_transform(self.val_transform_config,
                                            log_file=False)
        valset = self.init_dataset(self.valset_config,
                                   val_transform,
                                   log_file=False)

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
        self.model = self.init_model(log_file=False)

        #######################################################################
        # Start evaluating
        #######################################################################
        self.extract_one_phase(self.trainloader, phase='train')
        self.extract_one_phase(self.valloader, phase='val')

    def extract_one_phase(self, dataloader, phase):
        pbar = tqdm(total=len(dataloader), desc=f'Extracting {phase} features')

        all_imgs = []
        all_feats = []
        all_labels = []
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(dataloader):
                # collect labels
                batch_ori_imgs = batch_imgs * self.reform_std +\
                        self.reform_mean
                batch_resized_imgs = F.interpolate(batch_ori_imgs,
                                                   size=(112, 112),)
                all_imgs.append(batch_resized_imgs.detach())
                all_labels.append(batch_labels.detach())
                # batch_labels:torch.size([B])

                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_feats = self.model(batch_imgs, embedding=True)
                all_feats.append(batch_feats.detach().cpu())

                batch_prob = self.model.fc(batch_feats)
                batch_preds = batch_prob.max(1)[1]
                all_preds.append(batch_preds.detach().cpu())
                pbar.update()
        pbar.close()

        # set_trace()
        all_imgs = torch.vstack(all_imgs)
        all_feats = torch.vstack(all_feats)
        all_labels = torch.hstack(all_labels).numpy()
        all_preds = torch.hstack(all_preds).numpy()

        self.writer.add_embedding(mat=all_feats,
                                  metadata=all_labels,
                                  label_img=all_imgs,
                                  tag=f'{phase}_GT',
                                  global_step=self.total_epochs,)
        self.writer.add_embedding(mat=all_feats,
                                  metadata=all_preds,
                                  label_img=all_imgs,
                                  tag=f'{phase}_Pred',
                                  global_step=self.total_epochs,)
        self.writer.close()

        feat_fpath = join(self.exp_root, self.exp_name,
                          f'{phase}_features-labels.h5')
        # save feature and labels
        if os.path.exists(feat_fpath):  # h5不能重新写入
            os.remove(feat_fpath)

        with h5py.File(feat_fpath, 'w') as f:
            f['features'] = all_feats
            f['labels'] = all_labels

        print(f'Features-labels file is saved at "{feat_fpath}"\n')


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
