"""trainer script """
import random
import warnings
import argparse
import yaml
import numpy as np
import torch
# from pudb import set_trace
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from prefetch_generator import BackgroundGenerator
# Distribute Package
from apex import amp
from torch.nn.parallel import DistributedDataParallel
# Custom Package
from base.base_trainer import BaseTrainer
from utils import AccAverageMeter, rotation


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class Trainer(BaseTrainer):
    def __init__(self, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)
        # adv_config = config['adv']
        # self.adv_name = adv_config['name']
        # self.adv_param = adv_config['param']

    def train(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        # set_trace()
        train_transform = self.init_transform(self.train_transform_config)
        trainset = self.init_dataset(self.trainset_config, train_transform)
        train_sampler = self.init_sampler(trainset)

        self.trainloader = DataLoaderX(
            trainset,
            batch_size=self.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.train_num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )

        if self.local_rank != -1:
            print(f'global_rank {self.global_rank}/{self.world_size},'
                  f'local_rank {self.local_rank},'
                  f'sampler "{self.train_sampler_name}"')

        if self.local_rank in [-1, 0]:
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
        # Initialize Loss
        #######################################################################
        self.criterion = self.init_loss()

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.optimizer = self.init_optimizer()

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################
        if self.local_rank != -1:
            self.model, self.optimizer = amp.initialize(
                self.model,
                self.optimizer,
                opt_level='O1'
            )
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.init_lr_scheduler()

        #######################################################################
        # Start Training
        #######################################################################
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            # learning rate decay by epoch
            if self.lr_scheduler_mode == 'epoch':
                self.lr_scheduler.step()

            if self.local_rank != -1:  # 多卡同步sampler生成的索引块
                train_sampler.set_epoch(epoch)

            train_mr, train_loss = self.train_epoch(epoch)

            if self.local_rank in [-1, 0]:
                val_mr, val_recalls, val_loss = self.evaluate(epoch)

                self.logger.debug(
                    'Epoch[{epoch:>3d}/{total_epochs}] '
                    'Trainset Loss={train_loss:.4f} MR={train_mr:.2%} || '
                    'Valset Loss={val_loss:.4f} MR={val_mr:.2%}'.format(
                        epoch=epoch,
                        total_epochs=self.total_epochs,
                        train_loss=train_loss,
                        train_mr=train_mr,
                        val_loss=val_loss,
                        val_mr=val_mr,
                    )
                )

                if len(val_recalls) <= 20 and epoch == self.total_epochs:
                    self.logger.info("Class recalls:{val_recalls}\n\n")

                # Save log by tensorboard
                self.writer.add_scalar(f'{self.exp_name}/LearningRate',
                                       self.optimizer.param_groups[0]['lr'],
                                       epoch)
                self.writer.add_scalars(f'{self.exp_name}/Loss',
                                        {'train_loss': train_loss,
                                         'val_loss': val_loss}, epoch)
                self.writer.add_scalars(f'{self.exp_name}/Recall',
                                        {'train_mr': train_mr,
                                         'val_mr': val_mr}, epoch)

                self.save_checkpoint(epoch, val_mr, val_recalls)

        if self.local_rank in [-1, 0]:
            self.logger.info(
                f"===> Result directory: '{self.save_dir}'\n"
                f"*********************************************************"
                f"*********************************************************"
            )

    def train_epoch(self, epoch):
        self.model.train()

        train_pbar = tqdm(
            total=len(self.trainloader),
            desc='Train Epoch[{:>3d}/{}]'.format(epoch, self.total_epochs)
        )

        all_labels = []
        all_preds = []
        train_loss_meter = AccAverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.trainloader):
            if self.lr_scheduler_mode == 'iteration':
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            # Self supervised learning
            # Step 1: generate rotated samples and labels
            batch_selfsp_imgs, batch_selfsp_labels = rotation(batch_imgs)

            batch_selfsp_imgs = batch_selfsp_imgs.cuda()
            batch_selfsp_labels = batch_selfsp_labels.cuda()
            batch_imgs = batch_imgs.cuda()
            batch_labels = batch_labels.cuda()

            # Step 2: train with rotated imgs
            batch_prob = self.model(batch_imgs)
            batch_selfsp_prob = self.model(batch_selfsp_imgs, ssp=True)

            sp_loss = self.criterion(batch_prob,
                                     batch_labels)
            selfsp_loss = self.criterion(batch_selfsp_prob,
                                         batch_selfsp_labels)

            # Add progressive training
            # Startly, mainly use ssp; Finally, use supervision progressively.
            sp_weight = epoch / self.total_epochs
            total_loss = sp_weight * sp_loss + (1 - sp_weight) * selfsp_loss
            # total_loss = sp_loss + selfsp_loss
            if self.local_rank != -1:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self._reduce_loss(total_loss)
            else:
                total_loss.backward()
                self.optimizer.step()

            batch_pred = batch_prob.max(1)[1]
            train_loss_meter.update(total_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str(
                    'LR:{:.1e} Loss:{:.4f}'.format(
                        self.optimizer.param_groups[0]['lr'],
                        train_loss_meter.avg,
                    )
                )

        train_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        train_pbar.set_postfix_str('LR:{:.1e} Loss:{:.2f} MR:{:.2%}'.format(
                self.optimizer.param_groups[0]['lr'],
                train_loss_meter.avg, train_mr))
        train_pbar.close()

        return train_mr, train_loss_meter.avg

    def evaluate(self, epoch):
        self.model.eval()

        val_pbar = tqdm(
            total=len(self.valloader),
            ncols=0,
            desc='                Val'
        )

        all_labels = []
        all_preds = []
        val_loss_meter = AccAverageMeter()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.valloader):
                batch_imgs = batch_imgs.cuda()
                batch_labels = batch_labels.cuda()
                batch_probs = self.model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]
                avg_loss = self.criterion(batch_probs, batch_labels)
                val_loss_meter.update(avg_loss.item(), 1)

                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_preds.extend(batch_preds.cpu().numpy().tolist())

                val_pbar.update()

        val_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        val_recalls = metrics.recall_score(all_labels, all_preds, average=None)
        val_recalls = np.around(val_recalls, decimals=2).tolist()

        val_pbar.set_postfix_str('Loss:{:.2f} MR:{:.2%}'.format(
            val_loss_meter.avg, val_mr))
        val_pbar.close()

        return val_mr, val_recalls, val_loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='Local Rank for\
                        distributed training. if single-GPU, default: -1')
    parser.add_argument('--config_path', type=str, help='path of config file')
    args = parser.parse_args()
    return args


def _set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    warnings.filterwarnings('ignore')
    _set_seed()
    # set_trace()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
