"""trainer script """
import argparse
import random
import warnings

import numpy as np
import torch
import yaml
# Distribute Package
from apex import amp
# Custom Package
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
from sklearn import metrics
from torch.nn.parallel import DistributedDataParallel
# from pudb import set_trace
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter, get_weight_scheduler, rotation


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class Trainer(BaseTrainer):

    def __init__(self, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)
        self.weight_scheduler = self.network_param['weight_scheduler']
        self.truncate = self.network_param.get('truncate', 'None')

    def train(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        # set_trace()
        train_transform = self.init_transform(self.train_transform_config)
        trainset = self.init_dataset(self.trainset_config, train_transform)
        train_sampler = self.init_sampler(trainset)

        self.trainloader = DataLoaderX(trainset,
                                       batch_size=self.train_batch_size,
                                       shuffle=(train_sampler is None),
                                       num_workers=self.train_num_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       sampler=train_sampler)

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
        self.optimizer = self.init_optimizer(self.model)

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################

        if self.local_rank != -1:
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1')
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.local_rank],
                                                 output_device=self.local_rank,
                                                 find_unused_parameters=True)
        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.self(self.optimizer)

        #######################################################################
        # Start Training
        #######################################################################
        best_mr = 0.
        best_epoch = 1
        best_recalls = []

        for cur_epoch in range(self.start_epoch, self.total_epochs + 1):
            # learning rate decay by epoch

            if self.lr_scheduler_mode == 'epoch':
                self.lr_scheduler.step()

            if self.local_rank != -1:  # 多卡同步sampler生成的索引块
                train_sampler.set_epoch(cur_epoch)

            train_mr, train_loss = self.train_epoch(cur_epoch)

            if self.local_rank in [-1, 0]:
                val_mr, val_recalls, val_loss = self.evaluate(cur_epoch)

                self.logger.debug(
                    'Epoch[{epoch:>3d}/{total_epochs}] '
                    'Trainset Loss={train_loss:.4f} SP={sp_loss:.4f} '
                    'SSP={ssp_loss:.4f} MR={train_mr:.2%} || '
                    'Valset Loss={val_loss:.4f} MR={val_mr:.2%}'.format(
                        epoch=cur_epoch,
                        total_epochs=self.total_epochs,
                        train_loss=train_loss['total'],
                        sp_loss=train_loss['sp'],
                        ssp_loss=train_loss['ssp'],
                        train_mr=train_mr,
                        val_loss=val_loss,
                        val_mr=val_mr,
                    ))

                if len(val_recalls) <= 20 and cur_epoch == self.total_epochs:
                    self.logger.info(f"Class recalls:{val_recalls}\n\n")

                # Save log by tensorboard
                self.writer.add_scalar(f'{self.exp_name}/LearningRate',
                                       self.optimizer.param_groups[0]['lr'],
                                       cur_epoch)
                self.writer.add_scalars(
                    f'{self.exp_name}/Loss', {
                        'train_loss': train_loss['total'],
                        'sp_loss': train_loss['sp'],
                        'ssp_loss': train_loss['ssp'],
                        'val_loss': val_loss
                    }, cur_epoch)
                self.writer.add_scalars(f'{self.exp_name}/Recall', {
                    'train_mr': train_mr,
                    'val_mr': val_mr
                }, cur_epoch)

                is_best = val_mr > best_mr

                if is_best:
                    best_mr = val_mr
                    best_epoch = cur_epoch
                    best_recalls = val_recalls
                    self.logger.info(f"Best recalls now: {best_recalls}\n")
                self.save_checkpoint(cur_epoch, is_best, val_mr, val_recalls)

        if self.local_rank in [-1, 0]:
            self.logger.info(
                f"===> Best mean recall: {best_mr} (epoch{best_epoch})\n"
                f"Class recalls: {best_recalls}\n"
                f"===> Save directory: '{self.exp_dir}'\n"
                f"*********************************************************"
                f"*********************************************************")

    def train_epoch(self, cur_epoch):
        self.model.train()

        train_pbar = tqdm(total=len(self.trainloader),
                          desc='Train Epoch[{:>3d}/{}]'.format(
                              cur_epoch, self.total_epochs))

        all_labels = []
        all_preds = []
        train_loss_meter = AverageMeter()
        sp_loss_meter = AverageMeter()
        ssp_loss_meter = AverageMeter()

        for i, (batch_imgs, batch_labels) in enumerate(self.trainloader):
            if self.lr_scheduler_mode == 'iteration':
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            # Self supervised learning
            # Step 1: generate rotated samples and labels
            batch_ssp_imgs, batch_ssp_labels = rotation(batch_imgs)

            batch_ssp_imgs = batch_ssp_imgs.cuda(non_blocking=True)
            batch_ssp_labels = batch_ssp_labels.cuda(non_blocking=True)
            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)

            # Step 2: train with rotated imgs

            if self.truncate == 'fc':
                """Truncate the gradient of feature vector"""
                batch_featvec = self.model(batch_imgs,
                                           output_type='sp_featvec')
                batch_prob = self.model.fc(batch_featvec.detach())
            elif self.truncate == 'layer4':
                """Truncate the gradient of featmap3 from layer3"""
                batch_featmap3 = self.model(batch_imgs, output_type='featmap3')
                batch_featmap4 = self.model.layer4(batch_featmap3.detach())
                batch_featvec = self.model.avgpool(batch_featmap4)
                batch_featvec = torch.flatten(batch_featvec, 1)
                batch_prob = self.model.fc(batch_featvec)
            else:  # No truncate.
                batch_prob = self.model(batch_imgs, output_type='sp_logits')

            batch_ssp_prob = self.model(batch_ssp_imgs,
                                        output_type='ssp_logits')

            ssp_weight = get_weight_scheduler(
                cur_epoch,
                self.total_epochs,
                weight_scheduler=self.weight_scheduler)
            sp_loss = self.criterion(batch_prob, batch_labels)
            ssp_loss = self.criterion(batch_ssp_prob, batch_ssp_labels)
            total_loss = (1 - ssp_weight) * sp_loss + ssp_weight * ssp_loss

            if self.local_rank != -1:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self._reduce_tensor(total_loss)
            else:
                total_loss.backward()
                self.optimizer.step()

            batch_pred = batch_prob.max(1)[1]
            train_loss_meter.update(total_loss.item(), 1)
            sp_loss_meter.update(sp_loss.item(), 1)
            ssp_loss_meter.update(ssp_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str(
                    'LR:{:.1e} Loss:{:.2f} SP:{:.2f} SSP:{:.2f}'.format(
                        self.optimizer.param_groups[0]['lr'],
                        train_loss_meter.avg,
                        sp_loss_meter.avg,
                        ssp_loss_meter.avg,
                    ))

        train_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        train_pbar.set_postfix_str(
            'LR:{:.1e} Loss:{:.2f} SP:{:.2f} SSP:{:.2f} MR:{:.2%}'.format(
                self.optimizer.param_groups[0]['lr'], train_loss_meter.avg,
                sp_loss_meter.avg, ssp_loss_meter.avg, train_mr))
        train_pbar.close()

        train_loss_dict = {
            'total': train_loss_meter.avg,
            'sp': sp_loss_meter.avg,
            'ssp': ssp_loss_meter.avg,
        }

        return train_mr, train_loss_dict

    def evaluate(self, cur_epoch):
        self.model.eval()

        val_pbar = tqdm(total=len(self.valloader),
                        ncols=0,
                        desc='                Val')

        all_labels = []
        all_preds = []
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.valloader):
                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
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
    parser.add_argument('--local_rank',
                        type=int,
                        help='Local Rank for\
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
    torch.backends.cudnn.benchmark = True


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
