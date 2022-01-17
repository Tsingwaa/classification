###############################################################################
# Copyright (C) 2021 All rights reserved.
# Filename: train_adapt.py
# Author: Tsingwaa
# Email: zengchh3@gmail.com
# Created Time : 2021-12-21 22:34 Tuesday
# Last modified: 2021-12-21 22:34 Tuesday
# Description:
#
###############################################################################

import argparse
# import math
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from apex import amp
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
from torch.nn.parallel import DistributedDataParallel
# from pudb import set_trace
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter, ExpStat, switch_clean, switch_mix


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class Trainer(BaseTrainer):

    def __init__(self, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)
        adv_config = config['adv']
        self.adv_name = adv_config['name']
        self.clean_weight = adv_config['clean_weight']
        self.step_size = adv_config['step_size']
        self.num_steps = adv_config['num_steps']
        self.eps = adv_config['eps']

    def train(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        # set_trace()
        train_transform = self.init_transform(self.train_transform_name,
                                              **self.train_transform_params)
        trainset = self.init_dataset(self.trainset_name,
                                     transform=train_transform,
                                     **self.trainset_params)
        train_sampler = self.init_sampler(self.train_sampler_name,
                                          dataset=trainset,
                                          **self.trainloader_params)
        self.trainloader = DataLoaderX(trainset,
                                       batch_size=self.train_batchsize,
                                       shuffle=(train_sampler is None),
                                       num_workers=self.train_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       sampler=train_sampler)

        if self.local_rank != -1:
            print(f"global_rank {self.global_rank},"
                  f"world_size {self.world_size},"
                  f"local_rank {self.local_rank},"
                  f"sampler '{self.train_sampler_name}'")

        if self.local_rank in [-1, 0]:
            val_transform = self.init_transform(self.val_transform_name,
                                                **self.val_transform_params)
            valset = self.init_dataset(self.valset_name,
                                       transform=val_transform,
                                       **self.valset_params)
            self.valloader = DataLoaderX(
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
        self.model = self.init_model(self.network_name, **self.network_params)

        #######################################################################
        # Initialize Loss
        #######################################################################
        self.criterion = self.init_loss(self.loss_name, **self.loss_params)

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.opt = self.init_optimizer(self.opt_name, self.model.parameters(),
                                       **self.opt_params)

        #######################################################################
        # Initialize Adversarial Training
        #######################################################################
        attacker = self.init_module(
            self.adv_name,
            model=self.model,
            eps=self.eps,
            num_steps=self.num_steps,
            step_size=self.step_size,
        )

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################

        if self.local_rank != -1:
            self.model, self.opt = amp.initialize(self.model,
                                                  self.opt,
                                                  opt_level='O1')
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.local_rank],
                                                 output_device=self.local_rank,
                                                 find_unused_parameters=True)

        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.self(self.scheduler_name, self.opt,
                                      **self.scheduler_params)

        #######################################################################
        # Start Training
        #######################################################################
        best_mr = 0.
        best_epoch = 1
        best_group_mr = []
        last_mrs = []
        last_head_mrs = []
        last_mid_mrs = []
        last_tail_mrs = []
        self.final_epoch = self.start_epoch + self.total_epochs
        start_time = datetime.now()

        for cur_epoch in range(self.start_epoch, self.final_epoch):
            self.lr_scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            adv_stat, cln_stat, train_loss = self.train_epoch(
                cur_epoch,
                self.trainloader,
                self.model,
                self.criterion,
                self.opt,
                self.lr_scheduler,
                attacker,
                num_classes=trainset.num_classes,
                clean_weight=self.clean_weight)

            if self.local_rank in [-1, 0]:
                val_stat, val_loss = self.evaluate(
                    cur_epoch,
                    self.valloader,
                    self.model,
                    self.criterion,
                    num_classes=trainset.num_classes)

                if self.final_epoch - cur_epoch <= 10:
                    last_mrs.append(val_stat.mr)
                    last_head_mrs.append(val_stat.group_mr[0])
                    last_mid_mrs.append(val_stat.group_mr[1])
                    last_tail_mrs.append(val_stat.group_mr[2])

                self.log(
                    f"Epoch[{cur_epoch:>3d}/{self.final_epoch-1}] "
                    f"Train Loss={train_loss['final']:>4.2f}"
                    f" | Adv loss:{train_loss['adv']:>4.2f} "
                    f"mr:{adv_stat.mr:>6.2%} "
                    f"[{adv_stat.group_mr[0]:>3.0%},"
                    f"{adv_stat.group_mr[1]:>3.0%},"
                    f"{adv_stat.group_mr[2]:>3.0%}]"
                    f" | Cln loss:{train_loss['cln']:>4.2f} "
                    f"mr:{cln_stat.mr:>6.2%} "
                    f"[{cln_stat.group_mr[0]:>3.0%},"
                    f"{cln_stat.group_mr[1]:>3.0%},"
                    f"{cln_stat.group_mr[2]:>3.0%}]"
                    f"|| Val loss={val_loss:>4.2f} "
                    f"mr={val_stat.mr:>6.2%} "
                    f"[{val_stat.group_mr[0]:>3.0%},"
                    f"{val_stat.group_mr[1]:>3.0%},"
                    f"{val_stat.group_mr[2]:>3.0%}]",
                    log_level='file')

                # if len(val_recalls) <= 20 and cur_epoch == self.total_epochs:
                #     self.logger.info(f"Class recalls: {val_recalls}\n")

                # Save log by tensorboard
                self.writer.add_scalar(f'{self.exp_name}/LearningRate',
                                       self.opt.param_groups[0]['lr'],
                                       cur_epoch)
                self.writer.add_scalars(
                    f'{self.exp_name}/Loss', {
                        'train_loss': train_loss['final'],
                        'adv_loss': train_loss['adv'],
                        'clean_loss': train_loss['cln'],
                        'val_loss': val_loss
                    }, cur_epoch)
                self.writer.add_scalars(
                    f'{self.exp_name}/Recall', {
                        'train_adv_mr': adv_stat.mr,
                        'train_clean_mr': cln_stat.mr,
                        'val_mr': val_stat.mr
                    }, cur_epoch)
                self.writer.add_scalars(
                    f"{self.exp_name}/ADVGroupRecall", {
                        "head_mr": adv_stat.group_mr[0],
                        "mid_mr": adv_stat.group_mr[1],
                        "tail_mr": adv_stat.group_mr[2]
                    }, cur_epoch)
                self.writer.add_scalars(
                    f"{self.exp_name}/CLNGroupRecall", {
                        "head_mr": cln_stat.group_mr[0],
                        "mid_mr": cln_stat.group_mr[1],
                        "tail_mr": cln_stat.group_mr[2]
                    }, cur_epoch)
                self.writer.add_scalars(
                    f"{self.exp_name}/ValGroupRecall", {
                        "head_mr": val_stat.group_mr[0],
                        "mid_mr": val_stat.group_mr[1],
                        "tail_mr": val_stat.group_mr[2]
                    }, cur_epoch)
                is_best = val_stat.mr > best_mr

                if is_best:
                    best_mr = val_stat.mr
                    best_epoch = cur_epoch
                    best_group_mr = val_stat.group_mr

                if (not cur_epoch % self.save_period) or is_best:
                    self.save_checkpoint(epoch=cur_epoch,
                                         model=self.model,
                                         optimizer=self.opt,
                                         is_best=is_best,
                                         mr=val_stat.mr,
                                         group_mr=val_stat.group_mr,
                                         prefix=None,
                                         save_dir=self.exp_dir)

        end_time = datetime.now()
        dur_time = str(end_time - start_time)[:-7]  # 取到秒

        final_mr = np.around(np.mean(last_mrs), decimals=4)
        final_head_mr = np.around(np.mean(last_head_mrs), decimals=4)
        final_mid_mr = np.around(np.mean(last_mid_mrs), decimals=4)
        final_tail_mr = np.around(np.mean(last_tail_mrs), decimals=4)

        if self.local_rank in [-1, 0]:
            self.log(
                f"\n===> Total Runtime: {dur_time}\n\n"
                f"===> Best mean recall: {best_mr:>6.2%} (epoch{best_epoch})\n"
                f"Group recalls: {best_group_mr}\n\n"
                f"===> Final average mean recall of last 10 epochs:"
                f" {final_mr:>6.2%}\n"
                f"Average Group mean recalls: [{final_head_mr:>6.2%}, "
                f"{final_mid_mr:>6.2%}, {final_tail_mr:>6.2%}]\n\n"
                f"===> Save directory: '{self.exp_dir}'\n"
                f"*********************************************************"
                f"*********************************************************")

    def train_epoch(self,
                    cur_epoch,
                    trainloader,
                    model,
                    criterion,
                    optimizer,
                    lr_scheduler,
                    attacker,
                    clean_weight,
                    num_classes=None):
        model.train()
        train_pbar = tqdm(total=len(trainloader),
                          desc='Train Epoch[{:>3d}/{}]'.format(
                              cur_epoch, self.final_epoch - 1))

        final_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        cln_loss_meter = AverageMeter()
        adv_stat = ExpStat(num_classes)
        cln_stat = ExpStat(num_classes)

        for i, (batch_imgs, batch_labels) in enumerate(trainloader):
            optimizer.zero_grad()
            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)

            # Adversarial Training
            # Step 1: generate perturbed samples
            batch_adv_imgs = attacker.attack(batch_imgs, batch_labels)

            # Step 2: train with perturbed imgs
            # Joint clean and adversarial training, 并行加速运算
            batch_mix_imgs = torch.cat((batch_imgs, batch_adv_imgs), 0)
            model.apply(switch_mix)
            batch_mix_probs = model(batch_mix_imgs)
            # 将batch_mix_probs沿着0维，等分切为两份, 分别计算loss
            batch_cln_probs, batch_adv_probs = batch_mix_probs.chunk(2, 0)
            batch_cln_loss = criterion(batch_cln_probs, batch_labels)
            batch_adv_loss = criterion(batch_adv_probs, batch_labels)
            batch_final_loss = clean_weight * batch_cln_loss +\
                (1 - self.clean_weight) * batch_adv_loss

            if self.local_rank != -1:
                with amp.scale_loss(batch_final_loss, optimizer)\
                        as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()
                self._reduce_tensor(batch_final_loss)
            else:
                batch_final_loss.backward()
                optimizer.step()

            final_loss_meter.update(batch_final_loss.item(), 1)
            cln_loss_meter.update(batch_cln_loss.item(), 1)
            adv_loss_meter.update(batch_adv_loss.item(), 1)

            batch_cln_preds = batch_cln_probs.max(1)[1]
            batch_adv_preds = batch_adv_probs.max(1)[1]

            adv_stat.update(batch_labels, batch_adv_preds)
            cln_stat.update(batch_labels, batch_cln_preds)

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str(
                    "LR:{:.1e} Loss:{:>4.2f} ADV:{:>4.2f} CLN:{:>4.2f}".format(
                        optimizer.param_groups[-1]['lr'],
                        final_loss_meter.avg,
                        adv_loss_meter.avg,
                        cln_loss_meter.avg,
                    ))

        if self.local_rank in [-1, 0]:
            train_loss = {
                'final': final_loss_meter.avg,
                'adv': adv_loss_meter.avg,
                'cln': cln_loss_meter.avg
            }
            train_pbar.set_postfix_str(
                f"LR:{optimizer.param_groups[0]['lr']:.1e} "
                f"Loss:{final_loss_meter.avg:.1f} "
                f" | Adv loss={train_loss['adv']:.1f} "
                f"mr={adv_stat.mr:>6.2%} "
                f"[{adv_stat.group_mr[0]:>3.0%},"
                f"{adv_stat.group_mr[1]:>3.0%},"
                f"{adv_stat.group_mr[2]:>3.0%}]"
                f" | Cln loss={train_loss['cln']:.1f} "
                f"mr={cln_stat.mr:>6.2%} "
                f"[{cln_stat.group_mr[0]:>3.0%},"
                f"{cln_stat.group_mr[1]:>3.0%},"
                f"{cln_stat.group_mr[2]:>3.0%}]")

        return adv_stat, cln_stat, train_loss

    def evaluate(self,
                 cur_epoch,
                 valloader,
                 model,
                 criterion,
                 num_classes=None):
        model.eval()
        model.apply(switch_clean)
        val_pbar = tqdm(total=len(valloader),
                        ncols=0,
                        desc='                 Val')

        val_loss_meter = AverageMeter()
        val_stat = ExpStat(num_classes)
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(valloader):
                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
                batch_probs = model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]
                avg_loss = criterion(batch_probs, batch_labels)

                val_loss_meter.update(avg_loss.item(), 1)
                val_stat.update(batch_labels, batch_preds)

                val_pbar.update()

        val_pbar.set_postfix_str(f"loss:{val_loss_meter.avg:.1f} "
                                 f"mr:{val_stat.mr:>6.2%} "
                                 f"[{val_stat.group_mr[0]:>3.0%},"
                                 f"{val_stat.group_mr[1]:>3.0%},"
                                 f"{val_stat.group_mr[2]:>3.0%}]")
        val_pbar.close()

        return val_stat, val_loss_meter.avg


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


def main(args):
    warnings.filterwarnings('ignore')
    _set_seed()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
