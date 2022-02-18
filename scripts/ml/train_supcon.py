"""trainer script """
import argparse
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
# from pudb import set_trace
from torch import distributed as dist
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Trainer(BaseTrainer):

    def __init__(self, local_rank, config, seed):
        super(Trainer, self).__init__(local_rank, config, seed)

    def train(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        train_transform = self.init_transform(self.train_transform_name,
                                              **self.train_transform_params)
        train_simsiam_transform = self.init_transform(
            "SiameseTransform", base_transform=train_transform)
        trainset = self.init_dataset(self.trainset_name,
                                     transform=train_simsiam_transform,
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
                                       sampler=train_sampler,
                                       persistent_workers=True)

        if self.local_rank != -1:
            dist.barrier()

            if not self.train_sampler_name:
                self.train_sampler_name = "DistributedSampler"
            self.log(f"world_size={self.world_size}, "
                     f"local_rank={self.local_rank}, "
                     f"train_sampler='{self.train_sampler_name}'")

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     num_classes=trainset.num_classes,
                                     **self.network_params)

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################
        if self.local_rank != -1:
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        #######################################################################
        # Initialize Loss
        #######################################################################
        weight = self.get_class_weight(trainset.num_samples_per_cls,
                                       **self.loss_params)  # 包含weight_type
        self.criterion = self.init_loss(self.loss_name,
                                        weight=weight,
                                        **self.loss_params)

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.optimizer = self.init_optimizer(self.opt_name,
                                             self.model.parameters(),
                                             **self.opt_params)

        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.init_lr_scheduler(self.scheduler_name,
                                                   self.optimizer,
                                                   **self.scheduler_params)

        #######################################################################
        # Start Training
        #######################################################################
        if self.local_rank in [-1, 0]:
            start_time = datetime.now()

        self.final_epoch = self.start_epoch + self.total_epochs
        for cur_epoch in range(self.start_epoch, self.final_epoch):
            self.lr_scheduler.step()

            if self.local_rank != -1:
                dist.barrier()
                # barrier的作用是，阻塞进程
                # 确保每个进程都运行到这一行代码，才能继续执行，这样计算
                # 平均loss和平均acc的时候，不会出现因为进程执行速度不一致
                # 而导致错误
                train_sampler.set_epoch(cur_epoch)

            train_loss = self.train_epoch(
                cur_epoch=cur_epoch,
                trainloader=self.trainloader,
                model=self.model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                dataset=trainset,
            )

            if self.local_rank in [-1, 0]:
                self.log(
                    f"Epoch[{cur_epoch:>3d}/{self.final_epoch-1}] "
                    f"LR:{self.optimizer.param_groups[0]['lr']:.1e} "
                    f"Trainset Loss={train_loss:>4.2f} ",
                    log_level="file")

                if not cur_epoch % self.save_period:
                    self.save_checkpoint(epoch=cur_epoch,
                                         model=self.model,
                                         optimizer=self.optimizer,
                                         is_best=False,
                                         stat=None,
                                         prefix=f"seed{self.seed}",
                                         save_dir=self.exp_dir)

        if self.local_rank in [-1, 0]:
            end_time = datetime.now()
            dur_time = str(end_time - start_time)[:-7]  # 取到秒

            self.log(
                f"\n===> Total Runtime: {dur_time}\n"
                f"===> Save directory: '{self.exp_dir}'\n"
                f"*********************************************************"
                f"*********************************************************\n")

    def train_epoch(self, cur_epoch, trainloader, model, criterion, optimizer,
                    dataset, **kwargs):
        model.train()

        if self.local_rank in [-1, 0]:
            train_pbar = tqdm(
                total=len(trainloader),
                ncols=0,
                desc=f"Train Epoch[{cur_epoch:>3d}/{self.final_epoch-1}]")

        train_loss_meter = AverageMeter()

        for i, (batch_imgs, batch_targets) in enumerate(trainloader):
            batch_imgs = torch.cat([batch_imgs[0], batch_imgs[1]], dim=0)
            batch_imgs = batch_imgs.cuda(non_blocking=True)  # 2B imgs
            batch_targets = batch_targets.cuda(non_blocking=True)

            batch_feats = model(batch_imgs, out_type="pred_head")  # 2B * d
            batch_feats1, batch_feats2 = torch.chunk(batch_feats, 2, dim=0)
            batch_feats = torch.cat(
                [batch_feats1.unsqueeze(1),
                 batch_feats2.unsqueeze(1)], dim=1)  # B * 2(views) * d
            avg_loss = criterion(batch_feats, batch_targets)

            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            if self.local_rank != -1:
                dist.barrier()
                avg_loss = self._reduce_tensor(avg_loss)

            train_loss_meter.update(avg_loss.item(), 1)

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str(
                    f"LR:{optimizer.param_groups[0]['lr']:.1e} "
                    f"Loss:{train_loss_meter.avg:>5.3f}")

        if self.local_rank in [-1, 0]:
            train_pbar.set_postfix_str(
                f"LR:{optimizer.param_groups[0]['lr']:.1e} "
                f"Loss:{train_loss_meter.avg:>5.3f}")
            train_pbar.close()

        return train_loss_meter.avg

    # def evaluate(self, cur_epoch, valloader, model, criterion, dataset,
    #              **kwargs):
    #     model.eval()

    #     if self.local_rank in [-1, 0]:
    #         desc = kwargs.pop("desc", "Val")
    #         val_pbar = tqdm(total=len(valloader),
    #                         ncols=0,
    #                         desc=f"                 {desc}")
    #     val_loss_meter = AverageMeter()

    #     with torch.no_grad():
    #         for i, (batch_imgs, batch_targets) in enumerate(valloader):
    #             batch_imgs = batch_imgs.cuda(non_blocking=True)
    #             batch_targets = batch_targets.cuda(non_blocking=True)
    #             batch_embeddings = model(batch_imgs, out_type='vec')
    #             avg_loss = criterion(batch_embeddings, batch_targets)

    #             if self.local_rank != -1:
    #                 dist.barrier()
    #                 avg_loss = self._reduce_tensor(avg_loss)

    #             val_loss_meter.update(avg_loss.item(), 1)

    #             if self.local_rank in [-1, 0]:
    #                 val_pbar.update()
    #                 val_pbar.set_postfix_str(
    #                     f"Loss:{val_loss_meter.avg:>3.1f}")

    #     if self.local_rank in [-1, 0]:
    #         val_pbar.set_postfix_str(f"Loss:{val_loss_meter.avg:>4.2f}")
    #         val_pbar.close()

    #     return val_loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    # Local rank设定：单卡默认为-1，多卡不设定，ddp自动设定为0,1,...
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local Rank for distributed training. "
                        "if single-GPU, default set to -1")
    parser.add_argument("--config_path", type=str, help="path of config file")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--out_type", type=str, default="vec_norm")
    # parser.add_argument("--lr", default=0.5, type=float, help="learning rate")
    # parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
    args = parser.parse_args()

    return args


def _set_random_seed(seed=0, cuda_deterministic=False):
    """Set seed and control the balance between reproducity and efficiency
    Reproducity: cuda_deterministic = True
    Efficiency: cuda_deterministic = False
    """

    random.seed(seed)
    np.random.seed(seed)

    assert torch.cuda.is_available()
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, but more reproducible
        torch.backends.cudnn.deterministic = True  # 固定内部随机性
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # 输入尺寸一致，加速训练


def main(args):
    warnings.filterwarnings("ignore")
    _set_random_seed(seed=args.seed)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # compare lr and wd
    # config["experiment"]["name"] += f"_lr{args.lr}_wd{args.wd}"
    # config["optimizer"]["param"].update({
    #     "lr": float(args.lr),
    #     "weight_decay": float(args.wd),
    # })
    # vec norm
    # config["experiment"]["name"] += f"_{args.out_type}"

    trainer = Trainer(
        local_rank=args.local_rank,
        config=config,
        seed=args.seed,
        # out_type=args.out_type,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    main(args)
