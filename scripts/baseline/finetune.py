"""finetune script """
import argparse
import os
import random
import warnings
from datetime import datetime
from os.path import join

import numpy as np
import torch
import yaml
from apex import amp
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
# from pudb import set_trace
# from sklearn import metrics
from torch import distributed as dist
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, ExpStat


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class FineTuner(BaseTrainer):

    def __init__(self, args, local_rank=None, config=None):

        #######################################################################
        # Device setting
        #######################################################################
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True

        self.local_rank = local_rank

        if self.local_rank != -1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        #######################################################################
        # Experiment setting
        #######################################################################

        self.head_class_idx = config['head_class_idx']
        self.med_class_idx = config['med_class_idx']
        self.tail_class_idx = config['tail_class_idx']

        self.exp_config = config["experiment"]
        self.exp_name = self.exp_config["name"]
        self.finetune_config = config["finetune"]
        self.finetune_name = self.finetune_config["name"]

        self.user_root = os.environ["HOME"]
        self.exp_root = join(self.user_root, "Projects/Experiments")
        self.total_epochs = self.finetune_config["total_epochs"]

        self._set_configs(config)

        self.resume = True

        if "/" in self.exp_config["resume_fpath"]:
            self.resume_fpath = self.exp_config["resume_fpath"]
        else:
            # self.resume_fpath = join(self.exp_root, self.exp_name,
            #                          self.exp_config["resume_fpath"])
            self.resume_fpath = join(
                self.exp_root, self.exp_name,
                'seed_%d_%s' % (args.seed, self.exp_config["resume_fpath"]))
        self.checkpoint, resume_log = self.resume_checkpoint(self.resume_fpath)

        self.start_epoch = 1
        self.final_epoch = self.start_epoch + self.total_epochs

        if self.local_rank in [-1, 0]:
            self.eval_period = self.exp_config["eval_period"]
            self.save_period = self.exp_config["save_period"]
            self.exp_dir = join(self.exp_root, self.exp_name)
            os.makedirs(self.exp_dir, exist_ok=True)

            # Set logger to save .log file and output to screen.
            self.log_fpath = join(self.exp_dir, f"{self.finetune_name}.log")
            self.logger = self.init_logger(self.log_fpath)
            exp_init_log = f"\n****************************************"\
                f"****************************************************"\
                f"\nExperiment: Finetune {self.exp_name}\n"\
                f"Total_epochs: {self.total_epochs}\n"\
                f"Save dir: {self.exp_dir}\n"\
                f"Save peroid: {self.save_period}\n"\
                f"Resume Training: {self.resume}\n"\
                f"Distributed Training: "\
                f"{True if self.local_rank != -1 else False}\n"\
                f"**********************************************"\
                f"**********************************************\n"
            self.log(exp_init_log)
            self.log(resume_log)

        self.unfreeze_keys = self.finetune_config["unfreeze_keys"]

        ft_network_config = self.finetune_config.pop("network", None)

        if ft_network_config is not None:
            self.ft_network_name = ft_network_config["name"]
            self.ft_network_params = ft_network_config["param"]

        self.trainloader_params = self.finetune_config["trainloader"]

        self.train_sampler_name = self.trainloader_params.pop("sampler", None)
        self.train_batchsize = self.trainloader_params["batch_size"]
        self.train_workers = self.trainloader_params["num_workers"]

        ft_loss_config = self.finetune_config["loss"]
        self.loss_name = ft_loss_config["name"]
        self.loss_params = ft_loss_config["param"]

        ft_opt_config = self.finetune_config["optimizer"]
        self.opt_name = ft_opt_config["name"]
        self.opt_params = ft_opt_config["param"]

        ft_lrS_config = self.finetune_config["lr_scheduler"]
        self.scheduler_name = ft_lrS_config["name"]
        self.scheduler_params = ft_lrS_config["param"]

    def finetune(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        train_transform = self.init_transform(self.train_transform_name,
                                              **self.train_transform_params)
        trainset = self.init_dataset(self.trainset_name,
                                     transform=train_transform,
                                     **self.trainset_params)
        train_sampler = self.init_sampler(self.train_sampler_name,
                                          dataset=trainset,
                                          **self.trainloader_params)
        self.trainloader = DataLoaderX(
            dataset=trainset,
            batch_size=self.train_batchsize,
            shuffle=(train_sampler is None),
            num_workers=self.train_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )

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
                dataset=valset,
                batch_size=self.val_batchsize,
                shuffle=False,
                num_workers=self.val_workers,
                pin_memory=True,
                drop_last=False,
            )

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     resume=True,
                                     checkpoint=self.checkpoint,
                                     num_classes=trainset.num_classes,
                                     **self.network_params)
        self.freeze_model(self.model, unfreeze_keys=self.unfreeze_keys)

        #######################################################################
        # Initialize Loss
        #######################################################################
        weight = self.get_class_weight(trainset.num_samples_per_cls,
                                       **self.loss_params)
        self.criterion = self.init_loss(self.loss_name,
                                        weight=weight,
                                        **self.loss_params)

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.opt = self.init_optimizer(self.opt_name, self.model.parameters(),
                                       **self.opt_params)

        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.init_lr_scheduler(self.scheduler_name,
                                                   self.opt,
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
        start_time = datetime.now()

        for cur_epoch in range(self.start_epoch, self.final_epoch):
            # learning rate decay by epoch
            self.lr_scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            train_stat, train_loss = self.train_epoch(
                cur_epoch=cur_epoch,
                trainloader=self.trainloader,
                model=self.model,
                criterion=self.criterion,
                opt=self.opt,
                lr_scheduler=self.lr_scheduler,
                num_classes=trainset.num_classes,
            )

            if self.local_rank in [-1, 0]:
                val_stat, val_loss = self.evaluate(
                    cur_epoch=cur_epoch,
                    valloader=self.valloader,
                    model=self.model,
                    criterion=self.criterion,
                    num_classes=trainset.num_classes)

                if self.final_epoch - cur_epoch <= 5:
                    last_mrs.append(val_stat.mr)
                    last_head_mrs.append(val_stat.group_mr[0])
                    last_mid_mrs.append(val_stat.group_mr[1])
                    last_tail_mrs.append(val_stat.group_mr[2])

                self.log(
                    f"Epoch[{cur_epoch:>3d}/{self.final_epoch-1}] "
                    f"Train Loss={train_loss:>4.2f}"
                    f" MR={train_stat.mr:>6.2%} "
                    f"[{train_stat.group_mr[0]:>6.2%}, "
                    f"{train_stat.group_mr[1]:>6.2%}, "
                    f"{train_stat.group_mr[2]:>6.2%}"
                    f" || "
                    f"Val Loss={val_loss:>4.2f} "
                    f"MR={val_stat.mr:>6.2%} "
                    f"[{val_stat.group_mr[0]:>6.2%}, "
                    f"{val_stat.group_mr[1]:>6.2%}, "
                    f"{val_stat.group_mr[2]:>6.2%}",
                    log_level="file")

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
                                         prefix=self.finetune_name,
                                         save_dir=self.exp_dir)

        end_time = datetime.now()
        dur_time = str(end_time - start_time)[:-7]  # 取到秒

        final_mr = np.around(np.mean(last_mrs), decimals=4)
        final_head_mr = np.around(np.mean(last_head_mrs), decimals=4)
        final_mid_mr = np.around(np.mean(last_mid_mrs), decimals=4)
        final_tail_mr = np.around(np.mean(last_tail_mrs), decimals=4)

        if self.local_rank in [-1, 0]:
            self.logger.info(
                f"\n===> Total Runtime: {dur_time}\n\n"
                f"===> Best mean recall: {best_mr:>6.2%} (epoch{best_epoch})\n"
                f"Group recalls: [{best_group_mr[0]:>6.2%}, "
                f"{best_group_mr[1]:>6.2%}, {best_group_mr[2]:>6.2%}]\n\n"
                f"===> Final average mean recall of last 10 epochs:"
                f" {final_mr:>6.2%}\n"
                f"Average Group mean recalls: [{final_head_mr:6.2%}, "
                f"{final_mid_mr:>6.2%}, {final_tail_mr:>6.2%}]\n\n"
                f"===> Save directory: '{self.exp_dir}'\n"
                f"*********************************************************"
                f"*********************************************************\n")

    def train_epoch(self,
                    cur_epoch,
                    trainloader,
                    model,
                    criterion,
                    opt,
                    lr_scheduler,
                    ft_model=None,
                    num_classes=None):

        model.train()

        if ft_model is not None:
            ft_model.train()

        if self.local_rank in [-1, 0]:
            train_pbar = tqdm(
                total=len(trainloader),
                desc=f"Train Epoch[{cur_epoch:>2d}/{self.final_epoch-1}]")

        train_loss_meter = AverageMeter()
        train_stat = ExpStat(num_classes, self.head_class_idx,
                             self.med_class_idx, self.tail_class_idx)

        for i, (batch_imgs, batch_labels) in enumerate(trainloader):
            opt.zero_grad()

            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)

            if ft_model is not None:
                batch_feats = model(batch_imgs, out_type='feat')
                batch_probs = ft_model(batch_feats)
            else:
                batch_probs = model(batch_imgs)

            avg_loss = criterion(batch_probs, batch_labels)

            if self.local_rank != -1:
                with amp.scale_loss(avg_loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                self._reduce_tensor(avg_loss)
            else:
                avg_loss.backward()
                opt.step()

            batch_preds = torch.argmax(batch_probs, dim=1)
            train_loss_meter.update(avg_loss.item(), 1)
            train_stat.update(batch_labels, batch_preds)

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str(
                    f"LR:{opt.param_groups[0]['lr']:.1e} "
                    f"Loss:{train_loss_meter.avg:>4.2f}")

        if self.local_rank in [-1, 0]:
            train_pbar.set_postfix_str(f"LR:{opt.param_groups[0]['lr']:.1e} "
                                       f"Loss:{train_loss_meter.avg:>4.2f} "
                                       f"MR:{train_stat.mr:>6.2%} "
                                       f"[{train_stat.group_mr[0]:>3.0%}, "
                                       f"{train_stat.group_mr[1]:>3.0%}, "
                                       f"{train_stat.group_mr[2]:>3.0%}]")

            train_pbar.close()

        return train_stat, train_loss_meter.avg

    def evaluate(self,
                 cur_epoch,
                 valloader,
                 model,
                 criterion,
                 ft_model=None,
                 num_classes=None):

        model.eval()

        if ft_model is not None:
            ft_model.eval()

        if self.local_rank in [-1, 0]:
            val_pbar = tqdm(total=len(valloader),
                            ncols=0,
                            desc="                 Val")

        val_loss_meter = AverageMeter()
        val_stat = ExpStat(num_classes, self.head_class_idx,
                           self.med_class_idx, self.tail_class_idx)
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(valloader):
                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)

                if ft_model is not None:
                    batch_feats = model(batch_imgs, out_type='feat')
                    batch_probs = ft_model(batch_feats)
                else:
                    batch_probs = model(batch_imgs)

                avg_loss = criterion(batch_probs, batch_labels)
                val_loss_meter.update(avg_loss.item(), 1)

                batch_preds = torch.argmax(batch_probs, dim=1)
                val_stat.update(batch_labels, batch_preds)

                val_pbar.update()

        if self.local_rank in [-1, 0]:
            val_pbar.set_postfix_str(f"Loss:{val_loss_meter.avg:>4.2f} "
                                     f"MR:{val_stat.mr:>6.2%} "
                                     f"[{val_stat.group_mr[0]:>3.0%}, "
                                     f"{val_stat.group_mr[1]:>3.0%}, "
                                     f"{val_stat.group_mr[2]:>3.0%}]")
            val_pbar.close()

        return val_stat, val_loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        help="Local Rank for\
                        distributed training. if single-GPU, default: -1")
    parser.add_argument("--config_path", type=str, help="path of config file")
    parser.add_argument("--seed", type=int, default=0)
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
    warnings.filterwarnings("ignore")
    _set_seed()
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    finetuner = FineTuner(args, local_rank=args.local_rank, config=config)
    finetuner.finetune()


if __name__ == "__main__":
    args = parse_args()
    main(args)
