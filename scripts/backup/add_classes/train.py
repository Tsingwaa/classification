"""trainer script """
import argparse
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from apex import amp
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
# from pudb import set_trace
# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter, ExpStat


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Trainer(BaseTrainer):

    def __init__(self, args, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)
        self.args = args
        self.head_class_idx = config['head_class_idx']
        self.med_class_idx = config['med_class_idx']
        self.tail_class_idx = config['tail_class_idx']

    def train(self):

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

        val_transform = self.init_transform(self.val_transform_name,
                                            **self.val_transform_params)
        valset = self.init_dataset(self.valset_name,
                                   transform=val_transform,
                                   **self.valset_params)

        val_sampler = self.init_sampler(self.val_sampler_name,
                                        dataset=valset,
                                        **self.valloader_params)
        self.valloader = DataLoaderX(
            valset,
            batch_size=self.val_batchsize,
            shuffle=(val_sampler is None),
            num_workers=self.val_workers,
            pin_memory=True,
            drop_last=False,
            sampler=val_sampler,
        )
        val_sampler_train = self.init_sampler(self.val_sampler_name,
                                              dataset=trainset,
                                              **self.trainloader_params)
        self.valloader_train = DataLoaderX(
            trainset,
            batch_size=self.val_batchsize,
            shuffle=(val_sampler_train is None),
            num_workers=self.val_workers,
            pin_memory=True,
            drop_last=False,
            sampler=val_sampler_train,
        )

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     num_classes=trainset.num_classes,
                                     **self.network_params)

        #######################################################################
        # Initialize Loss
        #######################################################################
        weight = self.get_class_weight(
            num_samples_per_cls=trainset.num_samples_per_cls,
            **self.loss_params,  # ??????weight_type
        )
        self.criterion = self.init_loss(self.loss_name,
                                        weight=weight,
                                        **self.loss_params)

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.opt = self.init_optimizer(self.opt_name, self.model.parameters(),
                                       **self.opt_params)

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################

        if self.local_rank != -1:
            self.model = convert_syncbn_model(self.model).cuda(self.local_rank)
            self.model, self.opt = amp.initialize(self.model,
                                                  self.opt,
                                                  opt_level="O1")
            self.model = DistributedDataParallel(self.model)

        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.scheduler = self.init_lr_scheduler(self.scheduler_name, self.opt,
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

        if self.local_rank in [-1, 0]:
            start_time = datetime.now()

        for cur_epoch in range(self.start_epoch, self.final_epoch):
            self.scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            _, train_loss = self.train_epoch(
                cur_epoch,
                self.trainloader,
                self.model,
                self.criterion,
                self.opt,
                trainset.num_classes,
            )
            train_stat, train_loss = self.evaluate(
                cur_epoch,
                self.valloader_train,
                self.model,
                self.criterion,
                trainset.num_classes,
            )
            val_stat, val_loss = self.evaluate(cur_epoch, self.valloader,
                                               self.model, self.criterion,
                                               trainset.num_classes)

            if self.local_rank in [-1, 0]:

                if self.final_epoch - cur_epoch <= 5:
                    last_mrs.append(val_stat.mr)
                    last_head_mrs.append(val_stat.group_mr[0])
                    last_mid_mrs.append(val_stat.group_mr[1])
                    last_tail_mrs.append(val_stat.group_mr[2])

                self.log(
                    f"Epoch[{cur_epoch:>3d}/{self.final_epoch-1}] "
                    f"Trainset Loss={train_loss:>4.2f} "
                    f"MR={train_stat.mr:>6.2%} "
                    f"[{train_stat.group_mr[0]:>6.2%}, "
                    f"{train_stat.group_mr[1]:>6.2%}, "
                    f"{train_stat.group_mr[2]:>6.2%}"
                    f" || Valset Loss={val_loss:>4.2f} "
                    f"MR={val_stat.mr:>6.2%} "
                    f"[{val_stat.group_mr[0]:>6.2%}, "
                    f"{val_stat.group_mr[1]:>6.2%}, "
                    f"{val_stat.group_mr[2]:>6.2%}",
                    log_level='file')

                # Save log by tensorboard
                self.writer.add_scalar(f"{self.exp_name}/LR",
                                       self.opt.param_groups[-1]["lr"],
                                       cur_epoch)
                self.writer.add_scalars(f"{self.exp_name}/Loss", {
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, cur_epoch)
                self.writer.add_scalars(f"{self.exp_name}/Recall", {
                    "train_mr": train_stat.mr,
                    "val_mr": val_stat.mr
                }, cur_epoch)
                self.writer.add_scalars(
                    f"{self.exp_name}/TrainGroupRecall", {
                        "head_mr": train_stat.group_mr[0],
                        "mid_mr": train_stat.group_mr[1],
                        "tail_mr": train_stat.group_mr[2]
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
                    self.save_checkpoint(
                        epoch=cur_epoch,
                        model=self.model,
                        optimizer=self.opt,
                        is_best=is_best,
                        mr=val_stat.mr,
                        group_mr=val_stat.group_mr,
                        prefix='seed_%d' % (self.args.seed),
                        save_dir=self.exp_dir,
                        criterion=self.criterion,
                    )

        if self.local_rank in [-1, 0]:
            end_time = datetime.now()
            dur_time = str(end_time - start_time)[:-7]  # ?????????

            final_mr = np.around(np.mean(last_mrs), decimals=4)
            final_head_mr = np.around(np.mean(last_head_mrs), decimals=4)
            final_mid_mr = np.around(np.mean(last_mid_mrs), decimals=4)
            final_tail_mr = np.around(np.mean(last_tail_mrs), decimals=4)

            self.log(
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

    def train_epoch(self, cur_epoch, trainloader, model, criterion, opt,
                    num_classes, **kwargs):
        model.train()

        if self.local_rank in [-1, 0]:
            train_pbar = tqdm(
                total=len(trainloader),
                desc=f"Train Epoch[{cur_epoch:>3d}/{self.final_epoch-1}]")

        train_loss_meter = AverageMeter()
        train_stat = ExpStat(num_classes, self.head_class_idx,
                             self.med_class_idx, self.tail_class_idx)

        for i, (batch_imgs, batch_labels) in enumerate(trainloader):
            opt.zero_grad()

            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)
            batch_probs = model(batch_imgs, out_type='fc')
            avg_loss = criterion(batch_probs, batch_labels)

            if self.local_rank != -1:
                torch.distributed.barrier()
                # torch.distributed.barrier()?????????????????????????????????????????????????????????
                # ????????????????????????????????????????????????????????????loss?????????acc?????????
                # ???????????????????????????????????????????????????????????????
                with amp.scale_loss(avg_loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                avg_loss = self._reduce_tensor(avg_loss)
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

        if self.local_rank != -1:
            # all reduce the statistical confusion matrix
            torch.distributed.barrier()
            train_stat._cm = self._reduce_tensor(train_stat._cm, op='sum')

        if self.local_rank in [-1, 0]:
            train_pbar.set_postfix_str(f"LR:{opt.param_groups[0]['lr']:.1e} "
                                       f"Loss:{train_loss_meter.avg:>4.2f} "
                                       f"MR:{train_stat.mr:>6.2%} "
                                       f"[{train_stat.group_mr[0]:>3.0%}, "
                                       f"{train_stat.group_mr[1]:>3.0%}, "
                                       f"{train_stat.group_mr[2]:>3.0%}]")

            train_pbar.close()

        return train_stat, train_loss_meter.avg

    def evaluate(self, cur_epoch, valloader, model, criterion, num_classes):
        model.eval()

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

                batch_probs = model(batch_imgs, out_type='fc')
                batch_preds = torch.argmax(batch_probs, dim=1)
                avg_loss = criterion(batch_probs, batch_labels)

                val_loss_meter.update(avg_loss.item(), 1)
                val_stat.update(batch_labels, batch_preds)

                if self.local_rank in [-1, 0]:
                    val_pbar.update()

        if self.local_rank != -1:
            # all reduce the statistical confusion matrix
            torch.distributed.barrier()
            val_stat._cm = self._reduce_tensor(val_stat._cm, op='sum')

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
    parser.add_argument("--seed", type=int, help="rand_seed")
    args = parser.parse_args()

    return args


def _set_random_seed(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, but more reproducible
        torch.backends.cudnn.deterministic = True  # ?????????????????????
        torch.backends.cudnn.benchmark = False  # ?????????????????????
    else:
        torch.backends.cudnn.deterministic = False  # ????????????????????????
        torch.backends.cudnn.benchmark = True  # ?????????????????????????????????


def main(args):
    warnings.filterwarnings("ignore")
    _set_random_seed(seed=args.seed)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(args=args, local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
