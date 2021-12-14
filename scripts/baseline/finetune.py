"""finetune script """
import os
import math
import random
import warnings
import argparse
import yaml
import numpy as np
import torch
# from pudb import set_trace
from os.path import join
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn import metrics
from prefetch_generator import BackgroundGenerator
# Distribute Package
from torch import distributed as dist
from apex import amp
# Custom Package
from base.base_trainer import BaseTrainer
from utils import AverageMeter


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class FineTuner(BaseTrainer):
    def __init__(self, local_rank=None, config=None):

        #######################################################################
        # Device setting
        #######################################################################
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True

        self.local_rank = local_rank
        if self.local_rank != -1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        #######################################################################
        # Experiment setting
        #######################################################################
        self.exp_config = config['experiment']
        self.exp_name = self.exp_config['name']
        self.finetune_config = config['finetune']
        self.finetune_name = self.finetune_config['name']

        self.user_root = os.environ['HOME']
        self.exp_root = join(self.user_root, 'Experiments')
        self.total_epochs = self.finetune_config['total_epochs']
        self.resume = True
        if '/' in self.exp_config['resume_fpath']:
            self.resume_fpath = self.exp_config['resume_fpath']
        else:
            self.resume_fpath = join(self.exp_root, self.exp_name,
                                     self.exp_config['resume_fpath'])
        self.checkpoint, resume_log =\
            self.resume_checkpoint(self.resume_fpath)
        self.start_epoch = self.checkpoint['epoch'] + 1
        self.final_epoch = self.start_epoch + self.total_epochs

        if self.local_rank in [-1, 0]:
            self.eval_period = self.exp_config['eval_period']
            self.save_period = self.exp_config['save_period']
            self.save_dir = join(self.exp_root, self.exp_name)
            os.makedirs(self.save_dir, exist_ok=True)

            # self.tb_dir = join(self.exp_root, 'Tensorboard', self.exp_name)
            # os.makedirs(self.tb_dir, exist_ok=True)
            # self.writer = SummaryWriter(log_dir=self.tb_dir)

            # Set logger to save .log file and output to screen.
            self.log_fpath = join(self.save_dir,
                                  f'finetune{self.finetune_name}.log')
            self.logger = self.init_logger(self.log_fpath)
            exp_init_log = f'\n****************************************'\
                f'****************************************************'\
                f'\nExperiment: Finetune {self.exp_name}\n'\
                f'Start_epoch: {self.start_epoch}\n'\
                f'Total_epochs: {self.total_epochs}\n'\
                f'Save dir: {self.save_dir}\n'\
                f'Save peroid: {self.save_period}\n'\
                f'Resume Training: {self.resume}\n'\
                f'Distributed Training: '\
                f'{True if self.local_rank != -1 else False}\n'\
                f'**********************************************'\
                f'**********************************************\n'
            self.log(exp_init_log)
            self.log(resume_log)

        self._set_configs(config)

        self.unfreeze_keys = self.finetune_config['unfreeze_keys']

        self.trainloader_params = self.finetune_config['trainloader']
        self.train_sampler_name = self.trainloader_params['sampler']
        self.train_batchsize = self.trainloader_params['batch_size']
        self.train_workers = self.trainloader_params['num_workers']

        loss_config = self.finetune_config['loss']
        self.loss_name = loss_config['name']
        self.loss_params = loss_config['param']

        opt_config = self.finetune_config['optimizer']
        self.opt_name = opt_config['name']
        self.opt_config = opt_config['param']

        scheduler_config = self.finetune_config['lr_scheduler']
        self.scheduler_name = scheduler_config['name']
        self.scheduler_params = scheduler_config['param']

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
            self.valloader = DataLoaderX(valset,
                                         batch_size=self.val_batchsize,
                                         shuffle=False,
                                         num_workers=self.val_workers,
                                         pin_memory=True,
                                         drop_last=False,)

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     resume=True,
                                     checkpoint=self.checkpoint,
                                     **self.network_params)
        self.freeze_model(self.model, unfreeze_keys=self.unfreeze_keys)

        #######################################################################
        # Initialize Loss
        #######################################################################
        self.loss_params = self.update_class_weight(trainset.img_num,
                                                    **self.loss_params)
        self.criterion = self.init_loss(self.loss_name, **self.loss_params)

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.optimizer = self.init_optimizer(self.opt_name, self.model,
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
        best_mr = 0.
        best_epoch = 1
        best_group_recalls = []
        last_mrs = []
        last_head_mrs = []
        last_mid_mrs = []
        last_tail_mrs = []
        for cur_epoch in range(self.start_epoch, self.final_epoch):
            # learning rate decay by epoch
            self.lr_scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            train_mr, train_group_recalls, train_loss = self.train_epoch(
                cur_epoch,
                self.trainloader,
                self.model,
                self.criterion,
                self.optimizer,
                self.lr_scheduler,
                num_classes=trainset.cls_num)

            if self.local_rank in [-1, 0]:
                val_mr, val_group_recalls, val_loss = self.evaluate(
                    cur_epoch,
                    self.valloader,
                    self.model,
                    self.criterion,
                    num_classes=trainset.cls_num)

                if self.final_epoch - cur_epoch <= 5:
                    last_mrs.append(val_mr)
                    last_head_mrs.append(val_group_recalls[0])
                    last_mid_mrs.append(val_group_recalls[1])
                    last_tail_mrs.append(val_group_recalls[2])

                self.logger.debug(
                    "Epoch[{epoch:>3d}/{final_epoch}] "
                    "Trainset Loss={train_loss:.4f} MR={train_mr:.2%} || "
                    "Valset Loss={val_loss:.4f} MR={val_mr:.2%} "
                    "Head={head:.2%} Mid={mid:.2%} Tail={tail:.2%}".format(
                        epoch=cur_epoch,
                        final_epoch=self.final_epoch - 1,
                        train_loss=train_loss,
                        train_mr=train_mr,
                        val_loss=val_loss,
                        val_mr=val_mr,
                        head=val_group_recalls[0],
                        mid=val_group_recalls[1],
                        tail=val_group_recalls[2],
                    )
                )

                # Save log by tensorboard
                # self.writer.add_scalar(f"{self.exp_name}/LR",
                #                        self.optimizer.param_groups[-1]["lr"],
                #                        cur_epoch)
                # self.writer.add_scalars(f"{self.exp_name}/Loss",
                #                         {"train_loss": train_loss,
                #                          "val_loss": val_loss},
                #                         cur_epoch)
                # self.writer.add_scalars(f"{self.exp_name}/Recall",
                #                         {"train_mr": train_mr,
                #                          "val_mr": val_mr},
                #                         cur_epoch)
                # self.writer.add_scalars(f"{self.exp_name}/GroupRecall",
                #                         {"head_mr": group_recalls[0],
                #                          "mid_mr": group_recalls[1],
                #                          "tail_mr": group_recalls[2]},
                #                         cur_epoch)

                is_best = val_mr > best_mr
                if is_best:
                    best_mr = val_mr
                    best_epoch = cur_epoch
                    best_group_recalls = val_group_recalls
                if (not cur_epoch % self.save_period) or is_best:
                    self.save_checkpoint(epoch=cur_epoch,
                                         model=self.model,
                                         optimizer=self.optimizer,
                                         is_best=is_best,
                                         mr=val_mr,
                                         group_recalls=val_group_recalls,
                                         prefix='finetune'+self.finetune_name,
                                         save_dir=self.save_dir)

        final_mr = np.around(np.mean(last_mrs), decimals=4)
        final_head_mr = np.around(np.mean(last_head_mrs), decimals=4)
        final_mid_mr = np.around(np.mean(last_mid_mrs), decimals=4)
        final_tail_mr = np.around(np.mean(last_tail_mrs), decimals=4)

        if self.local_rank in [-1, 0]:
            self.logger.info(
                f"\n===> Best mean recall: {best_mr:.2%} (epoch{best_epoch})\n"
                f"Group recalls: {best_group_recalls}\n\n"
                f"===> Final average mean recall of last several epochs:"
                f" {final_mr:.2%}\n"
                f"Average Group mean recalls: [{final_head_mr:.2%}, "
                f"{final_mid_mr:.2%}, {final_tail_mr:.2%}]\n\n"
                f"===> Save directory: '{self.save_dir}'\n"
                f"*********************************************************"
                f"*********************************************************\n"
            )

    def train_epoch(self, cur_epoch, trainloader, model, criterion, optimizer,
                    lr_scheduler, num_classes=None):
        model.train()
        if self.local_rank in [-1, 0]:
            train_pbar = tqdm(
                total=len(trainloader),
                desc=f"Train Epoch[{cur_epoch:>3d}/{self.final_epoch-1}]")

        all_labels, all_preds = [], []
        train_loss_meter = AverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(trainloader):
            optimizer.zero_grad()

            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)
            batch_prob = model(batch_imgs)
            avg_loss = criterion(batch_prob, batch_labels)
            if self.local_rank != -1:
                with amp.scale_loss(avg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                self._reduce_loss(avg_loss)
            else:
                avg_loss.backward()
                optimizer.step()

            batch_pred = batch_prob.max(1)[1]
            train_loss_meter.update(avg_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str("LR:{:.1e} Loss:{:.4f}".format(
                        self.optimizer.param_groups[0]["lr"],
                        train_loss_meter.avg))

        train_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        train_recalls = metrics.recall_score(all_labels, all_preds,
                                             average=None)
        # seperate all classes into 3 groups: Head, Mid, Tail
        if num_classes is not None:
            head_classes = math.floor(num_classes / 3)
            tail_classes = head_classes
            train_group_recalls = [
                np.around(np.mean(train_recalls[:head_classes]),
                          decimals=4),
                np.around(np.mean(train_recalls[
                                    head_classes:num_classes-tail_classes]),
                          decimals=4),
                np.around(np.mean(train_recalls[num_classes-tail_classes:]),
                          decimals=4),
            ]
        else:
            train_group_recalls = [0., 0., 0.]
        if self.local_rank in [-1, 0]:
            train_pbar.set_postfix_str(
                f"LR:{optimizer.param_groups[-1]['lr']:.1e} "
                f"Loss:{train_loss_meter.avg:.2f} "
                f"MR:{train_mr:.2%} "
                f"Head:{train_group_recalls[0]:.0%} "
                f"Mid:{train_group_recalls[1]:.0%} "
                f"Tail:{train_group_recalls[2]:.0%}")
            train_pbar.close()

        return train_mr, train_group_recalls, train_loss_meter.avg

    def evaluate(self, cur_epoch, valloader, model, criterion, num_classes):
        model.eval()

        if self.local_rank in [-1, 0]:
            val_pbar = tqdm(total=len(valloader), ncols=0,
                            desc="                 Val")

        all_labels = []
        all_preds = []
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(valloader):
                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
                batch_probs = model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]
                avg_loss = criterion(batch_probs, batch_labels)
                val_loss_meter.update(avg_loss.item(), 1)

                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_preds.extend(batch_preds.cpu().numpy().tolist())

                val_pbar.update()

        val_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        val_recalls = metrics.recall_score(all_labels, all_preds, average=None)
        # val_recalls = np.around(val_recalls, decimals=4).tolist()

        # seperate all classes into 3 groups: Head, Mid, Tail
        if num_classes is not None:
            head_classes = math.floor(num_classes / 3)
            tail_classes = head_classes
            val_group_recalls = [
                np.around(np.mean(val_recalls[:head_classes]),
                          decimals=4),
                np.around(np.mean(val_recalls[
                                head_classes:num_classes-tail_classes]),
                          decimals=4),
                np.around(np.mean(val_recalls[num_classes-tail_classes:]),
                          decimals=4),
            ]
        else:
            val_group_recalls = [0., 0., 0.]
        if self.local_rank in [-1, 0]:
            val_pbar.set_postfix_str(
                f"Loss:{val_loss_meter.avg:.2f} MR:{val_mr:.2%} "
                f"Head:{val_group_recalls[0]:.0%} "
                f"Mid:{val_group_recalls[1]:.0%} "
                f"Tail:{val_group_recalls[2]:.0%}")
            val_pbar.close()

        return val_mr, val_group_recalls, val_loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local Rank for\
                        distributed training. if single-GPU, default: -1")
    parser.add_argument("--config_path", type=str, help="path of config file")
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
    finetuner = FineTuner(local_rank=args.local_rank, config=config)
    finetuner.finetune()


if __name__ == "__main__":
    args = parse_args()
    main(args)
