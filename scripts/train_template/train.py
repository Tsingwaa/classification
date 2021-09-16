"""trainer script """
import random
import logging
import warnings
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from prefetch_generator import BackgroundGenerator
# Distribute Package
from apex import amp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# Custom Package
from common.trainer.trainer import BaseTrainer
from utils import AverageMeter
dist.init_process_group(backend='nccl', init_method='env://')


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


class Trainer(BaseTrainer):
    def __init__(self, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)

    def train(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        train_transform = self.init_transform(self.train_transform_config)
        trainset = self.init_dataset(self.trainset_config, train_transform)

        if self.trainloader_name == "DistributedDataloader":
            train_sampler = DistributedSampler(trainset)
        else:
            raise NotImplementedError("train sampler not implemented for DDP")

        print(f'global_rank {self.global_rank}, world_size {self.world_size},\
              local_rank {self.local_rank},  {self.trainloader_name}')

        self.trainloader = DataLoaderX(
            trainset,
            batch_size=self.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.train_num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )

        if self.local_rank in [-1, 0]:
            eval_transform = self.init_transform(self.eval_transform_config)
            evalset = self.init_dataset(self.evalset_config, eval_transform)
            self.evalloader = DataLoaderX(
                evalset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.eval_num_workers,
                pin_memory=False,
                drop_last=False,
            )

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_network()
        self.model.cuda()
        if self.resume_training:
            model_state_dict, optimizer_state_dict, lr_scheduler_state_dict =\
                self.resume_checkpoint()
            self.model.load_state_dict(model_state_dict)

        #######################################################################
        # Initialize Loss
        #######################################################################
        self.loss = self.init_loss()

        #######################################################################
        # Initialize Optimizer
        #######################################################################
        self.optimizer = self.init_optimizer()
        if self.resume_training:
            self.optimizer.load_state_dict(optimizer_state_dict)

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################
        if self.local_rank != -1:
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1')
            self.model = DDP(self.model,
                             device_ids=[self.local_rank],
                             output_device=self.local_rank,
                             find_unused_parameters=True)

        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.init_lr_scheduler()
        if self.resume_training:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        #######################################################################
        # Start Training
        #######################################################################
        if self.local_rank in [-1, 0]:
            logging.info(
                '\nStart training: \n\
                Total epochs: {total_epochs}\n\
                Trainset size: {trainset_size}\n\
                Train batch size: {train_batch_size}\n\
                Evalset size: {evalset_size}\n\
                Eval batch size: {eval_batch_size}\n\
                '.format(
                    total_epochs=self.total_epochs,
                    trainset_size=len(trainset),
                    train_batch_size=self.train_batch_size,
                    evalset_size=len(evalset),
                    eval_batch_size=self.eval_batch_size
                )
            )

        best_acc = 0
        best_mr = 0
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            if self.local_rank != -1:
                train_sampler.set_epoch(epoch)

            train_acc, train_mr, train_ap, train_loss = self.train_epoch(epoch)

            if self.local_rank in [-1, 0]:
                eval_acc, eval_mr, eval_ap, eval_loss = self.evaluate(epoch)

                # Save log by tensorboard
                self.writer.add_scalars('Loss',
                                        {'train_loss': train_loss,
                                         'eval_loss': eval_loss}, epoch)
                self.writer.add_scalars('Accuracy',
                                        {'train_acc': train_acc,
                                         'eval_acc': eval_acc}, epoch)
                self.writer.add_scalars('Recall',
                                        {'train_mr': train_mr,
                                         'eval_mr': eval_mr}, epoch)
                self.writer.add_scalars('Precision',
                                        {'train_ap': train_ap,
                                         'eval_ap': eval_ap}, epoch)
                # Save checkpoint.
                if self.local_rank in [-1, 0]:
                    is_best = (best_acc < eval_acc or best_mr < eval_mr)
                    if best_acc < eval_acc:
                        best_acc = eval_acc
                    if best_mr < eval_mr:
                        best_mr = eval_mr
                    save_fname = '{}_epoch{}_acc{:.2%}_mr{:.2%}_state_dict\
                            .pth.tar'.format(self.network_name,
                                             str(epoch),
                                             eval_acc,
                                             eval_mr)
                    if not (epoch % self.save_period) or is_best:
                        self.save_checkpoint(epoch, save_fname, is_best,
                                             eval_acc, eval_mr, eval_ap)

    def train_epoch(self, epoch):
        self.model.train()

        train_pbar = tqdm(
            total=len(self.trainloader),
            ncols=10,
            desc='Train Epoch{:>3d}/{}'.format(epoch, self.total_epochs)
        )

        all_labels = []
        all_preds = []
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.trainloader):
            batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
            batch_prob = self.model(batch_imgs)

            self.optimizer.zero_grad()
            avg_loss = self.loss(batch_prob, batch_labels)
            if self.local_rank != -1:
                with amp.scale_loss(avg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self._reduce_loss(avg_loss)
            else:
                avg_loss.backward()
                self.optimizer.step()

            batch_pred = batch_prob.max(1)[1]
            acc = (batch_pred == batch_labels).float().mean()
            acc_meter.update(acc, 1)
            loss_meter.update(avg_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                postfix_info = 'lr:{:.2e} train loss:{:.4f} acc:{:.2%}'.format(
                    self.lr_scheduler.get_last_lr[0],
                    loss_meter.avg,
                    acc_meter.avg,
                )
                train_pbar.set_postfix_str(postfix_info)

                logging.info(
                    "Train Epoch[{epoch:>3d}/{total_epochs}] \
                     Iter[{this_iter}/{total_iters}] LR: {lr:.2e},\
                     Acc: {acc:.2%}, Loss: {loss:.4f}".format(
                        epoch=epoch,
                        total_epochs=self.total_epochs,
                        this_iter=(i + 1),
                        total_iters=len(self.trainloader),
                        lr=self.lr_scheduler.get_last_lr()[0],
                        acc=acc_meter.avg,
                        loss=loss_meter.avg
                    )
                )
            self.lr_scheduler.step()

        train_pbar.close()

        epoch_acc = metrics.accuracy_score(all_labels, all_preds)
        epoch_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        epoch_ap = metrics.average_precision_score(all_labels, all_preds)

        logging.info(
            "Train Epoch[{epoch:>3d}/{total_epochs}] Acc: {acc:.2%},\
            MR: {mr:.2%}, AP: {ap:.2%}, Loss: {loss:.4f}".format(
                epoch=epoch,
                total_epochs=self.total_epochs,
                acc=epoch_acc,
                mr=epoch_mr,
                ap=epoch_ap,
                loss=avg_loss.avg
            )
        )
        return epoch_acc, epoch_mr, epoch_ap, loss_meter.avg

    def evaluate(self, epoch):
        self.model.eval()

        eval_pbar = tqdm(
            total=len(self.evalloader),
            ncols=10,
            desc='Eval Epoch {:>3d}/{}'.format(epoch, self.total_epochs)
        )

        all_labels = []
        all_preds = []
        loss_meter = AverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.evalloader):
            batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
            batch_prob = self.model(batch_imgs)
            batch_pred = batch_prob.max(1)[1]
            avg_loss = self.loss(batch_prob, batch_labels)
            loss_meter.update(avg_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            eval_pbar.update()
            postfix_info = 'Eval loss:{:.4f}'.format(loss_meter.avg)
            eval_pbar.set_postfix_str(postfix_info)
        eval_pbar.close()

        eval_acc = metrics.accuracy_score(all_labels, all_preds)
        eval_mr = metrics.balanced_accuracy_score(all_labels, all_preds)
        eval_ap = metrics.average_precision_score(all_labels, all_preds)

        logging.info(
            "Epoch[{epoch:>3d}/{total_epochs}] Acc: {acc:.2%},\
            MR: {mr:.2%}, AP: {ap:.2%}, Loss: {loss:.4f}".format(
                epoch=epoch,
                total_epochs=self.total_epochs,
                acc=eval_acc,
                mr=eval_mr,
                ap=eval_ap,
                loss=avg_loss.avg
            )
        )
        return eval_acc, eval_mr, eval_ap, loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    return args


def set_seed(seed=0):
    random.set_seed(seed)
    np.random.set_seed(seed)
    torch.set_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    warnings.filterwarnings('ignore')
    set_seed(0)
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
