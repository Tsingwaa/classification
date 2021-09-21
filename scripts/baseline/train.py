"""trainer script """
import random
import logging
import warnings
import argparse
import pudb
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from pudb import set_trace
from prefetch_generator import BackgroundGenerator
# Distribute Package
from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# Custom Package
from base.base_trainer import BaseTrainer
from utils import AccAverageMeter


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
            train_sampler = None

        if self.local_rank != -1:
            print(f'global_rank {self.global_rank},'
                  f'world_size {self.world_size},'
                  f'local_rank {self.local_rank},'
                  f'{self.trainloader_name}')

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
        self.model = self.init_model()
        self.model.cuda()
        if self.resume:
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
        if self.resume:
            self.optimizer.load_state_dict(optimizer_state_dict)

        #######################################################################
        # Initialize DistributedDataParallel
        #######################################################################
        if self.local_rank != -1:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')

            self.model = DDP(self.model,
                             device_ids=[self.local_rank],
                             output_device=self.local_rank,
                             find_unused_parameters=True)

        #######################################################################
        # Initialize LR Scheduler
        #######################################################################
        self.lr_scheduler = self.init_lr_scheduler()
        if self.resume:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        #######################################################################
        # Start Training
        #######################################################################
        best_acc = 0
        best_mr = 0
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            if self.local_rank != -1:
                train_sampler.set_epoch(epoch)

            train_acc, train_mr, train_ap, train_loss = self.train_epoch(epoch)

            if self.local_rank in [-1, 0]:
                eval_acc, eval_mr, eval_ap, eval_loss = self.evaluate(epoch)

                logging.info(
                    "Epoch[{epoch:>3d}/{total_epochs}]'\
                    'Train Acc={train_acc:.2%}, MR={train_mr:.2%}, '\
                    'AP={train_ap:.2%}, Loss={train_loss:.4f} || '\
                    'Eval Acc={eval_acc:.2%}, MR={eval_mr:.2%}, '\
                    'AP={eval_ap:.2%}, Loss={eval_loss:.4f}".format(
                        epoch=epoch,
                        total_epochs=self.total_epochs,
                        train_acc=train_acc,
                        train_mr=train_mr,
                        train_ap=train_ap,
                        train_loss=train_loss,
                        eval_acc=eval_acc,
                        eval_mr=eval_mr,
                        eval_ap=eval_ap,
                        eval_loss=eval_loss
                    )
                )

                # Save log by tensorboard
                self.writer.add_scalar(f'{self.exp_name}/LearningRate',
                                       self.optimizer.param_groups[0]['lr'],
                                       epoch)
                self.writer.add_scalars(f'{self.exp_name}/Loss',
                                        {'train_loss': train_loss,
                                         'eval_loss': eval_loss}, epoch)
                self.writer.add_scalars(f'{self.exp_name}/Accuracy',
                                        {'train_acc': train_acc,
                                         'eval_acc': eval_acc}, epoch)
                self.writer.add_scalars(f'{self.exp_name}/Recall',
                                        {'train_mr': train_mr,
                                         'eval_mr': eval_mr}, epoch)
                self.writer.add_scalars(f'{self.exp_name}/Precision',
                                        {'train_ap': train_ap,
                                         'eval_ap': eval_ap}, epoch)
                # Save checkpoint.
                is_best = (best_acc < eval_acc or best_mr < eval_mr)
                if best_acc < eval_acc:
                    best_acc = eval_acc
                if best_mr < eval_mr:
                    best_mr = eval_mr
                save_fname = '{}_epoch{}_acc{:.2%}_mr{:.2%}_ap{:.2%}_'\
                    'state_dict.pth.tar'.format(
                        self.network_name,
                        str(epoch),
                        eval_acc,
                        eval_mr,
                        eval_ap
                    )
                if not (epoch % self.save_period) or is_best:
                    self.save_checkpoint(epoch, save_fname, is_best,
                                         eval_acc, eval_mr, eval_ap)
            # learning rate decay by epoch
            if self.lr_scheduler_name != 'CyclicLR':
                self.lr_scheduler.step()

    def train_epoch(self, epoch):
        self.model.train()

        train_pbar = tqdm(
            total=len(self.trainloader),
            desc='Train Epoch[{:>3d}/{}]'.format(epoch, self.total_epochs)
        )

        all_labels = []
        all_preds = []
        train_acc_meter = AccAverageMeter()
        train_loss_meter = AccAverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
            batch_prob = self.model(batch_imgs)

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
            train_acc_meter.update(acc, 1)
            train_loss_meter.update(avg_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str(
                    'LR:{:.1e} Loss:{:.4f} Acc:{:.2%}'.format(
                        self.lr_scheduler.get_last_lr()[0],
                        train_loss_meter.avg,
                        train_acc_meter.avg,
                    )
                )

            if self.lr_scheduler_name == 'CyclicLR':
                self.lr_scheduler.step()

        train_acc = metrics.accuracy_score(all_labels, all_preds)
        train_mr = metrics.recall_score(all_labels, all_preds,
                                        average='macro')
        train_ap = metrics.precision_score(all_labels, all_preds,
                                           average='macro')

        train_pbar.set_postfix_str(
            'Loss:{:.4f} Acc:{:.2%} MR:{:.2%} AP:{:.2%}'.format(
                train_loss_meter.avg, train_acc, train_mr, train_ap
            )
        )
        train_pbar.close()

        return train_acc, train_mr, train_ap, train_loss_meter.avg

    def evaluate(self, epoch):
        self.model.eval()

        eval_pbar = tqdm(
            total=len(self.evalloader),
            desc='\t\t\tEval'
        )

        all_labels = []
        all_preds = []
        eval_loss_meter = AccAverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.evalloader):
            batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
            batch_prob = self.model(batch_imgs)
            batch_pred = batch_prob.max(1)[1]
            avg_loss = self.loss(batch_prob, batch_labels)
            eval_loss_meter.update(avg_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            eval_pbar.update()

        eval_acc = metrics.accuracy_score(all_labels, all_preds)
        eval_mr = metrics.recall_score(all_labels, all_preds,
                                       average='macro')
        eval_ap = metrics.precision_score(all_labels, all_preds,
                                          average='macro')

        eval_pbar.set_postfix_str(
            'Loss:{:.4f} Acc:{:.2%} MR:{:.2%} AP:{:.2%}'.format(
                eval_loss_meter.avg, eval_acc, eval_mr, eval_ap
            )
        )
        eval_pbar.close()

        return eval_acc, eval_mr, eval_ap, eval_loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help='Local Rank for\
                        distributed training. if single-GPU, default: -1')
    parser.add_argument("--config_fpath", type=str, help='path of config file')
    args = parser.parse_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    warnings.filterwarnings('ignore')
    set_seed()
    pudb.set_trace()
    with open(args.config_fpath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
