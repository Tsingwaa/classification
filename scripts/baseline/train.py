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
        # set_trace()
        train_transform = self.init_transform(self.train_transform_config)
        trainset = self.init_dataset(self.trainset_config, train_transform)

        if self.trainloader_name == 'DistributedDataloader':
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

        #######################################################################
        # Initialize Loss
        #######################################################################
        self.loss = self.init_loss()

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
        last_train_accs = np.zeros(20)
        last_train_mrs = np.zeros(20)
        last_train_aps = np.zeros(20)
        last_train_losses = np.zeros(20)
        last_eval_accs = np.zeros(20)
        last_eval_mrs = np.zeros(20)
        last_eval_aps = np.zeros(20)
        last_eval_losses = np.zeros(20)
        best_acc = 0
        best_mr = 0
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            # learning rate decay by epoch
            if self.lr_scheduler_mode == 'epoch':
                self.lr_scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(epoch)

            train_acc, train_mr, train_ap, train_loss = self.train_epoch(epoch)

            if self.local_rank in [-1, 0]:
                eval_acc, eval_mr, eval_ap, eval_loss = self.evaluate(epoch)

                last_train_accs[epoch % 20] = train_acc
                last_train_mrs[epoch % 20] = train_mr
                last_train_aps[epoch % 20] = train_ap
                last_train_losses[epoch % 20] = train_loss
                last_eval_accs[epoch % 20] = eval_acc
                last_eval_mrs[epoch % 20] = eval_mr
                last_eval_aps[epoch % 20] = eval_ap
                last_eval_losses[epoch % 20] = eval_loss

                self.logger.debug(
                    'Epoch[{epoch:>3d}/{total_epochs}] '
                    'Train Acc={train_acc:.2%}, MR={train_mr:.2%}, '
                    'AP={train_ap:.2%}, Loss={train_loss:.4f} || '
                    'Eval Acc={eval_acc:.2%}, MR={eval_mr:.2%}, '
                    'AP={eval_ap:.2%}, Loss={eval_loss:.4f}'.format(
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
                # is_best = (best_acc < eval_acc or best_mr < eval_mr)
                is_best = best_mr < eval_mr
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
                self.save_checkpoint(epoch, save_fname, is_best,
                                     eval_acc, eval_mr, eval_ap)

        self.logger.info(
            "\n===> End experiment {}, results are saved at '{}'\n"
            "Train Set:\n"
            "\tAverage accuracy of the last 20 epochs: {:.2%}\n"
            "\tAverage recall of the last 20 epochs: {:.2%}\n"
            "\tAverage precision of the last 20 epochs: {:.2%}\n"
            "\tAverage losses of the last 20 epochs: {:.4f}\n"
            "Validation Set:\n"
            "\tAverage accuracy of the last 20 epochs: {:.2%}\n"
            "\tAverage recall of the last 20 epochs: {:.2%}\n"
            "\tAverage precision of the last 20 epochs: {:.2%}\n"
            "\tAverage losses of the last 20 epochs: {:.4f}\n"
            "*********************************************************"
            "*********************************************************".format(
                self.exp_name,
                self.save_dir,
                np.mean(last_train_accs),
                np.mean(last_train_mrs),
                np.mean(last_train_aps),
                np.mean(last_train_losses),
                np.mean(last_eval_accs),
                np.mean(last_eval_mrs),
                np.mean(last_eval_aps),
                np.mean(last_eval_losses),
            )
        )

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
            if self.lr_scheduler_mode == 'iteration':
                self.lr_scheduler.step()

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
                        self.optimizer.param_groups[0]['lr'],
                        train_loss_meter.avg,
                        train_acc_meter.avg,
                    )
                )

        train_acc = metrics.accuracy_score(all_labels, all_preds)
        train_mr = metrics.recall_score(all_labels, all_preds,
                                        average='macro')
        train_ap = metrics.precision_score(all_labels, all_preds,
                                           average='macro')

        train_pbar.set_postfix_str(
            'LR:{:.1e} Loss:{:.2f} Acc:{:.0%} MR:{:.0%} AP:{:.0%}'.format(
                self.optimizer.param_groups[0]['lr'],
                train_loss_meter.avg, train_acc, train_mr, train_ap
            )
        )
        train_pbar.close()

        return train_acc, train_mr, train_ap, train_loss_meter.avg

    def evaluate(self, epoch):
        self.model.eval()

        eval_pbar = tqdm(
            total=len(self.evalloader),
            ncols=0,
            desc='                Eval'
        )

        all_labels = []
        all_preds = []
        eval_loss_meter = AccAverageMeter()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(self.evalloader):
                batch_imgs = batch_imgs.cuda()
                batch_labels = batch_labels.cuda()
                batch_probs = self.model(batch_imgs)
                batch_preds = batch_probs.max(1)[1]
                avg_loss = self.loss(batch_probs, batch_labels)
                eval_loss_meter.update(avg_loss.item(), 1)

                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_preds.extend(batch_preds.cpu().numpy().tolist())

                eval_pbar.update()

        eval_acc = metrics.accuracy_score(all_labels, all_preds)
        eval_mr = metrics.recall_score(all_labels, all_preds,
                                       average='macro')
        eval_ap = metrics.precision_score(all_labels, all_preds,
                                          average='macro')

        eval_pbar.set_postfix_str(
            'Loss:{:.2f} Acc:{:.0%} MR:{:.0%} AP:{:.0%}'.format(
                eval_loss_meter.avg, eval_acc, eval_mr, eval_ap
            )
        )
        eval_pbar.close()

        return eval_acc, eval_mr, eval_ap, eval_loss_meter.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='Local Rank for\
                        distributed training. if single-GPU, default: -1')
    parser.add_argument('--config_fpath', type=str, help='path of config file')
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
    with open(args.config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
