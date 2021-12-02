"""trainer script """
import math
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
from utils import AverageMeter, switch_adv, switch_clean, switch_mix


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class Trainer(BaseTrainer):
    def __init__(self, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)
        adv_config = config['adv']
        self.adv_name = adv_config['name']
        self.adv_param = adv_config['param']
        self.joint_training = self.adv_param['joint_training']
        self.clean_weight = self.adv_param['clean_weight']

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
            print(f'global_rank {self.global_rank},'
                  f'world_size {self.world_size},'
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
        # Initialize Adversarial Training
        #######################################################################
        self.adv_param.update({"model": self.model})
        self.attacker = self.init_module(module_name=self.adv_name,
                                         module_param=self.adv_param)
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
        self.lr_scheduler = self.init_lr_scheduler(self.optimizer)

        #######################################################################
        # Start Training
        #######################################################################
        best_mr = 0.
        best_epoch = 1
        best_group_recalls = []
        last_20_mrs = []
        last_20_head_mrs = []
        last_20_mid_mrs = []
        last_20_tail_mrs = []
        for cur_epoch in range(self.start_epoch, self.total_epochs + 1):
            # learning rate decay by epoch
            if self.lr_scheduler_mode == 'epoch':
                self.lr_scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            train_mr, train_loss = self.train_epoch(cur_epoch)

            if self.local_rank in [-1, 0]:
                self.model.apply(switch_clean)
                val_mr, val_loss, group_recalls = self.evaluate(cur_epoch)

                if self.total_epochs - cur_epoch <= 20:
                    last_20_mrs.append(val_mr)
                    last_20_head_mrs.append(group_recalls[0])
                    last_20_mid_mrs.append(group_recalls[1])
                    last_20_tail_mrs.append(group_recalls[2])

                self.logger.debug(
                    'Epoch[{epoch:>3d}/{total_epochs}] '
                    'Trainset Loss={train_loss:.4f} '
                    'ADV={adv_loss:.4f} MR={adv_mr:.2%} '
                    'CLN={clean_loss:.4f} MR={clean_mr:.2%}'
                    '|| Valset Loss={val_loss:.4f} MR={val_mr:.2%} '
                    'Head={head:.2%} Mid={mid:.2%} Tail={tail:.2%}'.format(
                        epoch=cur_epoch,
                        total_epochs=self.total_epochs,
                        train_loss=train_loss['final'],
                        adv_loss=train_loss['adv'],
                        adv_mr=train_mr['adv'],
                        clean_loss=train_loss['clean'],
                        clean_mr=train_mr['clean'],
                        val_loss=val_loss,
                        val_mr=val_mr,
                        head=group_recalls[0],
                        mid=group_recalls[1],
                        tail=group_recalls[2],
                    )
                )

                # if len(val_recalls) <= 20 and cur_epoch == self.total_epochs:
                #     self.logger.info(f"Class recalls: {val_recalls}\n")

                # Save log by tensorboard
                self.writer.add_scalar(f'{self.exp_name}/LearningRate',
                                       self.optimizer.param_groups[0]['lr'],
                                       cur_epoch)
                self.writer.add_scalars(f'{self.exp_name}/Loss',
                                        {'train_loss': train_loss['final'],
                                         'adv_loss': train_loss['adv'],
                                         'clean_loss': train_loss['clean'],
                                         'val_loss': val_loss}, cur_epoch)
                self.writer.add_scalars(f'{self.exp_name}/Recall',
                                        {'train_adv_mr': train_mr['adv'],
                                         'train_clean_mr': train_mr['clean'],
                                         'val_mr': val_mr}, cur_epoch)
                self.writer.add_scalars(f"{self.exp_name}/GroupRecall",
                                        {"head_mr": group_recalls[0],
                                         "mid_mr": group_recalls[1],
                                         "tail_mr": group_recalls[2]},
                                        cur_epoch)
                is_best = val_mr > best_mr
                if is_best:
                    best_mr = val_mr
                    best_epoch = cur_epoch
                    best_group_recalls = group_recalls
                self.save_checkpoint(cur_epoch, is_best, val_mr, group_recalls)

        final_mr = np.around(np.mean(last_20_mrs), decimals=4)
        final_head_mr = np.around(np.mean(last_20_head_mrs), decimals=4)
        final_mid_mr = np.around(np.mean(last_20_mid_mrs), decimals=4)
        final_tail_mr = np.around(np.mean(last_20_tail_mrs), decimals=4)

        if self.local_rank in [-1, 0]:
            self.logger.info(
                f"\n===> Best mean recall: {best_mr:.2%} (epoch{best_epoch})\n"
                f"Group recalls: {best_group_recalls}\n\n"
                f"===> Final average mean recall of last 20 epochs:"
                f" {final_mr:.2%}\n"
                f"Average Group mean recalls: [{final_head_mr:.2%}, "
                f"{final_mid_mr:.2%}, {final_tail_mr:.2%}]\n\n"
                f"===> Save directory: '{self.save_dir}'\n"
                f"*********************************************************"
                f"*********************************************************"
            )

    def train_epoch(self, cur_epoch):
        self.model.train()

        train_pbar = tqdm(
            total=len(self.trainloader),
            desc='Train Epoch[{:>3d}/{}]'.format(cur_epoch, self.total_epochs)
        )

        all_labels = []
        all_adv_preds = []
        final_loss_meter = AverageMeter()
        if self.joint_training:
            all_clean_preds = []
            adv_loss_meter = AverageMeter()
            clean_loss_meter = AverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.trainloader):
            if self.lr_scheduler_mode == 'iteration':
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)

            # Adversarial Training
            # Step 1: generate perturbed samples
            batch_adv_imgs = self.attacker.attack(batch_imgs, batch_labels)

            # Step 2: train with perturbed imgs
            if not self.joint_training:
                # Only adversarial training
                self.model.apply(switch_adv)
                batch_adv_probs = self.model(batch_adv_imgs)
                batch_final_loss = self.criterion(batch_adv_probs,
                                                  batch_labels)
            else:
                # Joint clean and adversarial training, 并行加速运算
                batch_mix_imgs = torch.cat((batch_imgs, batch_adv_imgs), 0)
                self.model.apply(switch_mix)
                batch_mix_probs = self.model(batch_mix_imgs)
                # 将batch_mix_probs沿着0维，等分切为两份, 分别计算loss
                batch_probs, batch_adv_probs = batch_mix_probs.chunk(2, 0)
                batch_clean_loss = self.criterion(batch_probs, batch_labels)
                batch_adv_loss = self.criterion(batch_adv_probs, batch_labels)
                batch_final_loss = self.clean_weight * batch_clean_loss +\
                    (1 - self.clean_weight) * batch_adv_loss

            if self.local_rank != -1:
                with amp.scale_loss(batch_final_loss, self.optimizer)\
                        as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()
                self._reduce_loss(batch_final_loss)
            else:
                batch_final_loss.backward()
                self.optimizer.step()

            batch_adv_pred = batch_adv_probs.max(1)[1]
            final_loss_meter.update(batch_final_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_adv_preds.extend(batch_adv_pred.cpu().numpy().tolist())

            if self.joint_training:
                adv_loss_meter.update(batch_adv_loss.item(), 1)
                clean_loss_meter.update(batch_clean_loss.item(), 1)

                batch_clean_preds = batch_probs.max(1)[1]
                all_clean_preds.extend(
                    batch_clean_preds.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                if self.joint_training:
                    train_pbar.set_postfix_str(
                        "LR:{:.1e} Loss:{:.2f} Adv:{:.2f} Cln:{:.2f}".format(
                            self.optimizer.param_groups[0]['lr'],
                            final_loss_meter.avg,
                            adv_loss_meter.avg,
                            clean_loss_meter.avg,))
                else:
                    train_pbar.set_postfix_str(
                        'LR:{:.1e} Loss:{:.2f}'.format(
                            self.optimizer.param_groups[0]['lr'],
                            final_loss_meter.avg,))

        train_adv_mr = metrics.balanced_accuracy_score(all_labels,
                                                       all_adv_preds)

        if self.joint_training:
            train_clean_mr = metrics.balanced_accuracy_score(all_labels,
                                                             all_clean_preds)
            train_mr = {'adv': train_adv_mr, 'clean': train_clean_mr}
            train_loss = {'final': final_loss_meter.avg,
                          'adv': adv_loss_meter.avg,
                          'clean': clean_loss_meter.avg}
            postfix_str = 'LR:{:.1e} Loss:{:.1f} Adv:{:.1f} MR:{:.2%} | '\
                'Cln:{:.1f} MR:{:.2%}'.format(
                    self.optimizer.param_groups[0]['lr'],
                    final_loss_meter.avg,
                    adv_loss_meter.avg, train_adv_mr,
                    clean_loss_meter.avg, train_clean_mr)
        else:
            train_mr = {'adv': train_adv_mr}
            train_loss = {'final': final_loss_meter.avg}
            postfix_str = 'LR:{:.1e} Adv Loss:{:.2f} MR:{:.2%}'.format(
                self.optimizer.param_groups[0]['lr'],
                final_loss_meter.avg,
                train_adv_mr)
        train_pbar.set_postfix_str(postfix_str)
        train_pbar.close()

        return train_mr, train_loss

    def evaluate(self, cur_epoch):
        self.model.eval()

        val_pbar = tqdm(total=len(self.valloader), ncols=0,
                        desc='                 Val')

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
        # val_recalls = np.around(val_recalls, decimals=2).tolist()

        # seperate all classes into 3 groups: Head, Mid, Tail
        num_classes = self.network_param['num_classes']
        head_classes = math.floor(num_classes / 3)
        tail_classes = head_classes
        group_recalls = [
            np.around(
                np.mean(val_recalls[:head_classes]),
                decimals=4),
            np.around(
                np.mean(val_recalls[head_classes:num_classes-tail_classes]),
                decimals=4),
            np.around(
                np.mean(val_recalls[num_classes-tail_classes:]),
                decimals=4),
        ]

        val_pbar.set_postfix_str(
            f"Loss:{val_loss_meter.avg:.2f} MR:{val_mr:.2%} "
            f"Head:{group_recalls[0]:.0%} "
            f"Mid:{group_recalls[1]:.0%} "
            f"Tail:{group_recalls[2]:.0%}")
        val_pbar.close()

        return val_mr, val_loss_meter.avg, group_recalls


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
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
