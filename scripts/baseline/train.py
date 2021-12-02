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
from utils import AverageMeter


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=8)


class Trainer(BaseTrainer):
    def __init__(self, local_rank=None, config=None):
        super(Trainer, self).__init__(local_rank, config)

    def train(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
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
            print(f"global_rank {self.global_rank},"
                  f"world_size {self.world_size},"
                  f"local_rank {self.local_rank},"
                  f"sampler '{self.train_sampler_name}'")

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
        self.compute_class_weight(trainset.img_num)
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
                                                        opt_level="O1")
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.local_rank],
                                                 output_device=self.local_rank,
                                                 find_unused_parameters=True)
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
        last_20_mr = []
        last_20_head_mr = []
        last_20_mid_mr = []
        last_20_tail_mr = []
        self.final_epoch = self.start_epoch + self.total_epochs
        for cur_epoch in range(self.start_epoch, self.final_epoch):
            # learning rate decay by epoch
            if self.lr_scheduler_mode == "epoch":
                self.lr_scheduler.step()

            if self.local_rank != -1:
                train_sampler.set_epoch(cur_epoch)

            train_mr, train_loss = self.train_epoch(cur_epoch)

            if self.local_rank in [-1, 0]:
                val_mr, val_loss, group_recalls = self.evaluate(cur_epoch)

                if self.final_epoch - cur_epoch <= 20:
                    last_20_mr.append(val_mr)
                    last_20_head_mr.append(group_recalls[0])
                    last_20_mid_mr.append(group_recalls[1])
                    last_20_tail_mr.append(group_recalls[2])

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
                        head=group_recalls[0],
                        mid=group_recalls[1],
                        tail=group_recalls[2],
                    )
                )

                # if len(val_recalls) <= 20 and cur_epoch == self.total_epochs:
                #     self.logger.info(f"Class recalls: {val_recalls}\n")

                # Save log by tensorboard
                self.writer.add_scalar(f"{self.exp_name}/LR",
                                       self.optimizer.param_groups[0]["lr"],
                                       cur_epoch)
                self.writer.add_scalars(f"{self.exp_name}/Loss",
                                        {"train_loss": train_loss,
                                         "val_loss": val_loss},
                                        cur_epoch)
                self.writer.add_scalars(f"{self.exp_name}/Recall",
                                        {"train_mr": train_mr,
                                         "val_mr": val_mr},
                                        cur_epoch)
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

        final_mr = np.around(np.mean(last_20_mr), decimals=4)
        final_head_mr = np.around(np.mean(last_20_head_mr), decimals=4)
        final_mid_mr = np.around(np.mean(last_20_mid_mr), decimals=4)
        final_tail_mr = np.around(np.mean(last_20_tail_mr), decimals=4)

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

        if self.local_rank in [-1, 0]:
            train_pbar = tqdm(
                total=len(self.trainloader),
                desc="Train Epoch[{:>3d}/{}]".format(
                    cur_epoch, self.total_epochs)
            )

        all_labels = []
        all_preds = []
        train_loss_meter = AverageMeter()
        for i, (batch_imgs, batch_labels) in enumerate(self.trainloader):
            if self.lr_scheduler_mode == "iteration":
                self.lr_scheduler.step()

            self.optimizer.zero_grad()
            batch_imgs = batch_imgs.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)

            batch_prob = self.model(batch_imgs)

            avg_loss = self.criterion(batch_prob, batch_labels)
            if self.local_rank != -1:
                with amp.scale_loss(avg_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self._reduce_loss(avg_loss)
            else:
                avg_loss.backward()
                self.optimizer.step()

            batch_pred = batch_prob.max(1)[1]
            train_loss_meter.update(avg_loss.item(), 1)

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(batch_pred.cpu().numpy().tolist())

            if self.local_rank in [-1, 0]:
                train_pbar.update()
                train_pbar.set_postfix_str("LR:{:.1e} Loss:{:.4f}".format(
                        self.optimizer.param_groups[0]["lr"],
                        train_loss_meter.avg,))

        train_mr = metrics.balanced_accuracy_score(all_labels, all_preds)

        if self.local_rank in [-1, 0]:
            train_pbar.set_postfix_str(
                "LR:{:.1e} Loss:{:.2f} MR:{:.2%}".format(
                    self.optimizer.param_groups[0]["lr"],
                    train_loss_meter.avg, train_mr))
            train_pbar.close()

        return train_mr, train_loss_meter.avg

    def evaluate(self, cur_epoch):
        self.model.eval()

        if self.local_rank in [-1, 0]:
            val_pbar = tqdm(total=len(self.valloader), ncols=0,
                            desc="                 Val")

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
        # val_recalls = np.around(val_recalls, decimals=4).tolist()

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
        if self.local_rank in [-1, 0]:
            val_pbar.set_postfix_str(
                f"Loss:{val_loss_meter.avg:.2f} MR:{val_mr:.2%} "
                f"Head:{group_recalls[0]:.0%} "
                f"Mid:{group_recalls[1]:.0%} "
                f"Tail:{group_recalls[2]:.0%}")
            val_pbar.close()

        return val_mr, val_loss_meter.avg, group_recalls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local Rank for\
                        distributed training. if single-GPU, default: -1")
    parser.add_argument("--config_path", type=str, help="path of config file")
    args = parser.parse_args()
    return args


def _set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 固定内部随机性
    torch.backends.cudnn.deterministic = True
    # 输入尺寸一致，加速训练
    torch.backends.cudnn.benchmark = True


def main(args):
    warnings.filterwarnings("ignore")
    _set_random_seed()
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer = Trainer(local_rank=args.local_rank, config=config)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
