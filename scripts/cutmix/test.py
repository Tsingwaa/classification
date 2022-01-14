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


class Tester(BaseTrainer):

    def __init__(self, args, local_rank=None, config=None):

        #######################################################################
        # Device setting
        #######################################################################
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True
        self.args = args
        self.local_rank = local_rank

        if self.local_rank != -1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        #######################################################################
        # Experiment setting
        #######################################################################
        self.exp_config = config["experiment"]
        self.exp_name = self.exp_config["name"]
        self.test_config = config["test"]
        self.test_name = self.test_config["name"]

        self.user_root = os.environ["HOME"]
        self.exp_root = join(self.user_root, "Projects/Experiments")

        self._set_configs(config)

        self.resume = True

        if "/" in self.exp_config["resume_fpath"]:
            self.resume_fpath = self.exp_config["resume_fpath"]
        else:
            # self.resume_fpath = join(self.exp_root, self.exp_name,
            #                          'seed_%d_DRW_%s'%(self.args.seed, self.exp_config["resume_fpath"]))
            self.resume_fpath = join(self.exp_root, self.exp_name,
                                     'DRW_%s'%(self.exp_config["resume_fpath"]))

        self.checkpoint, resume_log = self.resume_checkpoint(self.resume_fpath)


        if self.local_rank in [-1, 0]:
            self.eval_period = self.exp_config["eval_period"]
            self.save_period = self.exp_config["save_period"]
            self.exp_dir = join(self.exp_root, self.exp_name)
            os.makedirs(self.exp_dir, exist_ok=True)

            # Set logger to save .log file and output to screen.
            self.log_fpath = join(self.exp_dir, f"{self.test_name}.log")
            self.logger = self.init_logger(self.log_fpath)
            exp_init_log = f"\n****************************************"\
                f"****************************************************"\
                f"\nExperiment: Test {self.exp_name}\n"\
                f"Save dir: {self.exp_dir}\n"\
                f"Save peroid: {self.save_period}\n"\
                f"Resume Training: {self.resume}\n"\
                f"Distributed Training: "\
                f"{True if self.local_rank != -1 else False}\n"\
                f"**********************************************"\
                f"**********************************************\n"
            self.log(exp_init_log)
            self.log(resume_log)

        test_transform_config = config["test_transform"]
        self.test_transform_name = test_transform_config["name"]
        self.test_transform_params = test_transform_config["param"]

        testset_config = config["test_dataset"]
        self.testset_name = testset_config["name"]
        self.testset_params = testset_config["param"]

        self.testloader_params = config["testloader"]
        self.test_sampler_name = self.testloader_params.pop("sampler", None)
        self.test_batchsize = self.testloader_params["batch_size"]
        self.test_workers = self.testloader_params["num_workers"]

        ft_loss_config = self.test_config["loss"]
        self.loss_name = ft_loss_config["name"]
        self.loss_params = ft_loss_config["param"]

    
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
        val_stat = ExpStat(num_classes)
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


    def test(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        
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

            test_transform = self.init_transform(self.test_transform_name,
                                                **self.test_transform_params)
            testset = self.init_dataset(self.testset_name,
                                       transform=test_transform,
                                       **self.testset_params)
            self.testloader = DataLoaderX(
                dataset=testset,
                batch_size=self.test_batchsize,
                shuffle=False,
                num_workers=self.test_workers,
                pin_memory=True,
                drop_last=False,
            )


        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     resume=True,
                                     checkpoint=self.checkpoint,
                                     num_classes=1000,
                                     **self.network_params)

        #######################################################################
        # Initialize Loss
        #######################################################################
        weight = None
        self.criterion = self.init_loss(self.loss_name,
                                        weight=weight,
                                        **self.loss_params)

        #######################################################################
        # Start Testing
        #######################################################################
        if self.local_rank in [-1, 0]:
            val_stat, val_loss = self.evaluate(
                cur_epoch=0,
                valloader=self.valloader,
                model=self.model,
                criterion=self.criterion,
                num_classes=1000)

            self.log(
                f"Val Loss={val_loss:>4.2f} "
                f"MR={val_stat.mr:>6.2%} "
                f"[{val_stat.group_mr[0]:>6.2%}, "
                f"{val_stat.group_mr[1]:>6.2%}, "
                f"{val_stat.group_mr[2]:>6.2%}",
                log_level="file")

            val_stat, val_loss = self.evaluate(
                cur_epoch=0,
                valloader=self.testloader,
                model=self.model,
                criterion=self.criterion,
                num_classes=1000)

            self.log(
                f"Val Loss={val_loss:>4.2f} "
                f"MR={val_stat.mr:>6.2%} "
                f"[{val_stat.group_mr[0]:>6.2%}, "
                f"{val_stat.group_mr[1]:>6.2%}, "
                f"{val_stat.group_mr[2]:>6.2%}",
                log_level="file")


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
    _set_seed(seed=args.seed)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tester = Tester(args, local_rank=args.local_rank, config=config)
    tester.test()


if __name__ == "__main__":
    args = parse_args()
    main(args)
