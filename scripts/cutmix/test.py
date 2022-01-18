"""finetune script """
import argparse
import os
import random
import warnings
# from datetime import datetime
from os.path import join

import numpy as np
import torch
import yaml
# from apex import amp
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

    def __init__(self, local_rank, config, seed):

        #######################################################################
        # Device setting
        #######################################################################
        self.local_rank = local_rank
        self.seed = seed

        if self.local_rank != -1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        #######################################################################
        # Experiment setting
        #######################################################################
        self.exp_config = config["experiment"]
        self.exp_name = self.exp_config["name"]
        self.test_config = config["test"]
        self.test_name = self.test_config["name"]

        self.user_root = os.environ["HOME"]
        self.exp_root = join(self.user_root, "Experiments")
        self._set_configs(config)
        self.resume = True

        if "/" in self.exp_config["resume_fpath"]:
            self.resume_fpath = self.exp_config["resume_fpath"]
        else:
            self.resume_fpath = join(
                self.exp_root, self.exp_name,
                f"seed_{self.seed}_{self.exp_config['resume_fpath']}")

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

    def evaluate(self, cur_epoch, valloader, model, criterion,
                 num_samples_per_cls, **kwargs):

        model.eval()

        ft_model = kwargs.pop("ft_model", "None")

        if ft_model is not None:
            ft_model.eval()

        if self.local_rank <= 0:
            desc = kwargs.pop("desc", "Val")
            eval_pbar = tqdm(total=len(valloader),
                             ncols=0,
                             desc=f"                 {desc}")

        eval_loss_meter = AverageMeter()
        eval_stat = ExpStat(num_samples_per_cls)
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

                if self.local_rank != -1:
                    torch.distributed.barrier()
                    # torch.distributed.barrier()的作用是，阻塞进程，确保每个进程都运行
                    # 到这一行代码，才能继续执行，这样计算平均loss和平均acc的时候
                    # 不会出现因为进程执行速度不一致而导致的错误
                    avg_loss = self._reduce_tensor(avg_loss)

                eval_loss_meter.update(avg_loss.item(), 1)

                batch_preds = torch.argmax(batch_probs, dim=1)
                eval_stat.update(batch_labels, batch_preds)

                eval_pbar.update()

        if self.local_rank != -1:
            # all reduce the statistical confusion matrix
            torch.distributed.barrier()
            eval_stat._cm = self._reduce_tensor(eval_stat._cm, op='sum')

        if self.local_rank <= 0:
            eval_pbar.set_postfix_str(f"Loss:{eval_loss_meter.avg:>4.2f} "
                                      f"MR:{eval_stat.mr:>6.2%} "
                                      f"[{eval_stat.group_mr[0]:>6.2%}, "
                                      f"{eval_stat.group_mr[1]:>6.2%}, "
                                      f"{eval_stat.group_mr[2]:>6.2%}]")
            eval_pbar.close()

        return eval_stat, eval_loss_meter.avg

    def test(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################

        if self.local_rank != -1:
            print(f"global_rank {self.local_rank},"
                  f"world_size {self.world_size},"
                  f"local_rank {self.local_rank},"
                  f"val '{self.val_sampler_name}'"
                  f"test '{self.test_sampler_name}'")

        # To get the num_samples_per_cls
        trainset = self.init_dataset(self.trainset_name,
                                     transform=None,
                                     **self.trainset_params)

        val_transform = self.init_transform(self.val_transform_name,
                                            **self.val_transform_params)
        valset = self.init_dataset(self.valset_name,
                                   transform=val_transform,
                                   **self.valset_params)
        val_sampler = self.init_sampler(valset, **self.valloader_params)
        self.valloader = DataLoaderX(dataset=valset,
                                     batch_size=self.val_batchsize,
                                     shuffle=False,
                                     num_workers=self.val_workers,
                                     pin_memory=True,
                                     drop_last=False,
                                     sampler=val_sampler)

        test_transform = self.init_transform(self.test_transform_name,
                                             **self.test_transform_params)
        testset = self.init_dataset(self.testset_name,
                                    transform=test_transform,
                                    **self.testset_params)
        test_sampler = self.init_sampler(testset, **self.testloader_params)
        self.testloader = DataLoaderX(dataset=testset,
                                      batch_size=self.test_batchsize,
                                      shuffle=False,
                                      num_workers=self.test_workers,
                                      pin_memory=True,
                                      drop_last=False,
                                      sampler=test_sampler)

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     resume=True,
                                     checkpoint=self.checkpoint,
                                     num_classes=trainset.num_classes,
                                     **self.network_params)

        #######################################################################
        # Initialize Loss
        #######################################################################
        # Default: weight=None
        self.criterion = self.init_loss(self.loss_name, **self.loss_params)

        #######################################################################
        # Start Testing
        #######################################################################
        cur_epoch = 0

        val_sampler.set_epoch(cur_epoch)
        val_stat, val_loss = self.evaluate(
            cur_epoch=cur_epoch,
            valloader=self.valloader,
            model=self.model,
            criterion=self.criterion,
            num_samples_per_cls=trainset.num_samples_per_cls)

        test_stat, test_loss = self.evaluate(
            cur_epoch=cur_epoch,
            valloader=self.testloader,
            model=self.model,
            criterion=self.criterion,
            num_samples_per_cls=trainset.num_samples_per_cls,
            desc="Test")

        if self.local_rank <= 0:
            self.log(f"Val Loss={val_loss:>4.2f} "
                     f"MR={val_stat.mr:>6.2%} "
                     f"[{val_stat.group_mr[0]:>6.2%}, "
                     f"{val_stat.group_mr[1]:>6.2%}, "
                     f"{val_stat.group_mr[2]:>6.2%}")
            self.log(f"Test Loss={test_loss:>4.2f} "
                     f"MR={test_stat.mr:>6.2%} "
                     f"[{test_stat.group_mr[0]:>6.2%}, "
                     f"{test_stat.group_mr[1]:>6.2%}, "
                     f"{test_stat.group_mr[2]:>6.2%}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int,
                        help="Local Rank for distributed training. "
                        "if single-GPU, default: -1")
    parser.add_argument("--config_path", type=str, help="path of config file")
    parser.add_argument("--seed", type=int, default=0, help="rand_seed")
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
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True  # 固定内部随机性
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # 输入尺寸一致，加速训练


def main(args):
    warnings.filterwarnings("ignore")
    _set_random_seed(seed=args.seed)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tester = Tester(local_rank=args.local_rank, config=config, seed=args.seed)
    tester.test()


if __name__ == "__main__":
    args = parse_args()
    main(args)
