"""Base Trainer"""
# ############# Build-in Package #############
# import math
import abc
# import shutil
import logging
import os
from os.path import join

# ########### Third-Party Package ############
import numpy as np
import torch
# ############## Custom package ##############
from data_loader.dataset.builder import Datasets
from data_loader.sampler.builder import Samplers
from data_loader.transform.builder import Transforms
from model.loss.builder import Losses
from model.module.builder import Modules
from model.network.builder import Networks
# from pudb import set_trace
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils import GradualWarmupScheduler


class BaseTrainer:

    def __init__(self, local_rank, config, seed):
        """ Base trainer for all experiments.  """

        #######################################################################
        # Device setting
        #######################################################################
        self.local_rank = local_rank
        self.seed = seed

        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()

        #######################################################################
        # Experiment setting
        #######################################################################
        self.exp_config = config["experiment"]
        self.exp_name = self.exp_config["name"]
        self.user_root = os.environ["HOME"]
        self.exp_root = join(self.user_root, "Experiments")
        self.start_epoch = self.exp_config["start_epoch"]
        self.total_epochs = self.exp_config["total_epochs"]

        self._set_configs(config)  # set common configs to run exp

        self.resume = self.exp_config["resume"]

        if self.resume:
            if "/" in self.exp_config["resume_fpath"]:
                self.resume_fpath = self.exp_config["resume_fpath"]
            else:
                self.resume_fpath = join(
                    self.exp_root, self.exp_name,
                    f"seed{self.seed}_{self.exp_config['resume_fpath']}")
            self.checkpoint, resume_log =\
                self.resume_checkpoint(self.resume_fpath)
            self.start_epoch = self.checkpoint["epoch"] + 1

        if self.local_rank in [-1, 0]:
            self.eval_period = self.exp_config["eval_period"]

            # Save experiment result
            self.save_period = self.exp_config["save_period"]
            self.exp_dir = join(self.exp_root, self.exp_name)
            os.makedirs(self.exp_dir, exist_ok=True)

            # Save tensorboard record
            self.tb_dir = join(self.exp_root, "Tensorboard", self.exp_name)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_dir)

            # Save stream and file logging record
            self.log_fpath = join(
                self.exp_dir,
                f"seed{self.seed}_" + self.exp_config['log_fname'],
            )
            self.logger = self.init_logger(self.log_fpath)

            exp_init_log = f"\n****************************************"\
                f"****************************************************"\
                f"\nExperiment: {self.exp_name}\n"\
                f"Start_epoch: {self.start_epoch}\n"\
                f"Total_epochs: {self.total_epochs}\n"\
                f"Save dir: {self.exp_dir}\n"\
                f"Tensorboard dir: {self.tb_dir}\n"\
                f"Save peroid: {self.save_period}\n"\
                f"Resume Training: {self.resume}\n"\
                f"Distributed Training: "\
                f"{True if self.local_rank != -1 else False}\n"\
                f"**********************************************"\
                f"**********************************************\n"

            self.log(exp_init_log)

            if self.resume:
                self.log(resume_log)

    def _set_configs(self, config):
        #######################################################################
        # Dataset setting
        #######################################################################
        train_transform_config = config["train_transform"]
        self.train_transform_name = train_transform_config["name"]
        self.train_transform_params = train_transform_config["param"]
        trainset_config = config["train_dataset"]
        self.trainset_name = trainset_config["name"]
        self.trainset_params = trainset_config["param"]

        val_transform_config = config["val_transform"]
        self.val_transform_name = val_transform_config["name"]
        self.val_transform_params = val_transform_config["param"]
        valset_config = config["val_dataset"]
        self.valset_name = valset_config["name"]
        self.valset_params = valset_config["param"]

        #######################################################################
        # Dataloader setting
        #######################################################################
        self.trainloader_params = config["trainloader"]
        self.train_batchsize = self.trainloader_params["batch_size"]

        if self.local_rank != -1:
            self.train_batchsize = self.train_batchsize // self.world_size
        self.train_workers = self.trainloader_params["num_workers"]
        self.train_sampler_name = self.trainloader_params["sampler"]

        self.valloader_params = config["valloader"]
        self.val_batchsize = self.valloader_params["batch_size"]

        if self.local_rank != -1:
            self.val_batchsize = self.val_batchsize // self.world_size
        self.val_workers = self.valloader_params["num_workers"]
        self.val_sampler_name = self.valloader_params["sampler"]

        #######################################################################
        # Network setting
        #######################################################################
        network_config = config["network"]
        self.network_name = network_config["name"]
        self.network_params = network_config["param"]

        #######################################################################
        # Loss setting
        #######################################################################
        loss_config = config["loss"]
        self.loss_name = loss_config["name"]
        self.loss_params = loss_config["param"]

        #######################################################################
        # Optimizer setting
        #######################################################################
        opt_config = config["optimizer"]
        self.opt_name = opt_config["name"]
        self.opt_params = opt_config["param"]

        #######################################################################
        # LR scheduler setting
        #######################################################################
        scheduler_config = config["lr_scheduler"]
        self.scheduler_name = scheduler_config["name"]
        self.scheduler_params = scheduler_config["param"]

    def init_transform(self, transform_name, **kwargs):
        log_level = kwargs.pop("log_level", "default")

        transform = Transforms.get(transform_name)(**kwargs)

        transform_init_log = f"===> Initialized {kwargs['phase']} "\
            f" {transform_name}: {kwargs}"
        self.log(transform_init_log, log_level)

        return transform

    def init_dataset(self, dataset_name, **kwargs):
        log_level = kwargs.pop("log_level", "default")
        prefix = kwargs.pop("prefix", "")  # tag dataset phase

        dataset = Datasets.get(dataset_name)(**kwargs)

        fold_str = f"fold-{kwargs['fold_i']} " \
            if "fold_i" in kwargs.keys() else ""
        dataset_init_log = f"===> Initialized {kwargs['phase']} "\
            f"{fold_str}{prefix}{dataset_name}: size={len(dataset)}, "\
            f"classes={dataset.num_classes}\n"\
            f"imgs_per_cls={dataset.num_samples_per_cls}"
        self.log(dataset_init_log, log_level)

        return dataset

    def init_sampler(self, sampler_name, **kwargs):
        log_level = kwargs.pop("log_level", "default")

        if self.local_rank != -1:
            sampler = DistributedSampler(kwargs["dataset"])
            sampler_init_log = "===> Initialized DistributedSampler"
        elif sampler_name in {None, "None", ""}:
            sampler = None
            sampler_init_log = "===> Initialized Default Sampler"
        else:
            sampler = Samplers.get(sampler_name)(**kwargs)
            sampler_init_log = f"===> Initialized {sampler_name} with"\
                f" resampled size={len(sampler)}"

        dataset = kwargs.pop("dataset", None)
        phase = dataset.phase
        kwargs.pop("sampler", None)
        sampler_init_log += f" for {phase}loader{kwargs}"
        self.log(sampler_init_log, log_level)

        return sampler

    def init_model(self,
                   network_name,
                   resume=False,
                   checkpoint=None,
                   except_keys=[],
                   **kwargs):
        log_level = kwargs.pop("log_level", "default")

        model = Networks.get(network_name)(**kwargs)
        total_params = self.count_model_params(model)
        model = model.cuda()

        _prefix = "Initialized"

        if resume:
            state_dict = checkpoint["model"]
            model = self.update_state_dict(model,
                                           state_dict,
                                           except_keys=except_keys)
            _prefix = "Resumed checkpoint model_params to"
        elif kwargs.get("pretrained", False):
            pretrained_path = kwargs["pretrained_fpath"]
            pretrained_path = join(self.user_root, pretrained_path)
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model = self.update_state_dict(model,
                                           state_dict,
                                           except_keys=["fc"])
            _prefix = "Resumed pretrained model_params to"

        kwargs.pop("checkpoint", None)
        model_init_log = f"===> {_prefix} {network_name}(total_params"\
            f"={total_params:.2f}m): {kwargs}"
        self.log(model_init_log, log_level)

        return model

    def get_class_weight(self, num_samples_per_cls, weight_type, **kwargs):
        num_samples_per_cls = torch.FloatTensor(num_samples_per_cls)

        if weight_type == "inverse":
            num_samples = torch.sum(num_samples_per_cls)
            num_classes = len(num_samples_per_cls)
            weight = num_samples / (num_classes * num_samples_per_cls)
            weight /= torch.sum(weight)
        elif weight_type == "proportion":
            num_classes = len(num_samples_per_cls)
            num_samples = torch.sum(num_samples_per_cls)
            weight = num_classes * num_samples_per_cls / num_samples
            # weight = 1 / weight
            weight /= torch.sum(weight)
        elif weight_type == "class-balanced":
            beta = kwargs["beta"]
            weight = (1.0 - beta) / \
                (1.0 - torch.pow(beta, num_samples_per_cls))
            weight /= torch.sum(weight)
        else:
            weight = None

        return weight

    def init_loss(self, loss_name, **kwargs):
        log_level = kwargs.pop("log_level", "default")

        weight = kwargs.get('weight', None)

        if weight is not None:
            display_weight = weight.numpy().round(2)
            self.log(f"===> Computed class_weight:\n{display_weight}")

        loss = Losses.get(loss_name)(**kwargs)

        kwargs.pop("weight")
        self.log(f"===> Initialized {loss_name}: {kwargs}", log_level)

        return loss.cuda()

    def init_optimizer(self, opt_name, model_params, **kwargs):
        # model_params = [
        #     {"model_params": [p for n, p in self.model.named_parameters()
        #                    if not any(nd in n for nd in ["bias", "bn"])],
        #      "weight_decay": self.weight_decay},
        #     {"model_params": [p for n, p in self.model.named_parameters()
        #                    if any(nd in n for nd in ["bias", "bn"])],
        #      "weight_decay": 0.0}]
        log_level = kwargs.pop("log_level", "default")

        optimizer = getattr(torch.optim, opt_name)(model_params, **kwargs)
        _prefix = "Initialized"

        # if kwargs.get("resume", False):
        #     checkpoint = kwargs.pop("checkpoint", None)
        #     optimizer = self.update_state_dict(optimizer,
        #                                        checkpoint["optimizer"])
        #     _prefix = "Resumed"

        self.log(f"===> {_prefix} {opt_name}: {kwargs}", log_level)

        return optimizer

    def init_lr_scheduler(self, scheduler_name, optimizer, **kwargs):
        warmup_epochs = kwargs.pop("warmup_epochs", 5)

        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               scheduler_name)(optimizer, **kwargs)
        self.log(f"===> Initialized {scheduler_name}: {kwargs}")

        if warmup_epochs > 0:
            lr_scheduler = GradualWarmupScheduler(optimizer,
                                                  multiplier=1,
                                                  warmup_epochs=warmup_epochs,
                                                  after_scheduler=lr_scheduler)
            self.log(f"===> Initialized gradual warmup scheduler: "
                     f"warmup_epochs={warmup_epochs}")

        return lr_scheduler

    def init_module(self, module_name, **kwargs):
        module = Modules.get(module_name)(**kwargs)
        del kwargs["model"]
        self.log(f"===> Initialized {module_name}: {kwargs}")

        return module

    def _reduce_tensor(self, tensor, op='mean'):
        reduced_tensor = tensor.clone()
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)

        if op == 'mean':
            reduced_tensor /= self.world_size

        return reduced_tensor

    def resume_checkpoint(self, resume_fpath, **kwargs):
        checkpoint = torch.load(resume_fpath, map_location="cpu")
        epoch = checkpoint["epoch"]
        is_best = checkpoint["is_best"]

        resume_log = f"===> Resume checkpoint from '{resume_fpath}'.\n"\
            f"checkpoint epoch: {epoch}\nIs_best: {is_best}\n"

        if "mr" in checkpoint.keys():
            mr = checkpoint["mr"]
            group_mr = checkpoint.get("group_mr", "-")
            resume_log += f"Mean recall: {mr:6.2%} {group_mr}\n"

        return checkpoint, resume_log

    def save_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        is_best,
        stat,
        save_dir,
        prefix=None,
        **kwargs,
    ):

        if self.local_rank == -1:
            model_state_dict = model.state_dict()
        else:
            model_state_dict = model.module.state_dict()

        checkpoint = {
            "model": model_state_dict,
            # "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "is_best": is_best,
            "mr": stat.mr,
            "cm": stat.cm,
            "recalls": stat.recalls,
            "group_mr": stat.group_mr,
        }
        checkpoint.update(kwargs)  # Add custom state dict.

        save_fname = "best.pth.tar" if is_best else "last.pth.tar"

        if prefix is not None:
            save_fname = prefix + "_" + save_fname

        save_path = join(save_dir, save_fname)
        torch.save(checkpoint, save_path)

    def init_logger(self, log_fpath):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Save log to file
        file_handler = logging.FileHandler(log_fpath, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler_formatter = logging.Formatter(
            "%(asctime)s: %(levelname)s:"
            " [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_handler_formatter)

        # print to the screen
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        # stream_handler.setFormatter(formatter)

        # add two handler to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def log(self, log, log_level="default"):
        if self.local_rank in [-1, 0]:
            if log_level == "default":
                # both stram and file
                self.logger.info(log)
            elif log_level == "stream":
                print(log)
            elif log_level == "file":
                self.logger.debug(log)

    def freeze_model(self, model, unfreeze_keys=["fc"]):
        """Freeze model parameters except some given keys
        Default: leave fc unfreezed
        """
        self.log(f"===> Freeze model except for keys{unfreeze_keys}")

        for named_key, var in model.named_parameters():
            if unfreeze_keys is None:
                var.requires_grad = False
            else:
                if any(key in named_key for key in unfreeze_keys):
                    var.requires_grad = True
                else:
                    var.requires_grad = False

    def count_model_params(self, model):
        total_params = 0.

        for x in filter(lambda p: p.requires_grad, model.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        total_params /= 10**6

        return total_params

    def update_state_dict(self, module, checkpoint_state_dict, except_keys=[]):
        """Only update state dict that the module needs and print those
        unupdated keys of the module"""
        module_state_dict = module.state_dict()
        items_to_update = {
            key: value

            for key, value in checkpoint_state_dict.items()

            if (key in module_state_dict.keys() and not any(
                ex_key in key for ex_key in except_keys))
        }
        keys_unupdate = [
            key for key in module_state_dict.keys()

            if key not in items_to_update.keys()
        ]

        self.log(f"Found unupdated keys from checkpoint: {keys_unupdate}")
        module_state_dict.update(items_to_update)
        module.load_state_dict(module_state_dict)

        return module

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, epoch):
        raise NotImplementedError
