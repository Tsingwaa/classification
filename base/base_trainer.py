"""Base Trainer"""
# ############# Build-in Package #############
import os
import math
import abc
# import shutil
import logging
from os.path import join
# ########### Third-Party Package ############
import numpy as np
import torch
# from pudb import set_trace
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
# ############## Custom package ##############
from data_loader.dataset.builder import build_dataset
from data_loader.sampler.builder import build_sampler
from data_loader.transform.builder import build_transform
from model.loss.builder import build_loss
from model.module.builder import build_module
from model.network.builder import build_network
from utils import GradualWarmupScheduler


class BaseTrainer:
    def __init__(self, local_rank=-1, config=None):
        """ Base trainer for all experiments.  """

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
        self.experiment_config = config['experiment']
        self.exp_name = self.experiment_config['name']
        self.user_root = os.environ['HOME']
        self.start_epoch = self.experiment_config['start_epoch']
        self.total_epochs = self.experiment_config['total_epochs']
        self.resume = self.experiment_config['resume']
        if '/' in self.experiment_config['resume_fpath']:
            self.resume_fpath = self.experiment_config['resume_fpath']
        else:
            self.resume_fpath = join(
                self.user_root, 'Experiments', self.exp_name,
                self.experiment_config['resume_fpath']
            )

        if self.resume:
            self.checkpoint, resume_log = self.resume_checkpoint()
            self.start_epoch = self.checkpoint['epoch'] + 1

        if self.local_rank in [-1, 0]:
            self.save_dir = join(
                self.user_root, 'Experiments', self.exp_name
            )
            self.tb_dir = join(
                self.user_root, 'Experiments', 'Tensorboard', self.exp_name
            )
            self.log_fname = self.experiment_config['log_fname']
            self.log_fpath = join(self.save_dir, self.log_fname)
            self.save_period = self.experiment_config['save_period']
            self.eval_period = self.experiment_config['eval_period']

            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)

            self.writer = SummaryWriter(log_dir=self.tb_dir)

            # Set logger to save .log file and output to screen.
            self.logger = self.init_logger(self.log_fpath)

            exp_init_log = f'\n****************************************'\
                f'****************************************************'\
                f'\nExperiment: {self.exp_name}\n'\
                f'Start_epoch: {self.start_epoch}\n'\
                f'Total_epochs: {self.total_epochs}\n'\
                f'Save dir: {self.save_dir}\n'\
                f'Tensorboard dir: {self.tb_dir}\n'\
                f'Save peroid: {self.save_period}\n'\
                f'Resume Training: {self.resume}\n'\
                f'Distributed Training: '\
                f'{True if self.local_rank != -1 else False}\n'\
                f'**********************************************'\
                f'**********************************************\n'
            self.log(exp_init_log)
            if self.resume:
                self.log(resume_log)

        self._set_configs(config)

    def _set_configs(self, config):
        #######################################################################
        # Dataset setting
        #######################################################################
        self.train_transform_config = config['train_transform']
        self.trainset_config = config['train_dataset']
        self.val_transform_config = config['val_transform']
        self.valset_config = config['val_dataset']

        #######################################################################
        # Dataloader setting
        #######################################################################
        self.trainloader_config = config['trainloader']
        self.trainloader_name = self.trainloader_config['name']
        self.trainloader_param = self.trainloader_config['param']
        self.train_sampler_name = self.trainloader_param['sampler']
        self.train_batch_size = self.trainloader_param['batch_size']
        self.train_num_workers = self.trainloader_param['num_workers']

        self.valloader_config = config['valloader']
        self.valloader_name = self.valloader_config['name']
        self.valloader_param = self.valloader_config['param']
        self.val_batch_size = self.valloader_param['batch_size']
        self.val_num_workers = self.valloader_param['num_workers']

        #######################################################################
        # Network setting
        #######################################################################
        self.network_config = config['network']
        self.network_name = self.network_config['name']
        self.network_param = self.network_config['param']

        #######################################################################
        # Loss setting
        #######################################################################
        self.loss_config = config['loss']
        self.loss_name = self.loss_config['name']
        self.loss_param = self.loss_config['param']

        #######################################################################
        # Optimizer setting
        #######################################################################
        self.optimizer_config = config['optimizer']
        self.optimizer_name = self.optimizer_config['name']
        self.optimizer_param = self.optimizer_config['param']
        self.weight_decay = self.optimizer_param['weight_decay']

        #######################################################################
        # LR scheduler setting
        #######################################################################
        self.warmup_lr_scheduler_config = config['warmup_lr_scheduler']
        self.warmup = self.warmup_lr_scheduler_config['warmup']
        self.warmup_param = self.warmup_lr_scheduler_config['param']
        self.lr_scheduler_config = config['lr_scheduler']
        self.lr_scheduler_name = self.lr_scheduler_config['name']
        self.lr_scheduler_param = self.lr_scheduler_config['param']
        self.lr_scheduler_mode = 'epoch' \
            if self.lr_scheduler_name != "CyclicLR" else 'iterations'

    def init_transform(self, transform_config=None, log_level='default'):
        transform_name = transform_config['name']
        transform_param = transform_config['param']
        transform = build_transform(transform_name, **transform_param)
        transform_init_log = f'===> Initialized {transform_name}: '\
                             f'{transform_param}'
        self.log(transform_init_log, log_level)
        return transform

    def init_dataset(self, dataset_config=None, transform=None,
                     log_level='default'):
        dataset_name = dataset_config['name']
        dataset_param = dataset_config['param']
        dataset_param['data_root'] = join(self.user_root, 'Data',
                                          dataset_param['data_root'])
        dataset_param['transform'] = transform
        dataset = build_dataset(dataset_name, **dataset_param)

        dataset_init_log = f'===> Initialized {dataset_param["phase"]}'\
            f' {dataset_name}: size={len(dataset)},'\
            f' classes={dataset.cls_num}'
        if dataset_param['phase'] == 'train':
            self.train_size = len(dataset)
            dataset_init_log += f'\nimgs_per_cls={dataset.img_num}'
        self.log(dataset_init_log, log_level)
        return dataset

    def init_sampler(self, dataset=None, sampler_param=None,
                     log_level='default'):
        if sampler_param is None:
            sampler_param = self.trainloader_param
        sampler_name = sampler_param['sampler']
        if sampler_name in {'None', ''}:
            sampler = None
            sampler_init_log = '===> Initialized default sampler'
        elif sampler_name == 'DistributedSampler':
            sampler = DistributedSampler(dataset)
            sampler_init_log = '===> Initialized DistributedSampler'
        else:
            sampler_param['dataset'] = dataset
            sampler = build_sampler(sampler_name, **sampler_param)
            sampler_init_log = f'===> Initialized {sampler_name} with'\
                f' resampled size={len(sampler)}'
        self.log(sampler_init_log, log_level)
        return sampler

    def init_model(self, network_name=None, network_param=None,
                   log_level='default'):
        if network_name is None:
            network_name = self.network_name
        if network_param is None:
            network_param = self.network_param

        model = build_network(network_name, **network_param)

        # Count the total amount of parameters with gradient.
        total_params = 0.
        for x in filter(lambda p: p.requires_grad, model.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        total_params /= 10 ** 6
        model.cuda()

        init = 'Initialized'
        if self.resume:
            model.load_state_dict(self.checkpoint['model'])
            init = 'Resumed'
        elif network_param['pretrained']:
            state_dict = torch.load(network_param['pretrained_fpath'],
                                    map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            init = 'Resumed pretrained'

        model_init_log = f"===> {init} {network_name}(total_params"\
            f"={total_params:.2f}m): {network_param}"
        self.log(model_init_log, log_level)
        return model

    def freeze_model(self, model, unfreeze_keys=['fc']):
        """Freeze model parameters except some given keys
        Default: leave fc unfreezed
        """
        for named_key, var in model.named_parameters():
            if any(key in named_key for key in unfreeze_keys):
                var.requires_grad = True
            else:
                var.requires_grad = False

    def init_optimizer(self, model, optimizer_name, optimizer_param=None,
                       checkpoint=None, resume=True):
        # model_params = [
        #     {
        #         'params': [p for n, p in self.model.named_parameters()
        #                    if not any(nd in n for nd in ['bias', 'bn'])],
        #         'weight_decay': self.weight_decay
        #     },
        #     {
        #         'params': [p for n, p in self.model.named_parameters()
        #                    if any(nd in n for nd in ['bias', 'bn'])],
        #         'weight_decay': 0.0
        #     }
        # ]
        if optimizer_name is None:
            optimizer_name = self.optimizer_name
        if optimizer_param is None:
            optimizer_param = self.optimizer_param
        try:
            optimizer = getattr(torch.optim, optimizer_name)(
                model.parameters(), **optimizer_param)
            if resume and self.resume:
                if checkpoint is None:
                    checkpoint = self.checkpoint
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            if self.local_rank in [-1, 0]:
                self.log(f'===> Initialized {optimizer_name}: '
                         f'{optimizer_param}')
            return optimizer
        except Exception as error:
            raise AttributeError(f'Optimizer initialize failed: {error} !')

    def init_lr_scheduler(self, optimizer, warmup=True):
        lrs_param = self.lr_scheduler_param
        if not warmup:
            self.warmup = False
        if self.lr_scheduler_name == 'CyclicLR':
            self.iter_num = math.ceil(self.train_size / self.train_batch_size)
            lrs_param['step_size_up'] *= self.iter_num
            lrs_param['step_size_down'] *= self.iter_num
        try:
            lr_scheduler = getattr(torch.optim.lr_scheduler,
                                   self.lr_scheduler_name)(
                                       optimizer, **lrs_param)
            self.log(
                f"===> Initialized {self.lr_scheduler_name}: {lrs_param}")
            if self.warmup:
                lr_scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=self.warmup_param['multiplier'],
                    warmup_epochs=self.warmup_param['warmup_epochs'],
                    after_scheduler=lr_scheduler,)
                self.log(f'===> Initialized warmup scheduler: '
                         f'{self.warmup_param}')
            return lr_scheduler
        except Exception as error:
            raise AttributeError(f'LR scheduler initial failed: {error} !')

    def init_module(self, module_name=None, module_param=None):
        module = build_module(module_name, **module_param)
        module_param.pop('model', None)
        self.log(f'===> Initialized {module_name}: {module_param}')
        return module

    def init_loss(self, loss_name=None, **kwargs):
        if loss_name is None:
            loss_name = self.loss_name
        loss = build_loss(loss_name, **self.loss_param)
        self.log(f'===> Initialized {loss_name}: {self.loss_param}')
        return loss

    def compute_class_weight(self, imgs_per_cls, **kwargs):
        """
        Args:
            imgs_per_class(List): imgs of each class
            weight_type(Str): select which type of weight
        Return:
            weight(Tensor): 1-D torch.Tensor
        """
        weight_type = self.loss_param['weight_type']
        if not isinstance(imgs_per_cls, torch.Tensor):
            imgs_per_cls = torch.FloatTensor(imgs_per_cls)

        if weight_type == 'class_weight':
            num_img = torch.sum(imgs_per_cls)
            num_cls = len(imgs_per_cls)
            weight = num_img / (num_cls * imgs_per_cls)
            weight /= torch.sum(weight)
        elif weight_type == 'CB':
            beta = self.loss_param['beta']
            weight = (1.0 - beta) / (1.0 - torch.pow(beta, imgs_per_cls))
            weight /= torch.sum(weight)
        else:
            weight = None

        if weight is not None:
            display_weight = weight.numpy().round(2)
            self.log(f'===> Computed class_weight:\n{display_weight}')

        self.loss_param.update({'weight': weight})

    def _reduce_loss(self, tensor):
        with torch.no_grad():
            dist.reduce(tensor, dst=0)
            if self.local_rank == 0:
                tensor /= self.world_size

    def resume_checkpoint(self, resume_fpath=None):
        if resume_fpath in {None, ''}:
            resume_fpath = self.resume_fpath
        checkpoint = torch.load(resume_fpath, map_location='cpu')
        mr = checkpoint['mr']
        recalls = checkpoint.get('group_recalls', None)
        resume_log = f'===> Resume checkpoint from "{resume_fpath}".\n'\
            f'Mean recall:{mr:.2%}\nGroup recalls:{recalls}\n'
        return checkpoint, resume_log

    def save_checkpoint(self, epoch, is_best=False, mr=None, ap=None,
                        group_recalls=[], prefix=None):
        if (not epoch % self.save_period) or is_best:
            checkpoint = {'model': self.model.state_dict()
                          if self.local_rank == -1 else
                          self.model.module.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'best': is_best,
                          'epoch': epoch,
                          'mr': mr,
                          'group_recalls': group_recalls}
            save_fname = 'best.pth.tar' if is_best else 'last.pth.tar'
            if prefix is not None:
                save_fname = prefix + save_fname
            save_path = join(self.save_dir, save_fname)
            torch.save(checkpoint, save_path)

    def init_logger(self, log_fpath):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Save log to file
        file_handler = logging.FileHandler(log_fpath, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler_formatter = logging.Formatter(
            '%(asctime)s: %(levelname)s:'
            ' [%(filename)s:%(lineno)d]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',)
        file_handler.setFormatter(file_handler_formatter)

        # print to the screen
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        # stream_handler.setFormatter(formatter)

        # add two handler to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def log(self, log, log_level='default'):
        if self.local_rank in [-1, 0]:
            if log_level == 'default':
                # both stram and file
                self.logger.info(log)
            elif log_level == 'stream':
                print(log)
            elif log_level == 'file':
                self.logger.debug(log)

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, epoch):
        raise NotImplementedError
