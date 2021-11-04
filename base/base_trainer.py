"""Base Trainer"""
# ############# Build-in Package #############
import os
# import shutil
import logging
from os.path import join
# ########### Third-Party Package ############
import numpy as np
import torch
# from pudb import set_trace
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
# ############## Custom package ##############
from model.loss.builder import build_loss
from model.network.builder import build_network
from model.module.builder import build_module
from data_loader.dataset.builder import build_dataset
from data_loader.sampler.builder import build_sampler
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
            dist.init_process_group(backend='nccl', init_method='env://')
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

        if self.local_rank in [-1, 0]:
            self.save_dir = join(
                self.user_root, 'Experiments', self.exp_name
            )
            self.tb_dir = join(
                self.user_root, 'Experiments/Tensorboard', self.exp_name
            )
            self.log_fname = self.experiment_config['log_fname']
            self.log_fpath = join(self.save_dir, self.log_fname)
            self.save_period = self.experiment_config['save_period']
            self.eval_period = self.experiment_config['eval_period']

            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)

            self.writer = SummaryWriter(self.tb_dir)

            # Set logger to save .log file and output to screen.
            if self.local_rank in [-1, 0]:
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
                    f'Distributed Training:'\
                    f' {True if self.local_rank != -1 else False}\n'\
                    f'**********************************************'\
                    f'**********************************************\n'
                self.logger.info(exp_init_log)

        if self.resume:
            self.checkpoint = self.resume_checkpoint()

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

    def init_transform(self, transform_config=None):
        script_path = transform_config['script_path']
        transform_name = transform_config['name']
        transform_param = transform_config['param']
        module = import_module(script_path)
        transform = getattr(module, transform_name)(**transform_param)

        if self.local_rank in [-1, 0]:
            transform_init_log = f'===> Initialized '\
                f'{transform_param["phase"]}'\
                f' {transform_name} from {script_path}'
            self.logger.info(transform_init_log)
        return transform

    def init_sampler(self, dataset=None):
        sampler_param = self.trainloader_config['param']
        sampler_name = sampler_param['sampler']
        if sampler_name == 'None':
            sampler = None
        elif sampler_name == 'DistributedSampler':
            sampler = DistributedSampler(dataset)
        else:
            sampler_param['dataset'] = dataset
            sampler = build_sampler(sampler_name, **sampler_param)
            if self.local_rank in [-1, 0]:
                sampler_init_log = f'===> Initialized {sampler_name} with'\
                    f' resampled size={len(sampler)}'
                self.logger.info(sampler_init_log)

        return sampler

    def init_dataset(self, dataset_config=None, transform=None):
        # set_trace()
        dataset_name = dataset_config['name']
        dataset_param = dataset_config['param']
        dataset_param['data_root'] = join(
            self.user_root, 'Data', dataset_param['data_root'])
        dataset_param['transform'] = transform
        dataset = build_dataset(dataset_name, **dataset_param)
        if dataset_param['phase'] == 'train':
            self.train_size = len(dataset)

        if self.local_rank in [-1, 0]:
            dataset_init_log = f'===> Initialized {dataset_param["phase"]}'\
                f' {dataset_name}(size={len(dataset)},'\
                f' classes={dataset.cls_num}).'
            self.logger.info(dataset_init_log)

        return dataset

    def init_optimizer(self):
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
        try:
            optimizer = getattr(torch.optim, self.optimizer_name)(
                self.model.parameters(), **self.optimizer_param)
            if self.resume:
                optimizer.load_state_dict(self.checkpoint['optimizer'])
            if self.optimizer_name == 'SGD':
                optimizer_init_log = f'===> Initialized {self.optimizer_name}'\
                    f' with init_lr={self.optimizer_param["lr"]}'\
                    f' momentum={self.optimizer_param["momentum"]}'\
                    f' nesterov={self.optimizer_param["nesterov"]}'
            else:
                optimizer_init_log = f'===> Initialized {self.optimizer_name}'

            if self.local_rank in [-1, 0]:
                self.logger.info(optimizer_init_log)

            return optimizer
        except Exception as error:
            logging.info(f'Optimizer initialize failed: {error} !')
            raise AttributeError(f'Optimizer initialize failed: {error} !')

    def init_lr_scheduler(self):
        if self.lr_scheduler_name == 'CyclicLR':
            self.iter_num = int(
                np.ceil(self.train_size / self.train_batch_size))
            self.lr_scheduler_param['step_size_up'] *= self.iter_num
            self.lr_scheduler_param['step_size_down'] *= self.iter_num
            lr_scheduler_init_log = '===> Initialized {} with step_size_up={}'\
                ' step_size_down={} base_lr={:.0e} max_lr={:.0e}'.format(
                    self.lr_scheduler_name,
                    self.lr_scheduler_param['step_size_up'],
                    self.lr_scheduler_param['step_size_down'],
                    self.lr_scheduler_param['base_lr'],
                    self.lr_scheduler_param['max_lr'],)
        elif self.lr_scheduler_name == 'MultiStepLR':
            lr_scheduler_init_log = '===> Initialized {} with milestones={}'\
                    ' gamma={}'.format(self.lr_scheduler_name,
                                       self.lr_scheduler_param['milestones'],
                                       self.lr_scheduler_param['gamma'])
        else:
            lr_scheduler_init_log = '===> Initialized {} with {}'.format(
                self.lr_scheduler_name,
                self.lr_scheduler_param)
        try:
            lr_scheduler = getattr(torch.optim.lr_scheduler,
                                   self.lr_scheduler_name)(
                                       self.optimizer,
                                       **self.lr_scheduler_param)
            if self.local_rank in [-1, 0]:
                self.logger.info(lr_scheduler_init_log)
            if self.resume:
                lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])

            if self.warmup:
                ret_lr_scheduler = GradualWarmupScheduler(
                    self.optimizer,
                    multiplier=self.warmup_param['multiplier'],
                    warmup_epochs=self.warmup_param['warmup_epochs'],
                    after_scheduler=lr_scheduler,)
                warmup_log = '===> Warmup for {} epochs with multiplier={}\n'\
                    ''.format(self.warmup_param["warmup_epochs"],
                              self.warmup_param['multiplier'],)
                if self.local_rank in [-1, 0]:
                    self.logger.info(warmup_log)
            else:
                ret_lr_scheduler = lr_scheduler
                if self.local_rank in [-1, 0]:
                    self.logger.info('\n')

            return ret_lr_scheduler
        except Exception as error:
            logging.info(f'LR scheduler initilize failed: {error} !')
            raise AttributeError(f'LR scheduler initial failed: {error} !')

    def init_model(self, network_name=None, network_param=None):
        if network_name is None:
            network_name = self.network_name
        if network_param is None:
            network_param = self.network_param

        model = build_network(network_name, **network_param)

        # Count the total amount of parameters with gradient.
        total_params = 0
        for x in filter(lambda p: p.requires_grad, model.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        total_params /= 10 ** 6

        pretrained = network_param['pretrained']
        if self.resume:
            model.load_state_dict(self.checkpoint['model'])
            model_init_log = '===> Resumed {} from "{}". Total'\
                'parameters: {:.2f}m'.format(
                    network_name, self.resume_fpath, total_params)

        elif pretrained:
            pretrained_fpath = network_param['pretrained_fpath']
            state_dict = torch.load(pretrained_fpath, map_location='cpu')
            if any('ResNet18', 'ResNet34', 'ResNet50', 'ResNet101') in\
               network_name:
                state_dict = {k: v for k, v in state_dict.items() if
                              'fc' not in k}

            model.load_state_dict(state_dict, strict=False)
            model_init_log = '===> Initialized pretrained {} from "{}". Total'\
                'parameters: {:.2f}m'.format(
                    network_name, pretrained_fpath, total_params)
        else:
            model_init_log = '===> Initialized {}. Total parameters:'\
                    ' {:.2f}m'.format(network_name, total_params)
        if self.local_rank in [-1, 0]:
            self.logger.info(model_init_log)

        model.cuda()

        return model

    def init_module(self, module_name=None, module_param=None):
        module = build_module(module_name, **module_param)
        module_init_log = '===> Initialized {}.'.format(module_name)
        if self.local_rank in [-1, 0]:
            self.logger.info(module_init_log)
        return module

    def init_loss(self, loss_name=None, **kwargs):
        if loss_name is None:
            loss_name = self.loss_name
        loss_param = self.loss_param if kwargs is None else kwargs
        loss = build_loss(loss_name, **loss_param)
        loss_init_log = f'===> Initialized {loss_name}'
        if self.local_rank in [-1, 0]:
            self.logger.info(loss_init_log)
        return loss

    def _reduce_loss(self, tensor):
        with torch.no_grad():
            dist.reduce(tensor, dst=0)
            if self.local_rank == 0:
                tensor /= self.world_size

    def resume_checkpoint(self, resume_fpath=None):
        if resume_fpath is None:
            resume_fpath = self.resume_fpath
        checkpoint = torch.load(resume_fpath, map_location='cpu')
        self.start_epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        mr = checkpoint['mr']
        ap = checkpoint['ap']
        resume_log = '\n===> Resume checkpoint from "{}".\nResume acc:{:.2%}'\
            ' mr:{:.2%} ap:{:.2%}'.format(resume_fpath, acc, mr, ap)
        return checkpoint, resume_log

    def save_checkpoint(self, epoch, acc=None, mr=None, ap=None):
        if epoch == self.total_epochs:
            checkpoint = {'model': self.model.state_dict()
                          if self.local_rank == -1 else
                          self.model.module.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'lr_scheduler': self.lr_scheduler.state_dict(),
                          'epoch': epoch,
                          'acc': acc,
                          'mr': mr,
                          'ap': ap}
            last_fpath = join(self.save_dir, 'last.pth.tar')
            torch.save(checkpoint, last_fpath)

    def init_logger(self, log_fpath):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Save log to file
        file_handler = logging.FileHandler(log_fpath)
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

    def train(self):
        pass

    def train_epoch(self, epoch):
        pass

    def evaluate(self, epoch):
        pass
