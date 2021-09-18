"""Base Trainer"""
import os
import os.path.join as join
import shutil
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
# Distribute Package
from torch import distributed as dist
# Custom package
from model.loss.builder import build_loss
from model.network.builder import build_network
from data_loader.dataset.builder import build_dataset
from data_loader.sampler.builder import build_sampler


class BaseTrainer:
    def __init__(self, local_rank=-1, config=None):
        """ Base trainer for all experiments.  """

        ##################################
        # Device setting
        ##################################
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True

        self.local_rank = local_rank
        if self.local_rank != -1:
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(self.local_rank)
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        ##################################
        # Experiment setting
        ##################################
        self.experiment_config = config['experiment']
        self.exp_name = self.experiment_config['name']
        self.start_epoch = self.experiment_config['start_epoch']
        self.total_epochs = self.experiment_config['total_epochs']
        if self.local_rank in [-1, 0]:
            self.save_root = self.experiment_config['save_root']
            self.save_dir = join(self.save_root, self.exp_name)
            self.tb_root = self.experiment_config['tb_root']
            self.tb_dir = join(self.tb_root, self.exp_name)
            self.log_fname = self.experiment_config['log_fname']
            self.log_fpath = join(self.save_dir, self.log_fname)
            self.save_period = self.experiment_config['save_period']
            self.eval_period = self.experiment_config['eval_period']

            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)

            self.writer = SummaryWriter(self.tb_dir)
            logging.basicConfig(
                filename=self.log_fpath,
                filemode='a+',
                format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]:\
                        %(message)s',
                level=logging.INFO
            )

        ##################################
        # Dataset setting
        ##################################
        self.train_transform_config = config['train_transform']
        self.trainset_config = config['train_dataset']
        self.eval_transform_config = config['eval_transform']
        self.evalset_config = config['eval_dataset']

        ##################################
        # Dataloader setting
        ##################################
        self.trainloader_config = config['trainloader']
        self.trainloader_name = self.trainloader_config["name"]
        self.trainloader_param = self.trainloader_config['param']
        self.train_sampler_name = self.trainloader_param['sampler']
        self.train_batch_size = self.trainloader_param["batch_size"]
        self.train_num_workers = self.trainloader_param["num_workers"]

        self.evalloader_config = config['evalloader']
        self.evalloader_name = self.evalloader_config['name']
        self.evalloader_param = self.evalloader_config['param']
        self.eval_batch_size = self.evalloader_param['batch_size']
        self.eval_num_workers = self.evalloader_param['num_workers']

        ##################################
        # Network setting
        ##################################
        self.network_config = config['network']
        self.network_name = self.network_config['name']
        self.network_param = self.network_config['param']
        self.resume_training = self.network_param['resume_training']
        self.resume_fpath = self.network_param['resume_fpath']

        ##################################
        # Loss setting
        ##################################
        self.loss_config = config['loss']
        self.loss_name = self.loss_config['name']
        self.loss_param = self.loss_config['param']

        ##################################
        # Optimizer setting
        ##################################
        self.optimizer_config = config['optimizer']
        self.optimizer_name = self.optimizer_config['name']
        self.optimizer_param = self.optimizer_param['param']

        ##################################
        # LR scheduler setting
        ##################################
        self.lr_scheduler_config = config['lr_scheduler']
        self.lr_scheduler_name = self.lr_scheduler_config['name']
        self.lr_scheduler_param = self.lr_scheduler_config['param']

    def init_transform(self, transform_config=None):
        script_path = transform_config['script_path']
        transform_name = transform_config['name']
        transform_param = transform_config['param']
        module = import_module(script_path)
        transform_init_log = f"Initialized {transform_name} from\
                {script_path} with {transform_param}."
        logging.info(transform_init_log)
        print(transform_init_log)
        transform = getattr(module, transform_name)(**transform_param)
        return transform

    def init_sampler(self, dataloader_config=None, dataset=None):
        sampler_param = dataloader_config['param']
        sampler_name = sampler_param['sampler']
        sampler_param['dataset'] = dataset
        self.log_init_param(sampler_name, sampler_param)
        sampler = build_sampler(sampler_name, **sampler_param)
        return sampler

    def init_dataset(self, dataset_config=None, transform=None):
        dataset_name = dataset_config['name']
        dataset_param = dataset_config['param']
        dataset_param['transform'] = transform
        self.log_init_param(dataset_name, dataset_param)
        dataset = build_dataset(dataset_name, **dataset_param)
        return dataset

    def init_optimizer(self):
        params = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in ['bias', 'bn'])],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in ['bias', 'bn'])],
                'weight_decay': 0.0
            }
        ]
        try:
            optimizer = getattr(torch.optim, self.optimizer_name)(
                params, **self.optimizer_param)
            self.log_init_param(self.optimizer_name, self.optimizer_param)
            return optimizer
        except Exception as error:
            logging.info(f"optimizer initialize failed: {error} !")
            raise AttributeError(f"optimizer initialize failed: {error} !")

    def init_lr_scheduler(self):
        try:
            lr_scheduler = getattr(torch.optim.lr_scheduler,
                                   self.lr_scheduler_name)(
                                       self.optimizer,
                                       **self.lr_scheduler_param)
            self.log_init_param(self.lr_scheduler_name,
                                self.lr_scheduler_param)
            return lr_scheduler
        except Exception as error:
            logging.info(f"LR scheduler initilize failed: {error} !")
            raise AttributeError(f"LR scheduler initial failed: {error} !")

    def init_model(self):
        model = build_network(self.network_name, config=self.network_config,
                              **self.network_param)
        use_pretrained = self.network_param['use_pretrained']
        if use_pretrained:
            pretrained_fpath = self.network_param['pretrained_fpath']
            state_dict = torch.load(pretrained_fpath, map_location='cpu')
            model.load_state_dict(state_dict)
        self.log_init_param(self.network_name, self.network_param)
        return model

    def init_loss(self):
        loss = build_loss(self.loss_name, **self.loss_param)
        self.log_init_param(self.loss_name, self.loss_param)
        return loss

    def _reduce_loss(self, tensor):
        with torch.no_grad():
            dist.reduce(tensor, dst=0)
            if self.local_rank == 0:
                tensor /= self.world_size

    def train(self):
        pass

    def train_epoch(self, epoch):
        pass

    def evaluate(self, epoch):
        pass

    def resume_checkpoint(self):
        checkpoint = torch.load(self.resume_fpath, map_location='cpu')
        model_state_dict = checkpoint['model']
        optimizer_state_dict = checkpoint['optimizer']
        lr_scheduler_state_dict = checkpoint['lr_scheduler']
        self.start_epoch = checkpoint['start_epoch']
        acc = checkpoint['acc']
        mr = checkpoint['mr']
        ap = checkpoint['ap']
        log_str = 'Resume checkpoint from {}.\nResume acc:{:.2%} mr:{:.2%}\
                ap:{:.2%}'.format(self.resume_fpath, acc, mr, ap)
        logging.info(log_str)
        print(log_str)

        return model_state_dict, optimizer_state_dict, lr_scheduler_state_dict

    def save_checkpoint(self, epoch, save_fname, is_best,
                        acc=None, mr=None, ap=None):
        checkpoint = {
            'model': self.model.state_dict() if self.local_rank != -1
            else self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'mr': mr,
            'ap': ap,
        }
        save_fpath = join(self.save_dir, save_fname)
        torch.save(checkpoint, save_fpath)
        if is_best:
            best_fpath = join(self.save_dir,
                              f'{self.network_name}_best.pth.tar')
            shutil.copyfile(save_fpath, best_fpath)

    def log_init_param(self, name, param):
        init_log = f"Initialized {name} with {param}."
        logging.info(init_log)
        print(init_log)
