"""TRAINING """
import argparse
import os
import pickle
# import shutil
import warnings
from os.path import expanduser, join

# import h5py
import torch
# import torch.nn.functional as F
import yaml
from base.base_trainer import BaseTrainer
from prefetch_generator import BackgroundGenerator
# from pudb import set_trace
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Extractor(BaseTrainer):

    def __init__(self, local_rank, config, seed):
        """Extractor to extract feature"""

        self.local_rank = local_rank
        self.seed = seed

        self.exp_config = config["experiment"]
        self.exp_name = self.exp_config["name"]
        self.finetune_config = config["finetune"]
        self.finetune_name = self.finetune_config["name"]

        self.exp_root = expanduser("~/Experiments")
        self.total_epochs = self.finetune_config["total_epochs"]

        self._set_configs(config)

        self.resume = True

        if "/" in self.exp_config["resume_fpath"]:
            self.resume_fpath = self.exp_config["resume_fpath"]
        else:
            self.resume_fpath = join(
                self.exp_root, self.exp_name,
                f"seed{self.seed}_{self.finetune_name}"
                f"_{self.exp_config['resume_fpath']}")

        self.checkpoint, resume_log = self.resume_checkpoint(self.resume_fpath)

        if self.local_rank in [-1, 0]:
            self.eval_period = self.exp_config["eval_period"]  # default: 1
            self.save_period = self.exp_config["save_period"]  # default: 10
            self.exp_dir = join(self.exp_root, self.exp_name)
            os.makedirs(self.exp_dir, exist_ok=True)

            # Set logger to save .log file and output to screen.
            self.log_fpath = join(self.exp_dir,
                                  f"seed{self.seed}_{self.finetune_name}.log")
            self.logger = self.init_logger(self.log_fpath)
            exp_init_log = f"\n****************************************"\
                f"****************************************************"\
                f"\nExperiment: Finetune {self.exp_name}\n"\
                f"Extract features:\n"\
                f"Save dir: {self.exp_dir}\n"\
                f"Save peroid: {self.save_period}\n"\
                f"Resume Training: {self.resume}\n"\
                f"Distributed Training: "\
                f"{True if self.local_rank != -1 else False}\n"\
                f"**********************************************"\
                f"**********************************************\n"
            self.log(exp_init_log)
            self.log(resume_log)

        self.unfreeze_keys = []

        ft_network_config = self.finetune_config.pop("network", None)

        if ft_network_config is not None and ft_network_config["name"]:
            self.network_name = ft_network_config["name"]
            self.network_params = ft_network_config["param"]

        self.trainloader_params = self.finetune_config["trainloader"]
        self.train_sampler_name = self.trainloader_params.pop("sampler", None)
        self.train_batchsize = self.trainloader_params["batch_size"]
        self.train_workers = self.trainloader_params["num_workers"]

    def extract(self):
        #######################################################################
        # Initialize Dataset and Dataloader
        #######################################################################
        val_transform = self.init_transform(self.val_transform_name,
                                            **self.val_transform_params)
        trainset = self.init_dataset(self.trainset_name,
                                     transform=val_transform,
                                     **self.trainset_params)

        # self.reform_mean = torch.tensor(trainset.mean).view(1, 3, 1, 1)
        # self.reform_std = torch.tensor(trainset.std).view(1, 3, 1, 1)

        # train_sampler = self.init_sampler(self.train_sampler_name,
        #                                   dataset=trainset,
        #                                   **self.trainloader_params)
        self.trainloader = DataLoaderX(trainset,
                                       batch_size=self.train_batchsize,
                                       shuffle=False,
                                       num_workers=self.train_workers,
                                       pin_memory=True,
                                       drop_last=False,
                                       sampler=None)

        valset = self.init_dataset(self.valset_name,
                                   transform=val_transform,
                                   **self.valset_params)
        # val_sampler = self.init_sampler(self.val_sampler_name,
        #                                 dataset=valset,
        #                                 **self.valloader_params)
        self.valloader = DataLoaderX(valset,
                                     batch_size=self.val_batchsize,
                                     shuffle=False,
                                     num_workers=self.val_workers,
                                     pin_memory=True,
                                     drop_last=False,
                                     sampler=None)

        #######################################################################
        # Initialize Network
        #######################################################################
        self.model = self.init_model(self.network_name,
                                     resume=True,
                                     checkpoint=self.checkpoint,
                                     num_classes=trainset.num_classes,
                                     except_keys=[],
                                     **self.network_params)
        self.freeze_model(self.model, unfreeze_keys=[])

        #######################################################################
        # Start evaluating
        #######################################################################
        self.extract_one_phase(self.trainloader, phase='train')
        self.extract_one_phase(self.valloader, phase='test')

    def extract_one_phase(self, dataloader, phase):
        pbar = tqdm(total=len(dataloader), desc=f'Extracting {phase} features')

        # all_imgs = []
        all_feats = []
        all_labels = []
        # all_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_imgs, batch_labels) in enumerate(dataloader):
                # collect labels
                # batch_ori_imgs = batch_imgs * self.reform_std +\
                #     self.reform_mean
                # batch_resized_imgs = F.interpolate(
                #     batch_ori_imgs,
                #     size=(112, 112),
                # )
                # all_imgs.append(batch_resized_imgs.detach())
                all_labels.append(batch_labels.detach())
                # batch_labels:torch.size([B])

                batch_imgs = batch_imgs.cuda(non_blocking=True)
                batch_feats = self.model(batch_imgs, out_type="fc1")
                all_feats.append(batch_feats.detach().cpu())

                # batch_prob = self.model.fc(batch_imgs, out_type="fc12")
                # batch_preds = batch_prob.max(1)[1]
                # all_preds.append(batch_preds.detach().cpu())
                pbar.update()
        pbar.close()

        # set_trace()
        # all_imgs = torch.vstack(all_imgs)
        all_feats = torch.vstack(all_feats).numpy()
        all_labels = torch.hstack(all_labels).numpy()
        # all_preds = torch.hstack(all_preds).numpy()

        # self.writer.add_embedding(
        #     mat=all_feats,
        #     metadata=all_labels,
        #     label_img=all_imgs,
        #     tag=f'{phase}_GT',
        #     global_step=self.total_epochs,
        # )
        # self.writer.add_embedding(
        #     mat=all_feats,
        #     metadata=all_preds,
        #     label_img=all_imgs,
        #     tag=f'{phase}_Pred',
        #     global_step=self.total_epochs,
        # )
        # self.writer.close()

        feats_labels_fpath = join(
            self.exp_root, self.exp_name,
            f'{self.finetune_name}_{phase}_features_labels.pickle')
        # save feature and labels

        if os.path.exists(feats_labels_fpath):
            os.remove(feats_labels_fpath)

        feats_labels = {
            "features": all_feats,
            "labels": all_labels,
        }
        with open(feats_labels_fpath, "wb") as f:  # dump <--> load
            pickle.dump(feats_labels, f)

        print(f'Features-labels file is saved at "{feats_labels_fpath}"\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='Local Rank for\
                        distributed training. if single-GPU, default: -1')
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    return args


def main(args):
    warnings.filterwarnings('ignore')
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    extractor = Extractor(local_rank=args.local_rank,
                          config=config,
                          seed=args.seed)
    extractor.extract()


if __name__ == '__main__':
    args = parse_args()
    main(args)
