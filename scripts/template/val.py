"""TRAINING
"""
import os
import logging
import warnings
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import yaml
import json

from importlib import import_module
from common.network.builder import build_network
from common.dataset.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    return args


class Validater:
    def __init__(self, config=None):
        self.data_loader = None
        self.net = None

    def build_transform(self, config=None):
        transform_config = config['transform']
        script_path = transform_config['script_path']
        transform_name = transform_config['name']
        transform_param = transform_config['param']
        module = import_module(script_path)
        transform = getattr(module, transform_name)(**transform_param)
        return transform

    def build_dataset(self, config=None, transform=None):
        dataset_config = config['dataset']
        if dataset_config is None:
            raise ValueError("Dataset Config can't be None")
        dataset_name = dataset_config['name']
        dataset_param = dataset_config['param']
        if transform is not None:
            dataset_param['transform'] = transform
        dataset = build_dataset(dataset_name, **dataset_param)
        return dataset

    def build_model(self, config):
        network_config = config['network']
        network_name = network_config['name']
        network_param = network_config['param']
        model = build_network(
            network_name, config=network_config, **network_param)
        return model

    def validate(self, config=None):
        ##################################
        # Initialize saving directory
        ##################################
        experiment_config = config['experiments']
        save_dir = experiment_config.get('save_dir', None)
        fname2url_fpath = experiment_config.get('fname2url_fpath', None)
        log_prefix = experiment_config.get('log_prefix', '')
        log_period = experiment_config.get('log_period', 5)
        model_name = experiment_config.get('model_name', "resnet18")
        epochs = experiment_config.get('epochs', 50)

        network_param = config['network']["param"]
        test_model_path = network_param.get('test_model_path', None)
        test_model_fname = os.path.split(test_model_path)[1]
        test_epoch = 140  # test_model_fname.split('_')[2]

        ##################################
        # training param setting
        ##################################
        dataloader_config = config['dataloader']
        dataloader_param = dataloader_config['param']
        batch_size = dataloader_param.get("batch_size", 16)
        num_workers = dataloader_param.get("num_workers", 8)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ##################################
        # Logging setting
        ##################################
        log_name = log_prefix + str(test_epoch) + '.log'
        logging.basicConfig(
            filename=os.path.join(save_dir, log_name),
            filemode='a+',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)
        warnings.filterwarnings("ignore")

        ##################################
        # Load dataset
        ##################################
        dataset_param = config['dataset']['param']
        map_fpath = dataset_param.get('mapping_file', None)
        transform = self.build_transform(config)
        val_dataset = self.build_dataset(config=config, transform=transform)

        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False,)

        net = self.build_model(config)
        resume_log_line = f"Loaded checkpoint at epoch {test_epoch} from: {test_model_path}"
        print(resume_log_line)
        logging.info(resume_log_line)
        net.load_state_dict(torch.load(
            test_model_path, map_location='cpu')['state_dict'])

        ##################################
        # Use cuda
        ##################################
        net.cuda()
        net.eval()

        ##################################
        # Prepare maps
        ##################################
        label2ctg = {}
        with open(map_fpath, 'r') as f_map:
            for line in f_map.readlines():
                ctg, label = line.strip().split('\t')
                label2ctg[int(label)] = ctg

        with open(fname2url_fpath, 'r') as f_fname2url:
            fname2url = json.load(f_fname2url)


        ##################################
        # Start validating
        ##################################
        right_cnt = 0
        total_cnt = 0
        true_list, pred_list, prob_list = [], [], []
        misclassified_url_dict = defaultdict(dict)
        handle = open(
            save_dir + f"{log_prefix}{test_epoch}_pred_grocery_all.txt", "w")
        pbar = tqdm(total=len(val_loader))
        with torch.no_grad():
            for cnt, (batch_img, batch_label,
                      batch_img_fname) in enumerate(val_loader):
                batch_img = batch_img.cuda()

                batch_prob = net(batch_img, y=None)
                batch_prob = F.softmax(batch_prob, dim=1)

                batch_prob = batch_prob.cpu().numpy()
                batch_pred = np.argmax(batch_prob, axis=1)

                true_list.extend(batch_label.numpy().tolist())
                pred_list.extend(batch_pred.tolist())
                prob_list.extend(batch_prob.tolist())
                for idx, (pred_prob, pred_label, gt_label, img_fname) in enumerate(zip(batch_prob.tolist(), batch_pred.tolist(), batch_label.numpy().tolist(), batch_img_fname)):
                    cls_score = pred_prob[pred_label]
                    gt_ctg = label2ctg[gt_label]
                    pred_ctg = label2ctg[pred_label]
                    img_url = fname2url[img_fname]
                    handle.write("{}\t{}\t{:.2f}\t{}\n".format(
                        gt_ctg, pred_ctg, cls_score, img_url))
                    total_cnt += 1
                    if gt_label != pred_label:
                        misclassified_url_dict[f"{gt_ctg}2{pred_ctg}"][img_url] = cls_score
                    else:
                        right_cnt += 1

                pbar.update()
        pbar.close()
        handle.close()

        with open(save_dir + f'{log_prefix}{test_epoch}_misclassified_urls.json', 'w') as f:
            json.dump(misclassified_url_dict, f, ensure_ascii=False, indent=4)

        print(f"right: {right_cnt}, total: {total_cnt}")
        logging.info(f"right: {right_cnt}, total: {total_cnt}")
        acc = metrics.accuracy_score(true_list, pred_list)
        mean_recall = metrics.balanced_accuracy_score(true_list, pred_list)
        #confusion_matrix = metrics.confusion_matrix(true_list, pred_list)

        print("Acc: {:>4.2%}\tMean Recall: {:>4.2%}".format(acc, mean_recall))
        logging.info(
            "Acc: {:>4.2%}\tMean Recall: {:>4.2%}".format(acc, mean_recall))

        # save true_list and pred_list
        np.save(
            save_dir + f"{log_prefix}{test_epoch}_true_list.npy", np.array(true_list))
        np.save(
            save_dir + f"{log_prefix}{test_epoch}_pred_list.npy", np.array(pred_list))
        np.save(
            save_dir + f"{log_prefix}{test_epoch}_prob_list.npy", np.array(prob_list))


def main(args):
    warnings.filterwarnings('ignore')
    handle = open(args.config_file, "r")
    config = yaml.load(handle, Loader=yaml.FullLoader)
    validater = Validater(config=config)
    validater.validate(config=config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
