"""Utils
Created: Nov 11,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import math
import os
# import random
from os.path import exists, join

import numpy as np
import torch
# import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

__all__ = ['count_model_params', 'label2onehot', 'AverageMeter']


def count_model_params(net):
    # Compute the total amount of parameters with gradient.
    total_params = 0.

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)

    return total_params


def label2onehot(targets, num_classes):
    """Transform label to one-hot vector."""

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    init_zeros = torch.zeros(len(targets), num_classes)

    return init_zeros.scatter_(1, targets.view(-1, 1), 1)


class TopKAccuracyMetric:

    def __init__(self, topk=(1, )):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cm_with_labels(targets, preds, classes):
    """
    Args:
        targets: number list, 1 * n
        preds: number list, 1 * n
        classes: class name list, 1 * C
    Return:
        a DataFram of confusion matrix with labels.
    Additional:
        cm_df.to_string() could be save in .log file.
        cm_df.to_csv() could be saved in .csv file.
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, preds)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    return cm_df


def rotation(inputs):
    batch = inputs.shape[0]
    target = torch.Tensor(np.random.permutation([0, 1, 2, 3] *
                                                (int(batch / 4) + 1)),
                          device=inputs.device)[:batch]

    target = target.long()
    image = torch.zeros_like(inputs)
    image.copy_(inputs)

    for i in range(batch):
        image[i, :, :, :] = torch.rot90(inputs[i, :, :, :], target[i], [1, 2])

    return image, target


def get_weight_scheduler(cur_epoch, total_epoch, weight_scheduler, **kwargs):
    """Return a decaying weight according to epoch"""

    if weight_scheduler == 'parabolic_incr':  # lower convex 0->1
        weight = (cur_epoch / total_epoch)**2.
    elif weight_scheduler == 'parabolic_decay':  # upper convex 1->0
        weight = 1. - (cur_epoch / total_epoch)**2.
    elif weight_scheduler == 'cosine_decay':  # upper then lower convex 1->0
        weight = math.cos((cur_epoch / total_epoch) * math.pi / 2.)
    elif weight_scheduler == 'linear_decay':  # linear 1->0
        weight = 1. - cur_epoch / total_epoch
    elif weight_scheduler == 'step_decay':  # step
        weight = 1. if cur_epoch <= kwargs['step_epoch'] else 0
    elif weight_scheduler == 'beta_distribution':
        weight = np.random.beta(kwargs['alpha'], kwargs['alpha'])
    else:  # Fixed
        weight = kwargs['weight']

    return weight


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax


def plot_features(features, labels, save_dir, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (N, num_features).
        labels: (N).
    """

    colors = ['C' + str(i) for i in range(num_classes)]

    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(list(range(num_classes)), loc='upper right')
    dirname = join(save_dir, prefix)

    if not exists(dirname):
        os.mkdir(dirname)
    save_name = join(dirname, 'epoch_' + str(epoch + 1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
