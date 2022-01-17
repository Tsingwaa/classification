import math

import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = [
    'eu_dist', 'cos_sim', 'get_preds_by_eudist', 'get_preds_by_cossim',
    'ExpStat'
]


def eu_dist(A, B, sqrt=False, device='cuda'):
    """Euclidean Distance
    A: m*d
    B: n*d

    """
    m, n = A.shape[0], B.shape[0]
    dist_mat = torch.square(A).sum(dim=1, keepdim=True).expand(m, n) +\
        torch.square(B).sum(dim=1, keepdim=True).expand(n, m).t()

    dist_mat.addmm_(A, B.t(), beta=1, alpha=-2)  # m*n distance matrix

    if sqrt:  # get standard euclidean distance.
        dist_mat = torch.sqrt(dist_mat)

    if device == 'cpu':
        dist_mat = dist_mat.cpu()
    else:
        dist_mat = dist_mat.cuda()

    return dist_mat


def get_preds_by_eudist(querys, keys):
    """search which keys is nearest for each query.
    querys: batch_size * d
    keys: num_classes * d
    """
    dist_mat = eu_dist(querys, keys, sqrt=False)  # batch_size * num_classes
    preds = dist_mat.min(1)[1]  # predicted classes: batch_size * 1

    return preds


def cos_sim(A, B, device=None):
    """Cosine Similarity
    A: m*d
    B: n*d
    """
    epsilon = 1e-7

    dist_mat = A.mm(B.t())
    A_norm = torch.norm(A, dim=1, keepdim=True)  # keepdim: mx1
    B_norm = torch.norm(B, dim=1, keepdim=True)  # keepdim: nx1
    AB_norm_mm = A_norm.mm(B_norm.t())
    dist_mat = dist_mat / AB_norm_mm
    dist_mat = torch.clamp(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = torch.arccos(dist_mat)

    if device == 'cpu':
        dist_mat = dist_mat.cpu()
    elif device == 'cuda':
        dist_mat = dist_mat.cuda()

    return dist_mat


def get_preds_by_cossim(querys, keys):
    """search which keys is nearest for each query.
    querys: batch_size * d
    keys: num_classes * d
    """
    sim_mat = cos_sim(querys, keys, device='cpu')
    preds = sim_mat.max(1)[1]

    return preds


class ExpStat(object):

    def __init__(self, num_classes, head_class_idx, med_class_idx, tail_class_idx):
        self.head_class_idx = head_class_idx
        self.med_class_idx = med_class_idx
        self.tail_class_idx = tail_class_idx
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._cm = torch.zeros((self.num_classes, self.num_classes),
                               dtype=torch.long).cuda()

    def update(self, labels, preds):
        """labels and preds(batch_size*1) are both computed from iteration"""
        labels, preds = labels.long(), preds.long()
        batch_size = preds.shape[0]

        for i in range(batch_size):
            self._cm[labels[i], preds[i]] += 1  # label[i] --> output[i]

    @property
    def cm(self):
        return self._cm

    @property
    def recalls(self):
        cm = self._cm.cpu().numpy()
        recalls = np.diag(cm) / np.sum(cm, axis=1)
        recalls[np.isnan(recalls)] = 0

        return recalls

    @property
    def mr(self):
        return np.mean(self.recalls)

    @property
    def group_mr(self):
        # head_cls_num = math.floor(self.num_classes / 3)
        # tail_cls_num = head_cls_num
        # head_mr = np.mean(self.recalls[:head_cls_num])
        
        # mid_mr = np.mean(self.recalls[head_cls_num:self.num_classes -
        #                               tail_cls_num])
        # tail_mr = np.mean(self.recalls[tail_cls_num:])
        head_mr = np.mean(self.recalls[self.head_class_idx[0]: self.head_class_idx[1]])
        mid_mr = np.mean(self.recalls[self.med_class_idx[0]: self.med_class_idx[1]])
        tail_mr = np.mean(self.recalls[self.tail_class_idx[0]: self.tail_class_idx[1]])
        return [
            np.around(head_mr, decimals=4),
            np.around(mid_mr, decimals=4),
            np.around(tail_mr, decimals=4)
        ]

    @property
    def precisions(self):
        cm = self._cm.cpu().numpy()
        precisions = np.diag(cm) / np.sum(cm, axis=0)
        precisions[np.isnan(precisions)] = 0

        return precisions

    @property
    def mp(self):
        return np.mean(self.precisions)

    @property
    def accuracy(self):
        cm = self._cm.cpu().numpy()

        return np.sum(np.diag(cm)) / cm.sum()

    def get_preds_by_eudist(self, querys, keys):
        """search which keys is nearest for each query.
        querys: batch_size * d
        keys: num_cls * d
        """
        dist_mat = eu_dist(querys, keys, sqrt=False)  # batch_size * num_cls

        return dist_mat.max(1)[1]  # predicted classes: batch_size * 1

    def get_preds_by_cossim(self, querys, keys):
        """search which keys is nearest for each query.
        querys: batch_size * d
        keys: num_classes * d
        """
        sim_mat = cos_sim(querys, keys)

        return sim_mat.max(1)[1]

    def plot_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues):

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self._cm.cpu().numpy()
        cm = cm.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=np.arange(self.num_classes),
            yticklabels=np.arange(self.num_classes),
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

        return fig
