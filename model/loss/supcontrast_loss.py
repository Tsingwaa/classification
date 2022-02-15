"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from model.loss.builder import Losses


@Losses.register_module("SupContrastLoss")
class SupContrastLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self,
                 temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07,
                 **kwargs):
        super(SupContrastLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [B, n_views, ...].
                e.g. n_views=2 [B, 2, embedding_dim]
            labels: ground truth of shape [B,].
            mask: contrastive mask of shape [B, B], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [B, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:  # usually: [B, 2, dim]
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # Unsupervised
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:  # Only given labels
            # build mask according to labels
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # [B, B]
        else:  # Only given mask
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # torch.unbind(features, dim=1): remove 1-dim
        # return a tuple within 2 elements.
        # [B, 2, embed_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # unbind ==> (tensor(B, embed_dim), tensor(B, embed_dim))
        # cat ==> tensor(2B, embed_dim)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # select the first
            anchor_count = 1
        elif self.contrast_mode == 'all':  # Supervised. class-level contrast
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # Unsupervised: [B, embed_dim] x [embed_dim, 2B] = [B, 2B]
        # Supervised[2B, embed_dim] x [embed_dim, 2B] = [2B, 2B]
        # torch.div: element-divide

        # for numerical stability: minus maximum of each column
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # [B*B, 2B*B]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # B * B, 2 * B * B
            1,  # replace dim
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,  # replace value
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
