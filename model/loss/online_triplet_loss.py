"""Modified from https://github.com/NegatioN/OnlineMiningTripletLoss"""
import torch
import torch.nn.functional as F
from model.loss.builder import Losses


@Losses.register_module("OnlineHardTripletLoss")
class OnlineHardTripletLoss(torch.nn.Module):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to
    form a triplet.

    Args:
        targets: targets of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix.
                 If false, output the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    def __init__(self, margin=100, squared=False, **kwargs):

        super(OnlineHardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, targets):

        # Get the pairwise distance matrix  (N, N)
        # the diagonal element is almost 0.
        pairwise_dist = _pairwise_distances(embeddings, squared=self.squared)

        # For each anchor, get the hardest positive (N, N)
        # First, we need to get a mask for every valid positive (they should
        # have same label)
        mask_anchor_positive = _get_anchor_positive_triplet_mask(
            targets).float()

        # We put to 0 any element where (a, p) is not valid
        # (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist  # (N, N)

        # shape (N, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative
        # (they should have different targets)
        # shape (N, N)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(
            targets).float()

        # We add the maximum value in each row to the invalid negatives
        # (label(a) == label(n)) to make sure the following minimum work.
        # shape: (N, 1)
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = hardest_positive_dist - hardest_negative_dist\
            + self.margin
        triplet_loss = F.relu(triplet_loss)
        avg_triplet_loss = triplet_loss.mean()

        return avg_triplet_loss


@Losses.register_module("OnlineTripletLoss")
class OnlineTripletLoss(torch.nn.Module):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive
    ones.

    Args:
        targets: targets of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    def __init__(self, margin, squared=False, **kwargs):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, targets):
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=self.squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i,
        # positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape
        # (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist\
            + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(targets)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (
            num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of
    # `dot_product`. This also provides more numerical stability (the diagonal
    # of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape: (1, batch_size) - (batch_size, batch_size) + (batch_size, 1)
    #        = (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product +\
        square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative
    # so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0
        # (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances


def _get_triplet_mask(targets):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n)
    is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - targets[i] == targets[j] and targets[i] != targets[k]
    Args:
        targets: int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(targets.size(0), device=targets.device).bool()
    indices_not_equal = ~indices_equal  # (N, N) diag elem is False
    i_not_equal_j = indices_not_equal.unsqueeze(2)  # (N, N, 1)
    i_not_equal_k = indices_not_equal.unsqueeze(1)  # (N, 1, N)
    j_not_equal_k = indices_not_equal.unsqueeze(0)  # (1, N, N)

    # (N, N, N)
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = targets.unsqueeze(0) == targets.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)  # (N, N, 1)
    i_equal_k = label_equal.unsqueeze(1)  # (N, 1, N)

    valid_mask = ~i_equal_k & i_equal_j  # (N, N, N): i!=k, i=j

    return valid_mask & distinct_indices


def _get_anchor_positive_triplet_mask(targets):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and
    have same label.
    Args:
        targets: int32 `Tensor` with shape [batch_size]
    Returns:
        mask: bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(targets.size(0), device=targets.device).bool()
    indices_not_equal = ~indices_equal

    # Check if targets[i] == targets[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and
    # the 2nd (batch_size, 1)
    targets_equal = targets.unsqueeze(0) == targets.unsqueeze(1)

    return targets_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(targets):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct targets.
    Args:
        targets: int32 `Tensor` with shape [batch_size]
    Returns:
        mask: bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if targets[i] != targets[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and
    # the 2nd (batch_size, 1)

    return ~(targets.unsqueeze(0) == targets.unsqueeze(1))
