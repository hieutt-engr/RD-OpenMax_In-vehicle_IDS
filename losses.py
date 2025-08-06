from __future__ import print_function
import torch

"""
Author: Aiyang Han (aiyangh@nuaa.edu.cn)
Date: May 24th, 2022
"""

import torch.nn as nn
import torch.nn.functional as F


class UniConLoss_Standard(nn.Module):
    """Universum-inspired Supervised Contrastive Learning: https://arxiv.org/abs/2204.10695"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(UniConLoss_Standard, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, universum, labels):
        """
        We include universum data into the calculation of InfoNCE.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            universum: universum data of shape [bsz*n_views, ...]
        Returns:
            A loss scalar.
        """
        # Get device from `features`
        device = features.device

        # Check and synchronize device for tensors
        labels = labels.to(device)
        universum = universum.to(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # Synchronize device for `mask`
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # shape of [bsz*n_views, feature_dimension]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # only show one view, shape of [bsz, feature_dimension]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # show all the views, shape of [bsz*n_views, feature_dimension]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, universum.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # find the biggest
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # make the size suited for similarity matrix

        # mask-out self-contrast cases, make value on the diagonal False
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class UniConLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.1, 
                 gamma=2.0, alpha=0.25):
        super(UniConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, features, universum, labels):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, universum.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute probability p_ij
        prob = torch.exp(log_prob)

        # compute focal weight
        focal_weight = self.alpha * (1 - prob) ** self.gamma

        # compute class weights dynamically based on batch
        class_counts = torch.bincount(labels.view(-1))
        class_weights = torch.zeros_like(class_counts, dtype=torch.float, device=device)
        for c in range(len(class_counts)):
            if class_counts[c] > 0:
                class_weights[c] = batch_size / (len(class_counts) * class_counts[c])
        # Apply class weights to labels
        weights = class_weights[labels.view(-1)]

        if anchor_count > 1:
            weights = weights.repeat(anchor_count)

        # compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = ((mask * focal_weight) * log_prob).sum(1) / mask.sum(1)

        # Apply dynamic weights
        mean_log_prob_pos = mean_log_prob_pos * weights

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss(nn.Module):
    """This part is from the pytorch implementation of SupCon.
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', epsilon=1e-6):
        """
        Args:
            alpha (float or Tensor): Weight for each class. If a number, apply evenly to all classes.
                                     If a Tensor, the size must match the number of classes.
            gamma (float): Weight adjustment parameter for easy and hard samples.
            reduction (str): Loss reduction mode, supports 'mean', 'sum', and 'none'.
            epsilon (float): Avoid log(0) or division by 0.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError("alpha must be float or Tensor")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Compute cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t: probability of the true class

        # Handle alpha (class weights)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Only take weights for classes appearing in the batch
                valid_classes = torch.unique(targets)
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha  # Single alpha value
        else:
            alpha_t = 1.0  # No weighting

        # Compute Focal Loss
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss

        # Avoid NaN by adding epsilon
        focal_loss = focal_loss.clamp(min=self.epsilon)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, device='cpu'):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]  # Select prototypes for each class
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss


def pairwise_loss(features, labels, margin=1.0):
    """
    Pairwise loss with margin to pull same-class together and push different-class apart
    """
    batch_size = features.size(0)
    if batch_size % 2 != 0:
        raise ValueError("Batch size must be even")

    left, right = torch.chunk(features, 2, dim=0)
    label_left, label_right = torch.chunk(labels, 2, dim=0)

    same_class = (label_left == label_right).float()
    diff_class = 1 - same_class

    # Euclidean distance
    distance = F.pairwise_distance(left, right)

    # Contrastive loss-style: pull same-class close, push different-class apart
    loss_same = same_class * distance.pow(2)
    loss_diff = diff_class * F.relu(margin - distance).pow(2)

    return (loss_same + loss_diff).mean()

#New 
### Loss này đang hoạt động tốt với 3 channel
# def fdef_loss(features: torch.Tensor, labels: torch.Tensor, temperature=0.1, margin=0.3):
#     """
#     Improved F-DEF Loss: cosine similarity + hard negative mining + margin
#     """
#     features = F.normalize(features, dim=1)
#     sim_matrix = torch.matmul(features, features.T) / temperature

#     labels = labels.view(-1, 1)
#     mask = torch.eq(labels, labels.T).float()
#     self_mask = torch.eye(labels.size(0), device=features.device)

#     # Remove self-comparison
#     pos_mask = mask * (1 - self_mask)
#     neg_mask = (1 - mask) * (1 - self_mask)

#     # --- Positive Loss ---
#     pos_sim = sim_matrix * pos_mask
#     pos_loss = F.relu(1.0 - pos_sim).sum() / (pos_mask.sum() + 1e-6)

#     # --- Negative Loss with hard mining ---
#     neg_sim = sim_matrix * neg_mask
#     hard_neg = (neg_sim > margin).float() * neg_sim
#     neg_loss = (hard_neg - margin).sum() / (neg_mask.sum() + 1e-6)

#     return pos_loss + neg_loss

## Cải tiện 1 chút: thêm logsumexp để tránh chia nhỏ trị số:
def fdef_loss(features, labels, temperature=0.1, margin=0.3):
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    self_mask = torch.eye(labels.size(0), device=features.device)

    pos_mask = mask * (1 - self_mask)
    neg_mask = (1 - mask) * (1 - self_mask)

    # Positive
    pos_sim = sim_matrix * pos_mask
    pos_loss = F.relu(1.0 - pos_sim).sum() / (pos_mask.sum() + 1e-6)

    # Negative: stable version
    neg_sim = sim_matrix * neg_mask
    hard_neg = (neg_sim > margin).float() * neg_sim
    neg_loss = F.relu(hard_neg - margin).sum() / (neg_mask.sum() + 1e-6)

    return pos_loss + neg_loss


# def fdef_loss(features: torch.Tensor, labels: torch.Tensor, temperature=0.1, margin=0.3):
#     """
#     F-DEF loss implementation with cosine similarity, positive and hard negative pairs.
    
#     Args:
#         features: Tensor of shape [2*B, D], stacked from two views (after encoder + projection)
#         labels: Tensor of shape [2*B]
#         temperature: Temperature scaling for cosine similarity
#         margin: Margin threshold for selecting hard negatives

#     Returns:
#         Scalar loss (positive + negative part)
#     """
#     device = features.device
#     features = F.normalize(features, dim=1)  # L2 normalization
#     sim_matrix = torch.matmul(features, features.T) / temperature  # Cosine similarity
#     sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)

#     labels = labels.view(-1, 1)
#     mask = torch.eq(labels, labels.T).float().to(device)

#     self_mask = torch.eye(sim_matrix.size(0), device=device)
#     pos_mask = mask * (1 - self_mask)  # Remove self-pair
#     neg_mask = (1 - mask) * (1 - self_mask)

#     # --- Positive pairs ---
#     pos_sim = sim_matrix[pos_mask.bool()]
#     pos_loss = F.relu(1.0 - pos_sim).mean()

#     # --- Negative pairs with margin (hard negatives only) ---
#     neg_sim = sim_matrix[neg_mask.bool()]
#     hard_neg_sim = neg_sim[neg_sim > margin]
#     if hard_neg_sim.numel() > 0:
#         neg_loss = (hard_neg_sim - margin).mean()
#     else:
#         neg_loss = torch.tensor(0.0, device=device)

#     return pos_loss + neg_loss


def cdef_loss(logits: torch.Tensor, labels: torch.Tensor, margin=0.2):
    """
    Improved C-DEF loss: stronger separation between logits and incorrect classes
    """
    logits = F.normalize(logits, dim=1)
    B, C = logits.shape
    device = logits.device

    labels_one_hot = torch.zeros(B, C, device=device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    labels_one_hot = F.normalize(labels_one_hot, dim=1)

    # --- Positive alignment ---
    pos_sim = torch.sum(logits * labels_one_hot, dim=1)
    pos_loss = F.relu(1.0 - pos_sim).mean()

    # --- Negative misalignment ---
    # Hard negatives: all non-true class entries in logits
    logits_neg = logits.clone()
    logits_neg[labels.unsqueeze(1).expand(-1, C).eq(torch.arange(C, device=device))] = 0  # zero-out true class

    neg_sim = torch.sum(logits_neg * labels_one_hot, dim=1)
    neg_loss = F.relu(neg_sim - margin).mean()

    return pos_loss + neg_loss

def opsupcon_loss(features, labels, prototypes, temperature=0.1, margin=0.4, lambda_pe=0.1, lambda_oh=0.5):
    """
    OOD-aware SupCon loss = SupCon(ID) + Tightness(ID) + Push-away(OOD)
    
    Args:
        features: [B, 2, D] (2 views per sample)
        labels:   [B], where label = -1 for OE samples
        prototypes: [C, D] class-wise normalized prototype
    Returns:
        loss_total
    """
    device = features.device
    B = features.shape[0]
    features = F.normalize(features, dim=2)  # [B, 2, D]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [2B, D]
    anchor_feature = contrast_feature
    anchor_count = 2
    logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)  # [2B, 2B]

    labels = labels.contiguous().view(-1)
    labels = torch.cat([labels, labels], dim=0)  # [2B]

    # Build mask only for ID samples
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)  # [2B, 2B]

    # Remove OE samples from positive pairs
    invalid = (labels == -1)
    mask[invalid, :] = 0
    mask[:, invalid] = 0

    # Compute SupCon Loss
    logits_mask = torch.ones_like(mask).fill_diagonal_(0)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    supcon_loss = -mean_log_prob_pos.mean()

    # === Tightness Loss (PE) ===
    if prototypes is not None:
        with torch.no_grad():
            norm_proto = F.normalize(prototypes, dim=1)
        # Chỉ lấy các ID samples
        idx_id = (labels != -1)
        id_features = anchor_feature[idx_id]  # [M, D]
        id_labels = labels[idx_id]            # [M]

        proto_targets = norm_proto[id_labels.long()]  # [M, D]
        loss_pe = (1 - F.cosine_similarity(id_features, proto_targets)).mean()
    else:
        loss_pe = torch.tensor(0.0, device=device)

    # === OE Push Loss (OH) ===
    with torch.no_grad():
        norm_proto = F.normalize(prototypes, dim=1)

    idx_oe = (labels == -1)
    if idx_oe.any():
        features_oe = anchor_feature[idx_oe]  # [K, D]
        sim_oe = torch.matmul(F.normalize(features_oe, dim=1), norm_proto.T)  # [K, C]
        max_sim = sim_oe.max(dim=1)[0]
        loss_oh = F.relu(max_sim - margin).mean()
    else:
        loss_oh = torch.tensor(0.0, device=device)

    # === Total Loss ===
    loss_total = supcon_loss + lambda_pe * loss_pe + lambda_oh * loss_oh

    return loss_total, {
        'supcon': supcon_loss.item(),
        'tightness': loss_pe.item(),
        'push_oe': loss_oh.item()
    }


def opsupcon_loss_single_view(features, labels, prototypes, temperature=0.1, margin=0.4, lambda_pe=0.1, lambda_oh=0.5):
    """
    Modified version of opsupcon_loss to work with single-view features: [B, D]
    """
    device = features.device
    B = features.shape[0]
    features = F.normalize(features, dim=1)  # [B, D]

    contrast_feature = features  # [B, D]
    anchor_feature = features
    anchor_count = 1  # only one view

    logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)  # [B, B]

    labels = labels.contiguous().view(-1)  # [B]

    # Build mask only for ID samples
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)  # [B, B]

    # Remove OE samples from positive pairs
    invalid = (labels == -1)
    mask[invalid, :] = 0
    mask[:, invalid] = 0

    # Compute SupCon Loss
    logits_mask = torch.ones_like(mask).fill_diagonal_(0)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    supcon_loss = -mean_log_prob_pos.mean()

    # Tightness loss (PE)
    if prototypes is not None:
        with torch.no_grad():
            norm_proto = F.normalize(prototypes, dim=1)
        idx_id = (labels != -1)
        id_features = features[idx_id]  # [M, D]
        id_labels = labels[idx_id]
        proto_targets = norm_proto[id_labels.long()]  # [M, D]
        loss_pe = (1 - F.cosine_similarity(id_features, proto_targets)).mean()
    else:
        loss_pe = torch.tensor(0.0, device=device)

    # OE Push Loss (OH)
    with torch.no_grad():
        norm_proto = F.normalize(prototypes, dim=1)

    idx_oe = (labels == -1)
    if idx_oe.any():
        features_oe = features[idx_oe]  # [K, D]
        sim_oe = torch.matmul(F.normalize(features_oe, dim=1), norm_proto.T)  # [K, C]
        max_sim = sim_oe.max(dim=1)[0]
        loss_oh = F.relu(max_sim - margin).mean()
    else:
        loss_oh = torch.tensor(0.0, device=device)

    loss_total = supcon_loss + lambda_pe * loss_pe + lambda_oh * loss_oh
    return loss_total, {
        'supcon': supcon_loss.item(),
        'tightness': loss_pe.item(),
        'push_oe': loss_oh.item()
    }
