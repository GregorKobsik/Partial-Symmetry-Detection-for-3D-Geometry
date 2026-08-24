import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss


def cross_entropy_loss(feat_1, feat_2):
    """ Normalized Temperature-scaled Cross Entropy Loss """
    ntxent_loss = NTXentLoss()
    labels = torch.arange(feat_1.size(0))
    embeddings = torch.cat([feat_1, feat_2], dim=0)
    labels = torch.cat([labels, labels], dim=0)
    return ntxent_loss(embeddings, labels)


def log_ratio_loss(emb_feat, target, normalize=False, metric='cosine'):
    """ Log-Ratio Loss"""
    # auxiliary variables
    dev = emb_feat.device
    m = emb_feat.shape[0] - 1
    epsilon = 1e-6
    pdist = nn.PairwiseDistance(p=2)

    if normalize:
        # Normalize Features
        emb_feat = F.normalize(emb_feat, dim=-1)

    def _log_ratio_loss(anker_feat, paired_feat, gt_dist, normalize, metric):
        # filter out indices
        idxs = torch.arange(0, m, device=dev)
        indc = idxs.repeat(m, 1).t() < idxs.repeat(m, 1)

        # uniform weight coefficients
        wgt = indc.clone().float()
        wgt /= wgt.sum()

        if metric == 'euclidean':
            # Euclidean Distance
            dist = pdist(anker_feat[None, :], paired_feat)  # [B]
        elif metric == 'cosine':
            # Cosine Distance
            dist = 1 - F.cosine_similarity(anker_feat[None, :], paired_feat)
        else:
            print(f"ERROR: {metric} invalid. Select from: 'euclidean', 'cosine'.")

        # compute ratios of anker to every two other samples
        log_dist = torch.log(dist + epsilon)  # [B]
        log_gt_dist = torch.log(gt_dist + epsilon)  # [B]
        diff_log_dist = log_dist.repeat(m, 1).t() - log_dist.repeat(m, 1)  # [B x B]
        diff_log_gt_dist = log_gt_dist.repeat(m, 1).t() - log_gt_dist.repeat(m, 1)  # [B x B]

        # compute mean squared differences
        log_ratio = (diff_log_dist - diff_log_gt_dist).pow(2)
        return (log_ratio * wgt).sum()

    # select every feature as anker feature and pair it against each other
    loss = 0
    for i in range(m + 1):
        anker_feat = emb_feat[i]
        paired_feat = torch.cat([emb_feat[:i], emb_feat[i + 1:]])
        paired_dist = torch.cat([target[i, :i], target[i, i + 1:]])
        loss += _log_ratio_loss(anker_feat, paired_feat, paired_dist, normalize, metric)
    return loss / (m + 1)


def l1_loss(emb_feat, target, normalize=False, metric='cosine', weighting='none'):
    """ L1 Loss"""
    if normalize:
        # Normalize Features
        emb_feat = F.normalize(emb_feat, dim=-1)

    if metric == 'euclidean':
        # Euclidean Distance
        dist = torch.cdist(emb_feat, emb_feat)  # [B x B]
    elif metric == 'cosine':
        # Cosine Distance (scaled to [0.0, 0.25])
        cosine_similarity = F.cosine_similarity(emb_feat[:, None, :], emb_feat[None, :, :], dim=-1)
        dist = (1 - cosine_similarity) / 8.0
    else:
        print(f"ERROR: {metric} invalid. Select from: 'euclidean', 'cosine'.")

    if weighting == 'none':
        # No Weighting
        return F.l1_loss(dist, target, reduction='mean')
    elif weighting == 'mean-var':
        # Mean-Var Weighting
        batch_size = emb_feat.shape[0]

        var = torch.var(target, dim=-1)
        alpha = var / torch.max(var)
        alpha = alpha.repeat_interleave(batch_size)

        mean = torch.mean(target, dim=-1)
        weighting = torch.abs(mean.repeat_interleave(batch_size) - dist) / mean.mean()

        l1_loss = F.l1_loss(dist, target.ravel(), reduction='none')
        return ((1 - alpha) * weighting * l1_loss + alpha * l1_loss).mean()
    else:
        print(f"ERROR: {weighting} invalid. Select from: 'none', 'mean-var'.")
