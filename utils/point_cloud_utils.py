import torch
from pytorch3d.loss import chamfer_distance


def normalize_point_cloud(pcl):
    """Normalize the point cloud to a unit sphere centered at 0."""
    pcl = pcl.detach().clone()
    pcl -= torch.mean(pcl, dim=0)[None]
    pcl /= torch.cdist(pcl, torch.zeros(1, 3, device=pcl.device)).max(dim=0).values[None]
    return pcl


def augment_pcl(pcl, variation=0.1):
    """Scale given point cloud independently with a random value in every axis direction."""
    anisotropic_scale = torch.FloatTensor(3, device=pcl.device).uniform_(1.0 - variation, 1.0 + variation)
    scale_invariant_augmentation = torch.FloatTensor(3, device=pcl.device).uniform_(-0.01, 0.01)
    return pcl.detach().clone() * anisotropic_scale + scale_invariant_augmentation


def pairwise_chamfer_distance(pcl, pcl_2=None):
    """Compute pairwise chamfer distance"""
    if isinstance(pcl, list):
        dev = pcl[0].device
    else:
        dev = pcl.device

    if pcl_2 is None:
        n_pcl = len(pcl)
        dists = torch.zeros((n_pcl, n_pcl), device=dev)
        for i in range(n_pcl):
            dists[i] = chamfer_distance(pcl[i].repeat(n_pcl, 1, 1).cuda(), pcl.cuda(), batch_reduction=None)[0].to(dev)

    else:
        dists = torch.zeros((len(pcl), len(pcl_2)), device=dev)
        for i, pcl_i in enumerate(pcl):
            for j, pcl_j in enumerate(pcl_2):
                dists[i, j] = chamfer_distance(pcl_i[None].cuda(), pcl_j[None].cuda())[0].squeeze().to(dev)
    return dists
