import numpy as np
import torch
import torch.nn.functional as F


def fmap2pointmap(C12, evecs_x, evecs_y, soft=True):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    dist = torch.cdist(torch.matmul(evecs_x, C12.t()), evecs_y)
    # print('dist shape', dist.shape, C12.shape, evecs_x.shape, evecs_y.shape)
    if soft:
        return dist.transpose(-1, -2)
    else:
        return dist.argmin(dim=-2)


def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21


def refine_pointmap_zoomout(p2p, evecs_x, evecs_y, k_start, step=1):
    """
    ZoomOut to refine a point-to-point map
    Args:
        p2p: point-to-point map: shape x -> shape y. [Vx]
        evecs_x: eigenvectors of shape x. [Vx, K]
        evecs_y: eigenvectors of shape y. [Vy, K]
        k_start (int): number of eigenvectors to start
        step (int, optional): step size. Default 1.
    """
    k_end = evecs_x.shape[1]
    inds = np.arange(k_start, k_end + step, step)

    p2p_refined = p2p
    for i in inds:
        C21_refined = pointmap2fmap(p2p_refined, evecs_x[:, :i], evecs_y[:, :i])
        p2p_refined = fmap2pointmap(C21_refined, evecs_y[:, :i], evecs_x[:, :i])

    return p2p_refined
