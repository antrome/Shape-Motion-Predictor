from __future__ import absolute_import, division, print_function
import numpy as np
import torch

def rotmat_to_euler(R,gpus):
    """
    Converts a rotation matrix to Euler angles
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: a (..., 3, 3) rotation matrix Tensor
    Returns:
      eul: a (..., 3) Euler angle representation of R
    """
    base_shape = [int(d) for d in R.shape][:-2]
    zero_dim = torch.zeros(base_shape).to(gpus)
    one_dim = torch.ones(base_shape).to(gpus)

    econd0 = (R[..., 0, 2] == one_dim)
    econd1 = (R[..., 0, 2] == -1.0 * one_dim)
    econd = econd0 | econd1

    e2 = torch.where(
        econd,
        torch.where(econd1, one_dim * np.pi / 2.0, one_dim * -np.pi / 2.0),
        -torch.asin(R[..., 0, 2])
    )
    e1 = torch.where(
        econd,
        torch.atan2(R[..., 1, 2], R[..., 0, 2]),
        torch.atan2(R[..., 1, 2] / torch.cos(e2), R[..., 2, 2] / torch.cos(e2))
    )
    e3 = torch.where(
        econd,
        zero_dim,
        torch.atan2(R[..., 0, 1] / torch.cos(e2), R[..., 0, 0] / torch.cos(e2))
    )

    eul = torch.stack([e1, e2, e3], -1)
    return eul

def expmap_to_rotmat(r,gpus):
    """
    Converts an exponential map angle to a rotation matrix
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      r: (..., 3) exponential map Tensor
    Returns:
      R: (..., 3, 3) rotation matrix Tensor
    """
    base_shape = [int(d) for d in r.shape][:-1]
    zero_dim = torch.zeros(base_shape).to(gpus)

    theta = torch.sqrt(torch.sum(torch.pow(r,2), -1, True) + 1e-8)
    r0 = r / theta

    r0x = torch.reshape(
        torch.stack([zero_dim, -1.0 * r0[..., 2], r0[..., 1],
                  zero_dim, zero_dim, -1.0 * r0[..., 0],
                  zero_dim, zero_dim, zero_dim], -1),
        base_shape + [3, 3]
    )
    trans_dims = range(len(r0x.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    r0x = r0x - r0x.permute(trans_dims)
    #r0x = r0x - torch.transpose(r0x, trans_dims)

    tile_eye = torch.FloatTensor(np.tile(np.reshape(np.eye(3), [1 for _ in base_shape] + [3, 3]), base_shape + [1, 1])).to(gpus)
    #theta = torch.expand_dims(theta, axis=-1)
    theta = theta.unsqueeze(-1)

    R = tile_eye + torch.sin(theta) * r0x + (1.0 - torch.cos(theta)) * torch.matmul(r0x, r0x)
    return R
