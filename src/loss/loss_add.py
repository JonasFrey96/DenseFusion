import torch
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    sys.path.append(os.path.join(os.getcwd() + '/src'))
    sys.path.append(os.path.join(os.getcwd() + '/lib'))

from rotations import quat_to_rot
from helper import knn


def loss_calculation_add(pred_r, pred_t, target, model_points, idx, sym_list):
    """ADD loss calculation

    Args:
        pred_r ([type]): BS * 3
        pred_t ([type]): BS * 4 'wxyz'
        idx ([type]): BS * 1

        model_points ([type]): BS * num_points * 3 : randomly selected points of the CAD model
        target ([type]): BS * num_points * 3 : model_points rotated and translated according to the regression goal (not ground truth because of data augmentation)

        sym_list ([list of integers]):
    Returns:
        [type]: [description]
    """
    bs, num_p, _ = target.shape
    num_point_mesh = num_p

    pred_r = pred_r / torch.norm(pred_r, dim=1).view(bs, 1)
    base = quat_to_rot(pred_r, 'wxyz').unsqueeze(1)
    base = base.view(-1, 3, 3).cuda().permute(0, 2, 1)  # transposed of R

    pred_t = pred_t.unsqueeze(1)

    pred = torch.add(torch.bmm(model_points, base), pred_t)
    tf_model_points = pred.view(target.shape)

    for i in range(bs):
        # ckeck if add-s or add
        if idx[i, 0].item() in sym_list:

            knn_obj = knn(
                ref=target[i, :, :], query=tf_model_points[i, :, :])
            inds = knn_obj.indices
            target[i, :, :] = target[i, inds[:, 0], :]

    dis = torch.mean(torch.norm((tf_model_points - target), dim=2), dim=1)
    return dis


class LossAddS(nn.Module):

    def __init__(self, sym_list):
        super(LossAddS, self).__init__()
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx):
        return loss_calculation_add(pred_r, pred_t, target, model_points, idx, self.sym_list)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from scipy.stats import special_ortho_group
    from helper import re_quat
    from deep_im import RearangeQuat

    device = 'cuda:0'
    bs = 100
    nr_points = 3000

    re_q = RearangeQuat(bs)
    mat = special_ortho_group.rvs(dim=3, size=bs)
    quat = R.from_matrix(mat).as_quat()
    q = torch.from_numpy(quat.astype(np.float32)).cuda()
    re_q(q, input_format='xyzw')
    pred_r = q.unsqueeze(0)

    pred_t_zeros = torch.zeros((bs, 3), device=device)
    pred_t_ones = torch.ones((bs, 3), device=device)

    model_points = torch.rand((bs, nr_points, 3), device=device)
    res = 0.15
    target_points = model_points + \
        torch.ones((bs, nr_points, 3), device=device) * \
        float(np.sqrt(res * res / 3))
    pred_r_unit = torch.zeros((bs, 4), device=device)
    pred_r_unit[:, 0] = 1

    sym_list = [0]
    loss_add = LossAddS(sym_list=sym_list)

    idx_sym = torch.zeros((bs, 1), device=device)
    idx_nonsym = torch.ones((bs, 1), device=device)

    points = target_points
    num_pt_mesh = nr_points

    loss = loss_add(pred_r_unit, pred_t_zeros,
                    target_points, model_points, idx_nonsym)
    print(f'dis = {loss} should be {res}')

    rand_index = torch.randperm(nr_points)
    target_points = model_points[:, rand_index, :]
    loss = loss_add(pred_r_unit, pred_t_zeros,
                    target_points, model_points, idx_nonsym)

    loss = loss_add(pred_r_unit, pred_t_zeros,
                    target_points, model_points, idx_sym)
