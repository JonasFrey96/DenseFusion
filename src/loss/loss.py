from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np
import random
from rotations import quat_to_rot


def knn(ref, query):
    """return indices of ref for each query point. L2 norm

    Args:
        ref ([type]): points * 3
        query ([type]): tar_points * 3

    Returns:
        [knn]: distance = query * 1 , indices = query * 1
    """
    mp2 = ref.unsqueeze(0).repeat(query.shape[0], 1, 1)
    tp2 = query.unsqueeze(1).repeat(1, ref.shape[0], 1)
    dist = torch.norm(mp2 - tp2, dim=2, p=None)
    knn = dist.topk(1, largest=False)
    return knn


def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list, device):
    """ works i checked if manually to give the same result as loss_calculation

    Args:
        pred_r ([type]): [description]
        pred_t ([type]): [description]
        pred_c ([type]): [description]
        target ([type]): [description]
        model_points ([type]): [description]
        idx ([type]): [description]
        points ([type]): [description]
        w ([type]): [description]
        refine ([type]): [description]
        num_point_mesh ([type]): [description]
        sym_list ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    bs, num_p, _ = pred_c.size()

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    base = quat_to_rot(pred_r.contiguous().view(-1, 4),
                       'wxyz', device=device)
    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()

    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(
        1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(
        1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)

    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)
    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    if not refine:
        if idx[0].item() in sym_list:
            knn_obj = knn(
                ref=target[0, :, :], query=pred[0, :, :])
            inds = knn_obj.indices
            target[0, :, :] = target[0, inds[:, 0], :]

    dis = torch.mean(torch.norm(
        (pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    pred_c = pred_c.view(bs, num_p)
    _, which_max = torch.max(pred_c, 1)

    dis = dis.view(bs, num_p)

    enum = torch.range(0, bs - 1, device=device, dtype=torch.long)

    ori_t_sel = ori_t.view(bs, num_p, 3)[enum, which_max, :]
    points_sel = points.view(bs, num_p, 3)[enum, which_max, :]
    ori_base_sel = ori_base.view(bs, num_p, 3, 3)[enum, which_max, :, :]

    t = ori_t_sel + points_sel
    ori_base = ori_base_sel

    points = points.view(bs, num_p, 3)
    ori_t = t.unsqueeze(1).repeat(1, num_p, 1).contiguous()
    new_points = torch.bmm(
        (points - ori_t), ori_base).contiguous().view(bs, num_p, 3)

    tmp1 = ori_target.view(bs, num_p, num_point_mesh, 3)
    new_target = tmp1[:, 0, :, :].view(bs, num_point_mesh, 3).contiguous()
    # ori_t 16 2000 3

    ori_t = t.unsqueeze(1).repeat(1, num_point_mesh, 1).contiguous().view(
        bs, num_point_mesh, 3)

    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    return loss, dis[enum, which_max], new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, device):
        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list, device)
