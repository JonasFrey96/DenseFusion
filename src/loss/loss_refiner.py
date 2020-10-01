from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np
import random

from rotations import quat_to_rot, compose_quat
from helper import knn

class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points, pred_r_current, pred_t_current, use_orig=False):
        bs, _ = pred_r.size()
        num_p = len(points[0])
        pred_r = pred_r / (torch.norm(pred_r, dim=1).view(bs, 1))
        base = quat_to_rot(pred_r.contiguous().view(-1, 4),
                           'wxyz', device=points.device)
        ori_base = base
        base = base.contiguous().transpose(2, 1).unsqueeze(
            0).contiguous().view(-1, 3, 3)

        model_points = model_points.view(
            bs, 1, self.num_pt_mesh, 3).view(bs, self.num_pt_mesh, 3)

        target = target.view(bs, 1, self.num_pt_mesh, 3).view(
            bs, self.num_pt_mesh, 3)

        ori_target = target
        pred_t = pred_t.unsqueeze(1).repeat(
            1, self.num_pt_mesh, 1).contiguous()  # .view(bs * num_p, 1, 3)
        ori_t = pred_t
        # model_points 16 x 2000 x 3
        # base 16 X 3 x 3
        # points 16 X 1 x 3
        pred = torch.add(torch.bmm(model_points, base), pred_t)

        if idx[0].item() in self.sym_list:
            knn_obj = knn(
                ref=target[0, :, :], query=pred[0, :, :])
            inds = knn_obj.indices
            target[0, :, :] = target[0, inds[:, 0], :]

        dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

        t = ori_t
        num_input_points = points.shape[1]
        points = points.view(bs, num_input_points, 3)

        ori_base = ori_base.view(bs, 3, 3).contiguous()
        ori_t = t[:, 0, :].unsqueeze(1).repeat(
            1, num_input_points, 1).contiguous().view(bs, num_input_points, 3)
        new_points = torch.bmm((points - ori_t), ori_base).contiguous()

        new_target = ori_target[0].view(1, self.num_pt_mesh, 3).contiguous()
        ori_t = t[:, 0, :].unsqueeze(1).repeat(
            1, self.num_pt_mesh, 1).contiguous().view(bs, self.num_pt_mesh, 3)
        new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

        # print('------------> ', dis.item(), idx[0].item())

        pred_r_current = compose_quat( pred_r_current, pred_r) #TODO check if this is working !!!

        return dis, new_points.detach(), new_target.detach(), pred_r_current, pred_t_current+pred_t
