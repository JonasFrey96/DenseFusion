import torch
from torch import nn


class RearangeQuat(nn.Module):

    def __init__(self, bs):
        """
        Args:
            batch_size ([int]): batch size of the quaternion. This allows reserving memory for shuffeling the quaternion before execution
        """
        super(RearangeQuat, self).__init__()

        self.mem = torch.zeros((bs))
        self.bs = bs

    def forward(self, q, input_format):
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        assert q.shape[0] == self.bs

        if input_format == 'xyzw':
            self.mem = q[:, 0].clone()

            q[:, 0] = q[:, 3]
            q[:, 3] = q[:, 2]
            q[:, 2] = q[:, 1]
            q[:, 1] = self.mem

        elif input_format == 'wxyz':
            self.mem = q[:, 0].clone()

            q[:, 0] = q[:, 1]
            q[:, 1] = q[:, 2]
            q[:, 2] = q[:, 3]
            q[:, 3] = self.mem
        return q


if __name__ == "__main__":
    bs = 10
    re_q = RearangeQuat(bs)
    from scipy.spatial.transform import Rotation as R
    from scipy.stats import special_ortho_group
    mat = special_ortho_group.rvs(dim=3, size=bs)
    quat = R.from_matrix(mat).as_quat()

    q = torch.from_numpy(quat)
    print('Input', q)
    re_q(q, input_format='xyzw')
    print('Output', q)
    re_q(q, input_format='wxyz')
    print('Same as Input', q)
