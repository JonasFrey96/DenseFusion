import torch
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    sys.path.append(os.path.join(os.getcwd() + '/src'))
    sys.path.append(os.path.join(os.getcwd() + '/lib'))

from rotations import norm_quat


def quaternion_raw_multiply(p, q):
    """

    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        p: Quaternions as tensor of shape (..., 4), real part first.
        q: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of p and q, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(p, -1)
    bw, bx, by, bz = torch.unbind(q, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def compose_quat(p, q):
    """
    input is wxyz
    Returns:
      out = normalized( p * q )
    """
    out = quaternion_raw_multiply(norm_quat(p), norm_quat(q))
    return norm_quat(out)


def test_compose_quat():
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from scipy.stats import special_ortho_group
    from rotations import RearangeQuat
    import time
    bs = 1000
    re_q = RearangeQuat(bs)
    mat = special_ortho_group.rvs(dim=3, size=bs)
    quat = R.from_matrix(mat).as_quat()
    q = torch.from_numpy(quat.astype(np.float32)).cuda()
    re_q(q, input_format='xyzw')

    mat2 = special_ortho_group.rvs(dim=3, size=bs)
    quat2 = R.from_matrix(mat2).as_quat()
    q2 = torch.from_numpy(quat2.astype(np.float32)).cuda()
    re_q(q2, input_format='xyzw')

    r1 = R.from_matrix(mat)
    R_out = r1 * R.from_matrix(mat2)

    print(f'scipy xyzw {R_out.as_quat()}')

    st = time.time()
    for i in range(0, 1000):
        out = compose_quat(q, q2)
    print(f'torch wxyz { compose_quat(q, q2) } ')
    print(f'took for 1000 iterations of {bs} bs {time.time()-st}s')


if __name__ == "__main__":
    test_compose_quat()
