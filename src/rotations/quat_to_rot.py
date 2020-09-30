import torch


def quat_to_rot(rot, conv='wxyz', device='cpu'):
    """converts quat into rotation matrix
    Args:
        rot ([type]): [description]
        conv (str, optional): [description]. Defaults to 'wxyz'.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    if conv == 'wxyz':
        w = rot[:, 0]
        x = rot[:, 1]
        y = rot[:, 2]
        z = rot[:, 3]
    elif conv == 'xyzw':
        y = rot[:, 1]
        z = rot[:, 2]
        w = rot[:, 3]
        x = rot[:, 0]
    else:
        raise Exception('undefined quaternion convention')

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    num_rotations = rot.shape[0]
    matrix = torch.empty((num_rotations, 3, 3), device=device)

    matrix[:, 0, 0] = x2 - y2 - z2 + w2
    matrix[:, 1, 0] = 2 * (xy + zw)
    matrix[:, 2, 0] = 2 * (xz - yw)

    matrix[:, 0, 1] = 2 * (xy - zw)
    matrix[:, 1, 1] = - x2 + y2 - z2 + w2
    matrix[:, 2, 1] = 2 * (yz + xw)

    matrix[:, 0, 2] = 2 * (xz + yw)
    matrix[:, 1, 2] = 2 * (yz - xw)
    matrix[:, 2, 2] = - x2 - y2 + z2 + w2

    return matrix
