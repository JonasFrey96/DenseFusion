import yagmail
from sklearn.neighbors import NearestNeighbors
import yaml
import numpy as np
import collections

import torch
import copy

def batched_index_select(t, inds, dim=1):
    """index batch tensor

    Args:
        t ([torch.Tensor]): BS x select_dim x Features or BS x select_dim x Feat1 x Feat2 
        dim ([int]): select_dim = 1
        inds ([torch.Tensor]): BS x select_dim

    Returns:
        [type]: [description]
    """
    if len(t.shape) == 3:
        dummy = inds.unsqueeze(2).expand(
            inds.size(0), inds.size(1), t.size(2))
    elif len(t.shape) == 4:
        dummy = inds.unsqueeze(2).unsqueeze(3).expand(
            inds.size(0), inds.size(1), t.size(2), t.size(3))
    elif len(t.shape) == 5:
        dummy = inds[:, :, None, None, None].expand(
            inds.size(0), inds.size(1), t.size(2), t.size(3), t.size(4))
    out = t.gather(dim, dummy)  # b x e x f
    return out



def flatten_list(d, parent_key='', sep='_'):
    items = []
    for num, element in enumerate(d):
        new_key = parent_key + sep + str(num) if parent_key else str(num)

        if isinstance(element, collections.MutableMapping):
            items.extend(flatten_dict(element, new_key, sep=sep).items())
        else:
            if isinstance(element, list):
                if isinstance(element[0], dict):
                    items.extend(flatten_list(element, new_key, sep=sep))
                    continue
            items.append((new_key, element))
    return items


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                if isinstance(v[0], dict):
                    items.extend(flatten_list(v, new_key, sep=sep))
                    continue
            items.append((new_key, v))
    return dict(items)


def pad(s, sym='-', p='l', length=80):
    if len(s) > length:
        return s
    else:
        if p == 'c':
            front = int((length - len(s)) / 2)
            s = sym * front + s
            back = int(length - len(s))
            s = s + sym * back
        if p == 'l':
            back = int(length - len(s))
            s = s + sym * back
        return s


def get_bbox_480_640(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280,
                   320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640

    # print(type(label))
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
