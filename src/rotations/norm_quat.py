import torch


def norm_quat(q):
    """normalize quaternion

    Args:
        q ([type]): BS * 4

    Raises:
        type: [description]

    Returns:
        [type]: BS * 4
    """
    # ToDo raise type and dim error
    return q / torch.norm(q, dim=1).unsqueeze(1)


def test_norm_quat():
    bs = 1000
    q = torch.rand(bs, 4)
    out = norm_quat(q)
    print(f'should equal bs {torch.sum(torch.norm(out, dim=1))} of BS {bs}')


if __name__ == "__main__":
    test_norm_quat()
