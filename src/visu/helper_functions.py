import datetime
import imageio
from pathlib import Path
import numpy as np


def save_image(image_array, tag='img', p_store='~/track_this/results/imgs/', time_stamp=False):
    """
    image_array = np.array [width,height,RGB]
    """
    if p_store[-1] != '/':
        p_store = p_store + '/'
    Path(p_store).mkdir(parents=True, exist_ok=True)

    if time_stamp:
        tag = str(datetime.datetime.now().replace(
            microsecond=0).isoformat()) + tag
    if image_array.dtype == np.float32:
        image_array = image_array.astype(np.uint8)
    imageio.imwrite(p_store + tag + '.jpg', image_array)
