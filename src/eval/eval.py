import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def dict_to_df(d):
    d1 = {}
    d1['metric'] = list(d.keys())
    d1['value'] = list(d.values())
    return pd.DataFrame.from_dict(d1)


def get_met(pre='test'):

    p = 'cfg/dataset/classes.txt'
    met = []
    with open(p) as f:
        while 1:
            class_input = f.readline()
            if not class_input:
                break
            met.append(f'{pre}_' + class_input[:-1] + '_auc [0 - 100]')
    return met


def get_pvn3d_dict(pre='test'):
    # ADD(S) AUC
    val = [79.3, 91.5, 96.9, 89.0, 97.9, 90.7, 97.1, 98.3, 87.9, 96.0, 96.9,
           95.9, 92.8, 96.0, 95.7, 91.1, 87.2, 91.6, 95.6, 90.5, 98.2]
    d = {}
    met = get_met(pre=pre)
    for k, v in zip(met, val):
        d[k] = v
    return d


def get_deepim_dict(pre='test'):
    # ADD(S) AUC
    val = [78.0, 91.4, 97.6, 90.3, 97.1, 92.2, 83.5, 98.0, 82.2, 94.9,
           97.4, 91.6, 87.0, 94.2, 97.2, 91.5, 92.7, 88.9, 77.9, 77.8, 97.6]
    d = {}
    met = get_met(pre=pre)
    for k, v in zip(met, val):
        d[k] = v
    return d


def get_df_dict(pre='test'):
    # ADD(S) AUC
    val = [73.2, 94.1, 96.5, 85.5, 94.7, 81.9, 93.3, 96.7, 83.6, 83.3,
           96.9, 89.9, 89.5, 88.9, 92.7, 92.8, 77.9, 93.0, 72.5, 69.9, 92.2]
    d = {}
    met = get_met(pre=pre)
    for k, v in zip(met, val):
        d[k] = v
    return d


def compare_df(df_1, df_2=None, key='auc'):
    if df_2 is None:
        df_2 = dict_to_df(get_df_dict())
    df_1 = df_1.sort_values(by=['metric'])
    df_2 = df_2.sort_values(by=['metric'])

    metrics, idx_1, idx_2 = [], [], []
    for j, metric in enumerate(df_1['metric']):
        # only get dis and check if the key is available in dl_bl_2
        # print( metric in df_2['metric'] )
        if metric.find(key) != -1 and metric in list(df_2['metric']) != -1:
            metrics.append(metric)
            idx_1.append(j)
            idx_2.append(list(df_2['metric']).index(metric))

    # plotting to console
    s1 = 45
    BOLD = '\033[1m'
    END = '\033[0m'
    m = [0]*20    
    for m, i1, i2 in zip(metrics, idx_1, idx_2):
        v1 = '%.1f' % float(list(df_1['value'])[i1])
        v2 = '%.1f' % float(list(df_2['value'])[i2])
        if v1 > v2:
            v1 = BOLD + v1 + END
        else:
            v2 = BOLD + v2 + END
        print(str(m), ':', ' ' * int(s1 - len(m)), v1, v2)
    # creating image

    w, h = 400, (len(m) * 15)
    table_img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(table_img)
    bl = (0, 0, 0, 255)
    d.line([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1), (0, 0)], bl, width=4)
    fnt = ImageFont.truetype(font='cfg/fonts/arial.ttf', size=10)
    h = 0
    for m, i1, i2 in zip(metrics, idx_1, idx_2):
        # draw text, half opacity
        d.text((10, 10 + h * 20), str(m) + ':', font=fnt, fill=bl)
        # draw text, full opacity
        v1 = '%.1f' % float(list(df_1['value'])[i1])
        v2 = '%.1f' % float(list(df_2['value'])[i2])
        if v1 < v2:
            d.text((250, 10 + h * 20), str(v1), font=fnt, fill=bl)
            d.text((320, 10 + h * 20), str(v2), fill=bl)
        else:
            d.text((250, 10 + h * 20), str(v1), fill=bl)
            d.text((320, 10 + h * 20), str(v2), font=fnt, fill=bl)
        h += 1
    return table_img


if __name__ == "__main__":
    img = compare(dict_to_df(get_deepim_dict()), df_2=None, key='auc')
    img.save('/home/jonfrey/Debug/comp.png')
    print(get_df_dict())
