import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import datetime
import sys
import os
import time
import shutil
import argparse
import logging
import signal
import pickle


# misc
import numpy as np
import pandas as pd
import random
import sklearn
from scipy.spatial.transform import Rotation as R
from math import pi
import coloredlogs
import datetime

sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))

import torch
import torch.autograd.profiler as profiler

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


from scipy.spatial.transform import Rotation as R

coloredlogs.install()

# network dense fusion
from lib.network import PoseNet, PoseRefineNet


# src modules
from helper import pad
from helper import re_quat, flatten_dict
from helper import get_bb_from_depth, get_bb_real_target
from helper import backproject_points_batch, backproject_points, backproject_point

from dataset import GenericDataset
from visu import Visualizer

from eval import *
from rotations import *
from loss import compute_auc, LossAddS

def ret_cropped_image(img):
    test = torch.nonzero(img[:, :, :])
    a = torch.max(test[:, 0]) + 1
    b = torch.max(test[:, 1]) + 1
    c = torch.max(test[:, 2]) + 1
    return img[:a, :b, :c]

def padded_cat(list_of_images, device):
    """returns torch.tensor of concatenated images with dim = max size of image padded with zeros

    Args:
        list_of_images ([type]): List of Images Channels x Heigh x Width

    Returns:
        padded_cat [type]: Tensor of concatination result len(list_of_images) x Channels x max(Height) x max(Width)
        valid_indexe: len(list_of_images) x 2
    """
    c = list_of_images[0].shape[0]
    h = [x.shape[1] for x in list_of_images]
    w = [x.shape[2] for x in list_of_images]
    max_h = max(h)
    max_w = max(w)
    padded_cat = torch.zeros(
        (len(list_of_images), c, max_h, max_w), device=device)
    for i, img in enumerate(list_of_images):
        padded_cat[i, :, :h[i], :w[i]] = img

    valid_indexes = torch.tensor([h, w], device=device)
    return padded_cat, valid_indexes

def tight_image_batch(img_batch, device):
    ls = []
    for i in range(img_batch.shape[0]):
        ls.append(ret_cropped_image(img_batch[i]))

    tight_padded_img_batch, valid_indexes = padded_cat(
        ls,
        device=device)
    return tight_padded_img_batch

def check_exp(exp):
    if exp['d_test'].get('overfitting_nr_idx', -1) != -1 or exp['d_train'].get('overfitting_nr_idx', -1) != -1:
        print('Overfitting on ONE batch is activated')
        time.sleep(5)

class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self._mode = 'init'

        # check experiment cfg for errors
        check_exp(exp)

        # logging h-params
        exp_config_flatten = flatten_dict(copy.deepcopy(exp))
        for k in exp_config_flatten.keys():
            if exp_config_flatten[k] is None:
                exp_config_flatten[k] = 'is None'

        self.hparams = exp_config_flatten
        self.hparams['lr'] = exp['training']['lr']
        self.test_size = exp['training']['test_size']
        self.env, self.exp = env, exp

        # number of input points to the network
        num_points_small = exp['d_train']['num_pt_mesh_small']
        num_points_large = exp['d_train']['num_pt_mesh_large']
        num_obj = exp['d_train']['objects']

        self.df_pose_estimator = PoseNet(
            num_points=exp['d_test']['num_points'], num_obj=num_obj)

        self.df_refiner = PoseRefineNet(
            num_points=exp['d_test']['num_points'], num_obj=num_obj)

        if exp.get('model', {}).get('df_load', False):
            self.df_pose_estimator.load_state_dict(
                torch.load(exp['model']['df_pose_estimator']))
            if exp.get('model', {}).get('df_refine', False):
                self.df_refiner.load_state_dict(
                    torch.load(exp['model']['df_refiner']))

        self.criterion_adds = LossAddS(sym_list=exp['d_train']['obj_list_sym'])

        self.visualizer = None
        
        self._dict_track = {}
        self.number_images_log_test = self.exp.get(
            'visu', {}).get('number_images_log_test', 1)
        self.counter_images_logged = 0
        self.init_train_vali_split = False

        mp = exp['model_path']
        fh = logging.FileHandler(f'{mp}/Live_Logger_Lightning.log')
        fh.setLevel(logging.DEBUG)
        logging.getLogger("lightning").addHandler(fh)
        
        self.start = time.time()
        
        # optional, set the logging level
        if self.exp.get('visu', {}).get('log_to_file', False):
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            logging.getLogger("lightning").addHandler(console)
            log = open(f'{mp}/Live_Logger_Lightning.log', "a")
            sys.stdout = log
            logging.info('Logging to File')

    def forward(self, batch):
        st = time.time()
        
        # unpack batch
        points, choose, img, target, model_points, idx = batch[0:6]
        depth_img, label, img_orig, cam = batch[6:10]
        gt_rot_wxyz, gt_trans, unique_desig = batch[10:13]
        log_scalars = {}
        bs = points.shape[0]


        out_rx, out_tx, out_cx, emb = self.df_pose_estimator(img ,x , choose, obj)

        for i in range( self.exp['training']['refine_iterations'] ):
            out_rx, out_tx = self.df_refiner(x,emb,obj)


        return pred_trans, pred_rot_wxyz, pred_points, log_scalars

    def training_step(self, batch, batch_idx):
        self._mode = 'train'
        st = time.time()
        total_loss = 0
        total_dis = 0
        
        # forward
        pred_trans, pred_rot_wxyz, pred_points, log_scalars = self(batch[0])

        # calculate loss
        self.criterion_adds() 

        if self.counter_images_logged < self.exp.get('visu', {}).get('images_train', 1):
            self.visu_batch(batch, pred_trans, pred_rot_wxyz, pred_points)

        # tensorboard logging
        tensorboard_logs = {'train_loss': float(loss)}

        tensorboard_logs = {**tensorboard_logs, **log_scalars}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': {'L_Seg': log_scalars['loss_segmentation'], 'L_Add': log_scalars['loss_pose_add'], 'L_Tra': log_scalars[f'loss_translation']}}

    def validation_epoch_end(self, outputs):
        avg_dict = {}
        self.counter_images_logged = 0  # reset image log counter

        # only keys that are logged in tensorboard are removed from log_scalars !
        for old_key in list(self._dict_track.keys()):
            if old_key.find('val') == -1:
                continue

            newk = 'avg_' + old_key
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))

            p = old_key.find('adds_dis')
            if p != -1:
                auc = compute_auc(self._dict_track[old_key])
                avg_dict[old_key[:p] + 'auc [0 - 100]'] = auc

            self._dict_track.pop(old_key, None)

        df1 = dict_to_df(avg_dict)
        df2 = dict_to_df(get_df_dict(pre='val'))
        img = compare_df(df1, df2, key='auc [0 - 100]')
        tag = 'val_table_res_vs_df'
        img.save(self.exp['model_path'] +
                 f'/visu/{self.current_epoch}_{tag}.png')
        self.logger.experiment.add_image(tag, np.array(img).astype(
            np.uint8), global_step=self.current_epoch, dataformats='HWC')

        avg_val_dis_float = float(avg_dict['avg_val_loss  [+inf - 0]'])
        return {'avg_val_dis_float': avg_val_dis_float,
                'avg_val_dis': avg_dict['avg_val_loss  [+inf - 0]'],
                'log': avg_dict}

    def train_epoch_end(self, outputs):
        self.counter_images_logged = 0  # reset image log counter
        avg_dict = {}
        for old_key in list(self._dict_track.keys()):
            if old_key.find('train') == -1:
                continue
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
            self._dict_track.pop(old_key, None)
        string = 'Time for one epoch: ' + str(time.time() - self.start)
        print(string)
        self.start = time.time()
        return {**avg_dict, 'log': avg_dict}

    def test_epoch_end(self, outputs):
        self.counter_images_logged = 0  # reset image log counter
        avg_dict = {}
        # only keys that are logged in tensorboard are removed from log_scalars !
        for old_key in list(self._dict_track.keys()):
            if old_key.find('test') == -1:
                continue

            newk = 'avg_' + old_key
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))

            p = old_key.find('adds_dis')
            if p != -1:
                auc = compute_auc(self._dict_track[old_key])
                avg_dict[old_key[:p] + 'auc [0 - 100]'] = auc

            self._dict_track.pop(old_key, None)

        avg_test_dis_float = float(avg_dict['avg_test_loss  [+inf - 0]'])

        df1 = dict_to_df(avg_dict)
        df2 = dict_to_df(get_df_dict(pre='test'))
        img = compare_df(df1, df2, key='auc [0 - 100]')
        tag = 'test_table_res_vs_df'
        img.save(self.exp['model_path'] +
                 f'/visu/{self.current_epoch}_{tag}.png')
        self.logger.experiment.add_image(tag, np.array(img).astype(
            np.uint8), global_step=self.current_epoch, dataformats='HWC')

        return {'avg_test_dis_float': avg_test_dis_float,
                'avg_test_dis': avg_dict['avg_test_loss  [+inf - 0]'],
                'log': avg_dict}

    def visu_pose(self, batch_idx, pred_r, pred_t, target, model_points, cam, img_orig, unique_desig, idx, store=True):
        if self.visualizer is None:
            self.visualizer = Visualizer(self.exp['model_path'] +
                                         '/visu/', self.logger.experiment)
        points = copy.deepcopy(target.detach().cpu().numpy())
        img = img_orig.detach().cpu().numpy()
        if self.exp['visu'].get('visu_gt', False):
            self.visualizer.plot_estimated_pose(tag='gt_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                                epoch=self.current_epoch,
                                                img=img,
                                                points=points,
                                                cam_cx=float(cam[0]),
                                                cam_cy=float(cam[1]),
                                                cam_fx=float(cam[2]),
                                                cam_fy=float(cam[3]),
                                                store=store)
            self.visualizer.plot_contour(tag='gt_contour_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                         epoch=self.current_epoch,
                                         img=img,
                                         points=points,
                                         cam_cx=float(cam[0]),
                                         cam_cy=float(cam[1]),
                                         cam_fx=float(cam[2]),
                                         cam_fy=float(cam[3]),
                                         store=store)

        t = pred_t.detach().cpu().numpy()
        r = pred_r.detach().cpu().numpy()

        rot = R.from_quat(re_quat(r, 'wxyz'))

        self.visualizer.plot_estimated_pose(tag='pred_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                            epoch=self.current_epoch,
                                            img=img,
                                            points=copy.deepcopy(
            model_points[:, :].detach(
            ).cpu().numpy()),
            trans=t.reshape((1, 3)),
            rot_mat=rot.as_matrix(),
            cam_cx=float(cam[0]),
            cam_cy=float(cam[1]),
            cam_fx=float(cam[2]),
            cam_fy=float(cam[3]),
            store=store)

        self.visualizer.plot_contour(tag='pred_contour_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                     epoch=self.current_epoch,
                                     img=img,
                                     points=copy.deepcopy(
            model_points[:, :].detach(
            ).cpu().numpy()),
            trans=t.reshape((1, 3)),
            rot_mat=rot.as_matrix(),
            cam_cx=float(cam[0]),
            cam_cy=float(cam[1]),
            cam_fx=float(cam[2]),
            cam_fy=float(cam[3]),
            store=store)

        render_img, depth, h_render = self.vm.get_closest_image_batch(
            i=idx.unsqueeze(0), rot=pred_r.unsqueeze(0), conv='wxyz')
        # get the bounding box !
        w = 640
        h = 480

        real_img = torch.zeros((1, 3, h, w), device=self.device)
        # update the target to get new bb

        base_inital = quat_to_rot(
            pred_r.unsqueeze(0), 'wxyz', device=self.device).squeeze(0)
        base_new = base_inital.view(-1, 3, 3).permute(0, 2, 1)
        pred_points = torch.add(
            torch.bmm(model_points.unsqueeze(0), base_inital.unsqueeze(0)), pred_t)
        # torch.Size([16, 2000, 3]), torch.Size([16, 4]) , torch.Size([16, 3])
        bb_ls = get_bb_real_target(
            pred_points, cam.unsqueeze(0))

        for j, b in enumerate(bb_ls):
            if not b.check_min_size():
                pass
            c = cam.unsqueeze(0)
            center_real = backproject_points(
                pred_t.view(1, 3), fx=c[j, 2], fy=c[j, 3], cx=c[j, 0], cy=c[j, 1])
            center_real = center_real.squeeze()
            b.move(-center_real[0], -center_real[1])
            b.expand(1.1)
            b.expand_to_correct_ratio(w, h)
            b.move(center_real[0], center_real[1])
            crop_real = b.crop(img_orig).unsqueeze(0)
            up = torch.nn.UpsamplingBilinear2d(size=(h, w))
            crop_real = torch.transpose(crop_real, 1, 3)
            crop_real = torch.transpose(crop_real, 2, 3)
            real_img[j] = up(crop_real)
        inp = real_img[0].unsqueeze(0)
        inp = torch.transpose(inp, 1, 3)
        inp = torch.transpose(inp, 1, 2)
        data = torch.cat([inp, render_img], dim=3)
        data = torch.transpose(data, 1, 3)
        data = torch.transpose(data, 2, 3)
        self.visualizer.visu_network_input(tag='render_real_comp_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                           epoch=self.current_epoch,
                                           data=data,
                                           max_images=1, store=store)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.pixelwise_refiner.parameters()}], lr=self.hparams['lr'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.exp['lr_cfg']['on_plateau_cfg']),
            **self.exp['lr_cfg']['scheduler']
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset_train = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        # initalize train and validation indices
        if not self.init_train_vali_split:
            self.init_train_vali_split = True
            self.indices_valid, self.indices_train = sklearn.model_selection.train_test_split(
                range(0, len(dataset_train)), test_size=self.test_size)

        dataset_subset = torch.utils.data.Subset(
            dataset_train, self.indices_train)

        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       **self.exp['loader'])
        return dataloader_train

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_test'],
            cfg_env=self.env)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      **self.exp['loader'])
        return dataloader_test

    def val_dataloader(self):
        dataset_val = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)
        # initalize train and validation indices
        if not self.init_train_vali_split:
            self.init_train_vali_split = True
            self.indices_valid, self.indices_train = sklearn.model_selection.train_test_split(
                range(0, len(dataset_val)), test_size=self.test_size)

        dataset_subset = torch.utils.data.Subset(
            dataset_val, self.indices_valid)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     **self.exp['loader'])
        return dataloader_val

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

def move_dataset_to_ssd(env, exp):
    # costum code to move dataset on cluster
    try:
        if env.get('leonhard', {}).get('copy', False):
            files = ['data', 'data_syn', 'models']
            p_ls = os.popen('echo $TMPDIR').read().replace('\n', '')

            p_ycb_new = p_ls + '/YCB_Video_Dataset'
            p_ycb = env['p_ycb']
            try:
                os.mkdir(p_ycb_new)
                os.mkdir('$TMPDIR/YCB_Video_Dataset')
            except:
                pass
            for f in files:

                p_file_tar = f'{p_ycb}/{f}.tar'
                logging.info(f'Copying {f} to {p_ycb_new}/{f}')

                if os.path.exists(f'{p_ycb_new}/{f}'):
                    logging.info(
                        "data already exists! Interactive session?")
                else:
                    start_time = time.time()
                    if f == 'data':
                        bashCommand = "tar -xvf" + p_file_tar + \
                            " -C $TMPDIR | awk 'BEGIN {ORS=\" \"} {if(NR%1000==0)print NR}\' "
                    else:
                        bashCommand = "tar -xvf" + p_file_tar + \
                            " -C $TMPDIR/YCB_Video_Dataset | awk 'BEGIN {ORS=\" \"} {if(NR%1000==0)print NR}\' "
                    os.system(bashCommand)
                    logging.info(
                        f'Transferred {f} folder within {str(time.time() - start_time)}s to local SSD')

            env['p_ycb'] = p_ycb_new
    except:
        env['p_ycb'] = p_ycb_new
        logging.info('Copying data failed')
    return exp, env

def move_background(env, exp):
    try:
        # Update the env for the model when copying dataset to ssd
        if env.get('leonhard', {}).get('copy', False):
            p_file_tar = env['p_background'] + '/indoorCVPR_09.tar'
            p_ls = os.popen('echo $TMPDIR').read().replace('\n', '')
            p_n = p_ls + '/Images'
            try:
                os.mkdir(p_n)
            except:
                pass

            if os.path.exists(f'{p_n}/office'):
                logging.info(
                    "data already exists! Interactive session?")
            else:
                start_time = time.time()
                bashCommand = "tar -xvf" + p_file_tar + \
                    " -C $TMPDIR | awk 'BEGIN {ORS=\" \"} {if(NR%1000==0)print NR}\' "
                os.system(bashCommand)

            env['p_background'] = p_n
    except:
        logging.info('Copying data failed')
    return exp, env

def load_from_file(p):
    if os.path.isfile(p):
        with open(p, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError
    return data


if __name__ == "__main__":
    seed_everything(42)

    def signal_handler(signal, frame):
        print('exiting on CRTL-C')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='cfg/exp/exp.yml',
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                        help='The environment yaml file.')
    args = parser.parse_args()
    exp_cfg_path = args.exp
    env_cfg_path = args.env

    exp = load_from_file(exp_cfg_path)
    env = load_from_file(env_cfg_path)

    if exp['model_path'].split('/')[-2] == 'debug':
        p = '/'.join(exp['model_path'].split('/')[:-1])
        try:
            shutil.rmtree(p)
        except:
            pass
        timestamp = '_'
    else:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    p = exp['model_path'].split('/')
    p.append(str(timestamp) + '_' + p.pop())
    new_path = '/'.join(p)
    exp['model_path'] = new_path
    model_path = exp['model_path']

    # copy config files to model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print((pad("Generating network run folder")))
    else:
        print((pad("Network run folder already exits")))

    if exp.get('visu', {}).get('log_to_file', False):
        log = open(f'{model_path}/Live_Logger_Lightning.log', "a")
        sys.stdout = log
        print('Logging to File')
    
    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]

    print(pad(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}'))
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')

    exp, env = move_dataset_to_ssd(env, exp)
    exp, env = move_background(env, exp)
    dic = {'exp': exp, 'env': env}
    model = TrackNet6D(**dic)

    early_stop_callback = EarlyStopping(
      **exp['early_stopping'])
    
    checkpoint_callback = ModelCheckpoint(
        filepath=exp['model_path'] + '/{epoch}-{avg_val_dis_float:.4f}',
        **exp['model_checkpoint'])

    if exp.get('checkpoint_restore', False):
        checkpoint = torch.load(
            exp['checkpoint_load'], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    with torch.autograd.set_detect_anomaly(True):
        trainer = Trainer(**exp['trainer'],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        default_root_dir=exp['model_path'])

        if exp.get('model_mode', 'fit') == 'fit':
            trainer.fit(model)
        elif exp.get('model_mode', 'fit') == 'test':
            trainer.test(model)
        else:
            print("Wrong model_mode defined in exp config")
            raise Exception
