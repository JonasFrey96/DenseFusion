import torch.utils.data as data
import torch


class Backend(data.Dataset):

    def __init__(self, cfg_d, cfg_env):
        self._length = 0

        self._num_pt_mesh_small = cfg_d['num_pt_mesh_small']
        self._num_pt_mesh_large = cfg_d['num_pt_mesh_large']
        self._num_pt = cfg_d['num_points']

        self._obj_list_sym = cfg_d['obj_list_sym']
        self._obj_list_fil = cfg_d['obj_list_fil']

    def __str__(self):
        # string = 'Dataloader %s conataining %d number of sequences' % (
        #     self._name, len(self)) TODO
        # string += '\n Total Objects: %d Sym Objects: %d' % (
        #     self.objects, len(self._obj_list_sym))
        return "Backend TODO"

    def __len__(self):
        return self._length

    def getElement(self, desig, obj_idx):
        pass
