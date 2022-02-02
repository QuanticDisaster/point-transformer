import sys
sys.path.append(r'/home/tidop/Documents/pt_transfo_adapted/point-transformer/')

import os

import numpy as np
#import SharedArray as SA
from torch.utils.data import Dataset

from util.custom_data_util import data_prepare
import torch

class S3DIS(Dataset):
    def __init__(self, split='train', data_root='', test_area=None, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        data_root = r'/home/tidop/Downloads/nubes de puntos/PT_S3DIS_pt_files'
        self.split, self.loop = split, loop
        self.data_root = os.path.join(data_root, split)
        self.data_list = sorted(os.listdir(self.data_root))
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = torch.load( os.path.join(self.data_root, self.data_list[ data_idx ]))
        
        coord, feat, label, offset = data#data[:, 0:3], data[:, 3:6], data[:, 6]
        #coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        coord, feat, label, offset = coord.numpy(), feat.numpy(), label.numpy(), offset.numpy()
        coord, feat, label = data_prepare(coord, feat, label, self.split, 0.04, 80000, None, False)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
