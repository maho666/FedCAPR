from __future__ import print_function, absolute_import
import os.path as osp
import tarfile

import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset

# class Dataset_MSMT(BaseImageDataset):
class MSMT17(BaseImageDataset):
    def __init__(self, root):
        super(MSMT17, self).__init__()
        dataset_dir = 'MSMT17_V1'
        self.data_dir = osp.join(root, dataset_dir)
        self.all_img_prefix = {}

        self.train_list = osp.join(self.data_dir, 'list_train.txt')
        self.val_list = osp.join(self.data_dir, 'list_val.txt')
        self.query_list = osp.join(self.data_dir, 'list_query.txt')
        self.gallery_list = osp.join(self.data_dir, 'list_gallery.txt')

        self.train_dir = osp.join(self.data_dir, 'train')
        self.test_dir = osp.join(self.data_dir, 'test')

        self.train   , self.train_pid, self.train_camid= self._pluck_msmt(self.train_list, self.train_dir)
        self.val     , self.val_pid, self.val_camid= self._pluck_msmt(self.val_list, self.train_dir)
        self.query   , self.query_pid, self.query_camid= self._pluck_msmt(self.query_list, self.test_dir)
        self.gallery , self.gallery_pid, self.gallery_camid= self._pluck_msmt(self.gallery_list, self.test_dir)

        self.train = self.train + self.val
        self.train_original = self.train
        self.train_pid = self.train_pid + self.val_pid
        self.train_camid = self.train_camid + self.val_camid

        self.print_dataset_statistics(self.train, self.query, self.gallery)

        # self.train, train_pids = self._pluck_msmt(osp.join(exdir, 'list_train.txt'), 'train')
        # self.val, val_pids = self._pluck_msmt(osp.join(exdir, 'list_val.txt'), 'train')
        # self.train = self.train + self.val
        # self.query, query_pids = self._pluck_msmt(osp.join(exdir, 'list_query.txt'), 'test')
        # self.gallery, gallery_pids = self._pluck_msmt(osp.join(exdir, 'list_gallery.txt'), 'test')

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    
    def _pluck_msmt(self, list_file, img_dir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        dataset = []
        pids = []
        camids = []
        new_pids = []
        for line in lines:
            line = line.strip()
            fname = line.split(' ')[0]
            pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
            cam = cam - 1 # start from 0
            if pid not in pids:
                pids.append(pid)
            if 'train' in img_dir:
                 # new index for CAP_master
                this_prefix = osp.basename(fname)
                if this_prefix not in self.all_img_prefix:
                    self.all_img_prefix[this_prefix] = len(self.all_img_prefix)
                img_idx = self.all_img_prefix[this_prefix]  # global index

                dataset.append((osp.join(img_dir,fname), pid, cam, img_idx))
                camids.append(cam)
                new_pids.append(pid)

            else:
                dataset.append((osp.join(img_dir,fname), pid, cam))
                camids.append(cam)
                new_pids.append(pid)
        return dataset, new_pids, camids

    def get_train_data_size(self):
        return self.num_train_imgs


