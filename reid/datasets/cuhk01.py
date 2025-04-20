import os.path as osp
import os
import glob
import numpy as np
import re

from ..utils.data import BaseImageDataset


class CUHK01(BaseImageDataset):

    def __init__(self, root, verbose=True, **kwargs):
        super(CUHK01, self).__init__()
        dataset_dir = 'cuhk01'
        self.data_dir = osp.join(root, dataset_dir)
        
        train, query, gallery, self.train_pid, self.train_camid, self.query_pid, self.query_camid, self.gallery_pid, self.gallery_camid = self._process_dir(self.data_dir)

        self.train = train
        self.train_original = self.train
        self.query = query
        self.gallery = gallery

        self.print_dataset_statistics(train, query, gallery)

        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir):
        pattern = re.compile(r'(\d{4})(\d{3})')
        trainset = []
        queryset = []
        galleryset = []
        train_pids = []
        train_camids = []
        query_pids = []
        query_camids = []
        gallery_pids = []
        gallery_camids = []
        
        all_img_prefix = {}

        for name in glob.glob(os.path.join(data_dir, '*.png')):
            pid = pattern.search(name).group(1)
            idx = pattern.search(name).group(2)
            pid = int(np.array(pid))
            idx = int(np.array(idx))
            img_path = name

            if (idx < 3): camid = 0
            else: camid = 1

            if pid <= 485:
                # new index for CAP_master
                this_prefix = osp.basename(img_path)
                if this_prefix not in all_img_prefix:
                    all_img_prefix[this_prefix] = len(all_img_prefix)
                img_idx = all_img_prefix[this_prefix]  # global index
                trainset.append((img_path, pid-1, camid, img_idx))
                train_pids.append(pid-1)
                train_camids.append(camid)
            elif ((idx == 1) or (idx == 2)):
                queryset.append((img_path, pid, camid))
                query_pids.append(pid)
                query_camids.append(camid)
            else:
                galleryset.append((img_path, pid, camid))
                gallery_pids.append(pid)
                gallery_camids.append(camid)
                
                
        return trainset, queryset, galleryset, train_pids, train_camids, query_pids, query_camids, gallery_pids, gallery_camids

    def get_train_data_size(self):
        return self.num_train_imgs



