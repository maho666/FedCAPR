import os.path as osp
import os
import glob
import re

from ..utils.data import BaseImageDataset


class iLIDS(BaseImageDataset):

    def __init__(self, root, verbose=True, **kwargs):
        super(iLIDS, self).__init__()
        dataset_dir = 'i-LIDS-VID/sequences'
        self.data_dir = osp.join(root, dataset_dir)
        c1_path = os.path.join(self.data_dir, 'cam1')
        c2_path = os.path.join(self.data_dir, 'cam2')

        train, self.train_pid, self.train_camid = self._make_train(c1_path, c2_path)
        query, self.query_pid, self.query_camid= self._make_query(c1_path)
        gallery, self.gallery_pid, self.gallery_camid= self._make_gallery(c2_path)

        self.train = train
        self.train_original = self.train
        self.query = query
        self.gallery = gallery
        self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _make_train(self, c1_path, c2_path):
        pattern = re.compile(r'(\d\d\d)')
        pid_container = set()
        dataset = []
        pids = []
        camids = []
        c_path = [c1_path, c2_path]
        camid = 0
        all_img_prefix = {}
        for path in c_path:
            for dir in sorted(os.listdir(path)):
                pid = pattern.search(dir).groups()
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            
            for dir in sorted(os.listdir(path)):    
                pid = pattern.search(dir).groups()
                pid = pid2label[pid]
                if pid < 200:
                    img_paths = glob.glob(os.path.join(path, dir, '*.png'))
                    for img_path in img_paths:
                        # new index for CAP_master
                        this_prefix = img_path
                        if this_prefix not in all_img_prefix:
                            all_img_prefix[this_prefix] = len(all_img_prefix)
                        img_idx = all_img_prefix[this_prefix]  # global index

                        dataset.append((img_path, pid, camid, img_idx))
                        pids.append(pid)
                        camids.append(camid)
            camid += 1

        return dataset, pids, camids
    
    def _make_query(self, c1_path):
        pattern = re.compile(r'(\d\d\d)')
        dataset = []
        pids = []
        camids= []
        pid_container = set()
        for dir in sorted(os.listdir(c1_path)):
            pid = pattern.search(dir).groups()
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
            
        for dir in sorted(os.listdir(c1_path)):
            pid = pattern.search(dir).groups()
            pid = pid2label[pid]
            if pid >= 200:
                img_paths = glob.glob(os.path.join(c1_path, dir, '*.png'))
                for img_path in img_paths:
                    dataset.append((img_path, pid, 0))
                    pids.append(pid)
                    camids.append(0)
        return dataset, pids, camids
        
    def _make_gallery(self, c2_path):
        pattern = re.compile(r'(\d\d\d)')
        dataset = []
        pids = []
        camids= []  
        pid_container = set()
        for dir in sorted(os.listdir(c2_path)):
            pid = pattern.search(dir).groups()
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
            
        for dir in sorted(os.listdir(c2_path)):
            pid = pattern.search(dir).groups()
            pid = pid2label[pid]
            if pid >= 200:
                img_paths = glob.glob(os.path.join(c2_path, dir, '*.png'))
                for img_path in img_paths:
                    dataset.append((img_path, pid, 1))
                    pids.append(pid)
                    camids.append(1)

        return dataset, pids, camids

    def get_train_data_size(self):
        return self.num_train_imgs
        
                    


            
