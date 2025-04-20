from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import shutil
import os
from ..utils.data import BaseImageDataset
# from ..utils.osutils import mkdir_if_missing
# from ..utils.serialization import write_json

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # self._check_before_run()

        train, self.train_pid, self.train_camid = self._process_dir(self.train_dir, relabel=True)
        query, self.query_pid, self.query_camid = self._process_dir(self.query_dir, relabel=False)
        gallery, self.gallery_pid, self.gallery_camid = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            # print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.train_original = self.train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    # def _check_before_run(self):
    #     """Check if all files are available before going deeper"""
    #     if not osp.exists(self.dataset_dir):
    #         raise RuntimeError("'{}' is not available".format(self.dataset_dir))
    #     if not osp.exists(self.train_dir):
    #         raise RuntimeError("'{}' is not available".format(self.train_dir))
    #     if not osp.exists(self.query_dir):
    #         raise RuntimeError("'{}' is not available".format(self.query_dir))
    #     if not osp.exists(self.gallery_dir):
    #         raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        pids = []
        camids= []
        all_img_prefix = {}
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: 
                pid = pid2label[pid]
                # new index for CAP_master
                this_prefix = osp.basename(img_path)
                if this_prefix not in all_img_prefix:
                    all_img_prefix[this_prefix] = len(all_img_prefix)
                img_idx = all_img_prefix[this_prefix]  # global index

                dataset.append((img_path, pid, camid, img_idx))
                pids.append(pid)
                camids.append(camid)
            else:
                dataset.append((img_path, pid, camid))
                pids.append(pid)
                camids.append(camid)

        return dataset, pids, camids

    def get_train_data_size(self):
        return self.num_train_imgs