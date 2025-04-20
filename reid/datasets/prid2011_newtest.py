import os.path as osp
import os
import glob
import re

from ..utils.data import BaseImageDataset


class Prid2011(BaseImageDataset):

    def __init__(self, root, verbose=True, **kwargs):
        super(Prid2011, self).__init__()
        dataset_dir = 'prid_2011/multi_shot'
        self.data_dir = osp.join(root, dataset_dir)

        path_a = '/home/remote/tchsu/prid_statistic/cama_num.txt'
        path_b = '/home/remote/tchsu/prid_statistic/camb_num.txt'
        file_a = open(path_a, 'r')
        file_b = open(path_b, 'r')
        self.camb_num = []
        self.cama_num = []
        for line in file_a:
            self.cama_num.append(int(line.split('\n')[0]))
        for line in file_b:
            self.camb_num.append(int(line.split('\n')[0]))

        self.all_img_prefix = {}

        train1, query1, gallery1, train_pid1, query_pid1, gallery_pid1, train_camid1, query_camid1, gallery_camid1 = self._make_cam0_dir(self.data_dir)
        train2, query2, gallery2, train_pid2, query_pid2, gallery_pid2, train_camid2, query_camid2, gallery_camid2 = self._make_cam1_dir(self.data_dir)
        train = train1 + train2
        query = query1 + query2
        gallery = gallery1 + gallery2
        self.train_pid = train_pid1 + train_pid2
        self.train_camid = train_camid1 + train_camid2
        self.gallery_pid =   gallery_pid1 + gallery_pid2
        self.gallery_camid = gallery_camid1 + gallery_camid2
        self.query_pid =   query_pid1 + query_pid2
        self.query_camid = query_camid1 + query_camid2
        self.train = train
        self.query = query
        self.gallery = gallery
        self.train_original = self.train

        self.print_dataset_statistics(train, query, gallery)

        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    ## dataset_balance ##
    def _make_cam0_dir(self, data_dir):
        cam0_dir_path = osp.join(data_dir, 'cam_a')
        dirs = os.listdir(cam0_dir_path)
        query_dataset = []
        train_dataset = []
        gallery_dataset = []
        query_ids = []
        train_ids = []
        gallery_ids = []
        query_camid = []
        gallery_camid = []
        train_camid = [ ]
        for dir in dirs:
            pid = int(dir.split("_")[1])
            num = 0
            if (pid > 100):
                imgs_path = glob.glob(osp.join(cam0_dir_path, dir, "*.png"))
                max = min(self.camb_num[pid-1], self.cama_num[pid-1])
                # for img_path in imgs_path:
                for i in range(max):
                    try:
                        img_path = imgs_path[i]
                    except:
                        break
                     # new index for CAP_master
                    this_prefix = img_path
                    if this_prefix not in self.all_img_prefix:
                        self.all_img_prefix[this_prefix] = len(self.all_img_prefix)
                    img_idx = self.all_img_prefix[this_prefix]  # global index

                    train_dataset.append((img_path, pid-101, 0, img_idx))
                    train_ids.append(pid-101)
                    train_camid.append(0)
            else:
                imgs_path = glob.glob(osp.join(cam0_dir_path, dir, "*.png"))
                for img_path in imgs_path:
                    if (num==0):
                        query_dataset.append((img_path, pid-1, 0))
                        query_ids.append(pid-1)
                        query_camid.append(0)
                        num += 1
                    elif (num < 35):
                        gallery_dataset.append((img_path, pid-1, 0))
                        gallery_ids.append(pid-1)
                        gallery_camid.append(0)
                        num += 1
                    else:
                        continue
        return train_dataset, query_dataset, gallery_dataset, train_ids, query_ids, gallery_ids, train_camid, query_camid, gallery_camid

    def _make_cam1_dir(self, data_dir):
        cam1_dir_path = osp.join(data_dir, "cam_b")
        dirs = os.listdir(cam1_dir_path)
        train_dataset = []
        gallery_dataset = []
        query_dataset = []
        train_ids = []
        gallery_ids = []
        query_ids = []
        train_camid = []
        gallery_camid = []
        query_camid = []
        for dir in dirs:
            pid = int(dir.split("_")[1])
            num = 0
            if (pid > 100) and (pid <= 200):
                imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
                max = min(self.camb_num[pid-1], self.cama_num[pid-1])
                # for img_path in imgs_path:
                for i in range(max):
                    try:
                        img_path = imgs_path[i]
                    except:
                        break
                     # new index for CAP_master
                    this_prefix = img_path
                    if this_prefix not in self.all_img_prefix:
                        self.all_img_prefix[this_prefix] = len(self.all_img_prefix)
                    img_idx = self.all_img_prefix[this_prefix]  # global index

                    train_dataset.append((img_path, pid-101, 1, img_idx))
                    train_ids.append(pid-101)
                    train_camid.append(1)
            elif (pid > 200) and (pid <= 385):
                imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
                # for img_path in imgs_path:
                max = min(self.camb_num[pid-1], self.cama_num[pid-1])
                # for img_path in imgs_path:
                for i in range(max):
                    try:
                        img_path = imgs_path[i]
                    except:
                        break
                     # new index for CAP_master
                    this_prefix = img_path
                    if this_prefix not in self.all_img_prefix:
                        self.all_img_prefix[this_prefix] = len(self.all_img_prefix)
                    img_idx = self.all_img_prefix[this_prefix]  # global index

                    train_dataset.append((img_path, pid+85, 1, img_idx))
                    train_ids.append(pid+85)
                    train_camid.append(1)
            elif (pid <= 100):
                imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
                for img_path in imgs_path:
                    if (num==0):
                        query_dataset.append((img_path, pid-1, 1))
                        query_ids.append(pid-1)
                        query_camid.append(1)
                        num += 1
                    elif (num < 35):
                        gallery_dataset.append((img_path, pid-1, 1))
                        gallery_ids.append(pid-1)
                        gallery_camid.append(1)
                        num += 1
                    else:
                        continue
            else:
                continue
        return train_dataset, query_dataset, gallery_dataset, train_ids, query_ids, gallery_ids, train_camid, query_camid, gallery_camid

    ### dataset
    # def _make_cam0_dir(self, data_dir):
    #     cam0_dir_path = osp.join(data_dir, 'cam_a')
    #     dirs = os.listdir(cam0_dir_path)
    #     query_dataset = []
    #     train_dataset = []
    #     gallery_dataset = []
    #     query_ids = []
    #     train_ids = []
    #     gallery_ids = []
    #     query_camid = []
    #     gallery_camid = []
    #     train_camid = [ ]
    #     for dir in dirs:
    #         pid = int(dir.split("_")[1])
    #         num = 0
    #         if (pid > 100):
    #             imgs_path = glob.glob(osp.join(cam0_dir_path, dir, "*.png"))
    #             max = min(self.camb_num[pid-1], self.cama_num[pid-1])
    #             # for img_path in imgs_path:
    #             for i in range(max):
    #                 img_path = imgs_path[i]
    #                 # try:
    #                 #     img_path = imgs_path[i]
    #                 # except:
    #                 #     break
    #                 train_dataset.append((img_path, pid-101, 0))
    #                 train_ids.append(pid-101)
    #                 train_camid.append(0)
    #         else:
    #             imgs_path = glob.glob(osp.join(cam0_dir_path, dir, "*.png"))
    #             for img_path in imgs_path:
    #                 if (num==0):
    #                     query_dataset.append((img_path, pid-1, 0))
    #                     query_ids.append(pid-1)
    #                     query_camid.append(0)
    #                     num += 1
    #                 # elif (num < 35):
    #                 #     gallery_dataset.append((img_path, pid-1, 0))
    #                 #     gallery_ids.append(pid-1)
    #                 #     gallery_camid.append(0)
    #                 #     num += 1
    #                 else:
    #                     break
    #                     #continue
    #     return train_dataset, query_dataset, gallery_dataset, train_ids, query_ids, gallery_ids, train_camid, query_camid, gallery_camid

    # def _make_cam1_dir(self, data_dir):
    #     cam1_dir_path = osp.join(data_dir, "cam_b")
    #     dirs = os.listdir(cam1_dir_path)
    #     train_dataset = []
    #     gallery_dataset = []
    #     query_dataset = []
    #     train_ids = []
    #     gallery_ids = []
    #     query_ids = []
    #     train_camid = []
    #     gallery_camid = []
    #     query_camid = []
    #     for dir in dirs:
    #         pid = int(dir.split("_")[1])
    #         num = 0
    #         if (pid > 100) and (pid <= 200):
    #             imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
    #             max = min(self.camb_num[pid-1], self.cama_num[pid-1])
    #             # for img_path in imgs_path:
    #             for i in range(max):
    #                 img_path = imgs_path[i]
    #                 # try:
    #                 #     img_path = imgs_path[i]
    #                 # except:
    #                 #     break
    #                 train_dataset.append((img_path, pid-101, 1))
    #                 train_ids.append(pid-101)
    #                 train_camid.append(1)
    #         elif (pid > 200) and (pid <= 385):
    #             imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
    #             # for img_path in imgs_path:
    #             max = min(self.camb_num[pid-1], self.cama_num[pid-1])
    #             # for img_path in imgs_path:
    #             for i in range(max):
    #                 img_path = imgs_path[i]
    #                 # try:
    #                 #     img_path = imgs_path[i]
    #                 # except:
    #                 #     break
    #                 train_dataset.append((img_path, pid+85, 1))
    #                 train_ids.append(pid-)
    #                 train_camid.append(1)
    #         elif (pid <= 100):
    #             imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
    #             for img_path in imgs_path:
    #                 # if (num==0):
    #                 #     query_dataset.append((img_path, pid-1, 1))
    #                 #     query_ids.append(pid-1)
    #                 #     query_camid.append(1)
    #                 #     num += 1
    #                 # elif (num < 35):
    #                 if (num ==0):
    #                     gallery_dataset.append((img_path, pid-1, 1))
    #                     gallery_ids.append(pid-1)
    #                     gallery_camid.append(1)
    #                     num += 1
    #                 else:
    #                     break
    #                     #continue
    #         else:
    #             # continue
    #             imgs_path = glob.glob(osp.join(cam1_dir_path, dir, "*.png"))
    #             if (num ==0):
    #                     gallery_dataset.append((img_path, pid-1, 1))
    #                     gallery_ids.append(pid-1)
    #                     gallery_camid.append(1)
    #                     num += 1
    #             else:
    #                 break

    #     return train_dataset, query_dataset, gallery_dataset, train_ids, query_ids, gallery_ids, train_camid, query_camid, gallery_camid

    def get_train_data_size(self):
        return self.num_train_imgs




