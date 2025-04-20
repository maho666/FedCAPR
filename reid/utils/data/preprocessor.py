from __future__ import absolute_import
import os.path as osp
from PIL import Image
from torchvision.transforms import functional as F
import torch
import torch.utils.data as data
import random


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        #fname, pid, camid = self.dataset[index]
        single_data = self.dataset[index]
        fname, pid, camid = single_data[0], single_data[1], single_data[2]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


class SourcePreprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(SourcePreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        #fname, pid, camid = self.dataset[index]
        fname, pid, camid, img_idx, accum_label = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, img_idx, accum_label


class UnsupervisedTargetPreprocessor(object):
    def __init__(self, dataset, root=None, num_cam=6, transform=None, has_pseudo_label=False):
        super(UnsupervisedTargetPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.num_cam = num_cam
        self.has_pseudo_label = has_pseudo_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.has_pseudo_label:
            fname, pid, camid, img_idx, pseudo_label, accum_label = self.dataset[index]
        else:
            fname, pid, camid, img_idx = self.dataset[index]

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # # 2 part split
        # img1 = img[:, :128, :]
        # img2 = img[:, 128:, :]

        # if self.has_pseudo_label:
        #     return img1, img2, fname, pid, img_idx, camid, pseudo_label, accum_label
        # else:
        #     return img1, img2, fname, pid, img_idx, camid
        if self.has_pseudo_label:
            return img, fname, pid, img_idx, camid, pseudo_label, accum_label
        else:
            return img, fname, pid, img_idx, camid


class ClassUniformlySampler(data.sampler.Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''
    def __init__(self, samples, class_position, k, has_outlier=False, cam_num=0):

        self.samples = samples
        self.class_position = class_position
        self.k = k
        self.has_outlier = has_outlier
        self.cam_num = cam_num
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        id_dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]   # from which index to obtain the label
            if class_index not in list(id_dict.keys()):
                id_dict[class_index] = [index]
            else:
                id_dict[class_index].append(index)
        return id_dict

    def _generate_list(self, id_dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''
        sample_list = []

        dict_copy = id_dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        outlier_cnt = 0
        for key in keys:
            value = dict_copy[key]
            if self.has_outlier and len(value)<=self.cam_num:
                random.shuffle(value)
                sample_list.append(value[0])  # sample outlier only one time
                outlier_cnt += 1
            elif len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k    # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
        if outlier_cnt > 0:
            print('in Sampler: outlier number= {}'.format(outlier_cnt))
        return sample_list


class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

