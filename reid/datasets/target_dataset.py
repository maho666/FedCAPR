import glob
import re
import os
from pathlib import Path
from torch.utils.data.dataset import Dataset
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .prid2011_newtest import Prid2011
from .cuhk03 import CUHK03
from .ilids import iLIDS
from .viper import Viper
from .cuhk01 import CUHK01

class DS(Dataset):
    def __init__(self, target, root, **kwargs):
        super(DS, self).__init__()
        self.target = target
        if "_" in self.target :
            self.data, self.camids = self._process_dir(root, relabel=False)
        else:
            self.data, self.camids = self._process_dir(root, relabel=True)

        self.num_pids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)
        
    def _process_dir(self, dir_path, relabel=False):
        dataset = []
        camids = []
        is_special = ('cuhk03-np-detected' in self.target) or ('Duke' in self.target) or ('Market' in self.target) or ('mamt17' in self.target)

        if is_special:
            pattern = re.compile(r'_c(\d)')
        else:
            pattern = re.compile(r'cam_(\d)')

        '''
        # append to 20
        if relabel:
            if "ilids" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            # for i in range(60):
                            for i in range(5):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "cuhk03-np-detected" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            # print(self.target, pattern, img_path)
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(2):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "prid" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            # print(self.target, pattern, img_path)
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(2):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "viper" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            # for i in range(20):
                            for i in range(10):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "cuhk01" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            # for i in range(6):
                            for i in range(5):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "3dpes" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            # print(self.target, pattern, img_path)
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            # for i in range(30):
                            for i in range(4):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            else: 
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            if relabel:
                                dataset.append((img_path, pid, camid, img_idx))
                                camids.append(camid)
                                img_idx += 1
                            else:
                                dataset.append((img_path, pid, camid))
                                camids.append(camid)
                    else:
                        pass
        else:
            img_idx = 0
            for pid_dir in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, pid_dir)):
                    img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                    for img_path in img_paths:
                        camid = int(pattern.search(img_path).groups()[0])
                        if is_special:
                            camid -= 1
                        pid = int(pid_dir)
                        if relabel:
                            dataset.append((img_path, pid, camid, img_idx))
                            camids.append(camid)
                            img_idx += 1
                        else:
                            dataset.append((img_path, pid, camid))
                            camids.append(camid)
                else:
                    pass
        '''
        # # append to 25
        # if relabel:
        #     if "ilids" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(6):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk03-np-detected" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(2):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "prid" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(2):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "viper" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(12):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk01" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(6):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "3dpes" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(5):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     else: 
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     if relabel:
        #                         dataset.append((img_path, pid, camid, img_idx))
        #                         camids.append(camid)
        #                         img_idx += 1
        #                     else:
        #                         dataset.append((img_path, pid, camid))
        #                         camids.append(camid)
        #             else:
        #                 pass
        # else:
        #     img_idx = 0
        #     for pid_dir in os.listdir(dir_path):
        #         if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #             img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #             for img_path in img_paths:
        #                 camid = int(pattern.search(img_path).groups()[0])
        #                 if is_special:
        #                     camid -= 1
        #                 pid = int(pid_dir)
        #                 if relabel:
        #                     dataset.append((img_path, pid, camid, img_idx))
        #                     camids.append(camid)
        #                     img_idx += 1
        #                 else:
        #                     dataset.append((img_path, pid, camid))
        #                     camids.append(camid)
        #         else:
        #             pass

        '''# append to 15
        if relabel:
            if "ilids" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(4):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "cuhk03-np-detected" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            # print(self.target, pattern, img_path)
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(1):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "prid" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            # print(self.target, pattern, img_path)
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(1):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "viper" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(8):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "cuhk01" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(4):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            elif "3dpes" in self.target:
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            for i in range(4):
                                if relabel:
                                    dataset.append((img_path, pid, camid, img_idx))
                                    camids.append(camid)
                                    img_idx += 1
                                else:
                                    dataset.append((img_path, pid, camid))
                                    camids.append(camid)
                    else:
                        pass
            else: 
                img_idx = 0
                for pid_dir in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, pid_dir)):
                        img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                        for img_path in img_paths:
                            camid = int(pattern.search(img_path).groups()[0])
                            if is_special:
                                camid -= 1
                            pid = int(pid_dir)
                            if relabel:
                                dataset.append((img_path, pid, camid, img_idx))
                                camids.append(camid)
                                img_idx += 1
                            else:
                                dataset.append((img_path, pid, camid))
                                camids.append(camid)
                    else:
                        pass
        else:
            img_idx = 0
            for pid_dir in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, pid_dir)):
                    img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                    for img_path in img_paths:
                        camid = int(pattern.search(img_path).groups()[0])
                        if is_special:
                            camid -= 1
                        pid = int(pid_dir)
                        if relabel:
                            dataset.append((img_path, pid, camid, img_idx))
                            camids.append(camid)
                            img_idx += 1
                        else:
                            dataset.append((img_path, pid, camid))
                            camids.append(camid)
                else:
                    pass
        '''
        # # append to 40
        # if relabel:
        #     if "ilids" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     # for i in range(60):
        #                     for i in range(10):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk03-np-detected" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(4):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "prid" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(3):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "viper" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     # for i in range(20):
        #                     for i in range(20):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk01" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     # for i in range(6):
        #                     for i in range(10):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "3dpes" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     # for i in range(30):
        #                     for i in range(9):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     else: 
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     if relabel:
        #                         dataset.append((img_path, pid, camid, img_idx))
        #                         camids.append(camid)
        #                         img_idx += 1
        #                     else:
        #                         dataset.append((img_path, pid, camid))
        #                         camids.append(camid)
        #             else:
        #                 pass
        # # append to 30
        # if relabel:
        #     if "ilids" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(7):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk03-np-detected" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(3):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "prid" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(2):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "viper" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(15):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk01" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(7):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "3dpes" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(7):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     else: 
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     if relabel:
        #                         dataset.append((img_path, pid, camid, img_idx))
        #                         camids.append(camid)
        #                         img_idx += 1
        #                     else:
        #                         dataset.append((img_path, pid, camid))
        #                         camids.append(camid)
        #             else:
        #                 pass
    
        # append to 10
        # if relabel:
        #     if "ilids" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(3):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk03-np-detected" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(2):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     # elif "prid" in self.target:
        #     #     img_idx = 0
        #     #     for pid_dir in os.listdir(dir_path):
        #     #         if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #     #             img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #     #             for img_path in img_paths:
        #     #                 # print(self.target, pattern, img_path)
        #     #                 camid = int(pattern.search(img_path).groups()[0])
        #     #                 if is_special:
        #     #                     camid -= 1
        #     #                 pid = int(pid_dir)
        #     #                 for i in range(2):
        #     #                     if relabel:
        #     #                         dataset.append((img_path, pid, camid, img_idx))
        #     #                         camids.append(camid)
        #     #                         img_idx += 1
        #     #                     else:
        #     #                         dataset.append((img_path, pid, camid))
        #     #                         camids.append(camid)
        #     #         else:
        #     #             pass
        #     elif "viper" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(5):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "cuhk01" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(3):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     elif "3dpes" in self.target:
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     # print(self.target, pattern, img_path)
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     for i in range(2):
        #                         if relabel:
        #                             dataset.append((img_path, pid, camid, img_idx))
        #                             camids.append(camid)
        #                             img_idx += 1
        #                         else:
        #                             dataset.append((img_path, pid, camid))
        #                             camids.append(camid)
        #             else:
        #                 pass
        #     else: 
        #         img_idx = 0
        #         for pid_dir in os.listdir(dir_path):
        #             if os.path.isdir(os.path.join(dir_path, pid_dir)):
        #                 img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
        #                 for img_path in img_paths:
        #                     camid = int(pattern.search(img_path).groups()[0])
        #                     if is_special:
        #                         camid -= 1
        #                     pid = int(pid_dir)
        #                     if relabel:
        #                         dataset.append((img_path, pid, camid, img_idx))
        #                         camids.append(camid)
        #                         img_idx += 1
        #                     else:
        #                         dataset.append((img_path, pid, camid))
        #                         camids.append(camid)
        #             else:
        #                 pass
    
        img_idx = 0
        for pid_dir in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, pid_dir)):
                img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                for img_path in img_paths:
                    camid = int(pattern.search(img_path).groups()[0])
                    if is_special:
                        camid -= 1
                    pid = int(pid_dir)
                    if relabel:
                        dataset.append((img_path, pid, camid, img_idx))
                        camids.append(camid)
                        img_idx += 1
                    else:
                        dataset.append((img_path, pid, camid))
                        camids.append(camid)
            else:
                pass
                
        return dataset, camids
    
    
    def get_imagedata_info(self, data):
        pids, cams = [], []
        try:
            for _, pid, camid in data:
                pids += [pid]
                cams += [camid]
        except:
            for _, pid, camid, _ in data:
                pids += [pid]
                cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
    
class old_DS(Dataset):
    def __init__(self, target, root, **kwargs):
        super(old_DS, self).__init__()
        self.target = target
        #root = "/home/remote/tchsu/ICE_fed/examples/data/"
        if ('cuhk03-np-detected' in self.target):
            dataset = CUHK03(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams
            
        elif('Duke' in self.target):
            dataset = DukeMTMC(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams
            
            self.num_pids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)
        elif('Market' in self.target):
            dataset = Market1501(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams
            

            self.num_pids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)
        elif('ilids' in self.target):
            dataset = iLIDS(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams
            

            self.num_pids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)
        elif('cuhk01' in self.target):
            dataset = CUHK01(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams
            
            self.num_pids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)
        elif('prid' in self.target):
            dataset = Prid2011(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams
            
        elif('viper' in self.target):
            dataset = Viper(root)
            if "query" in self.target:
                self.data, self.camids = dataset.query, dataset.query_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams
            elif "gallery" in self.target:
                self.data, self.camids = dataset.gallery, dataset.gallery_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams
            else:
                self.data, self.camids = dataset.train, dataset.train_camid
                self.num_pids, self.num_imgs, self.num_cams = dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams           
        elif('3dpes' in self.target):
            train_root = "/home/tchsu/ICE_fed/examples/processed_data/3dpes/pytorch/train_all/"
            query_root = "/home/tchsu/ICE_fed/examples/processed_data/3dpes/pytorch/query/"
            gallery_root = "/home/tchsu/ICE_fed/examples/processed_data/3dpes/pytorch/gallery/"
            if "query" in self.target:
                self.data, self.camids = self._process_dir(query_root, relabel=False)
            elif "gallery" in self.target:
                self.data, self.camids = self._process_dir(gallery_root, relabel=False)
            else:
                self.data, self.camids = self._process_dir(train_root, relabel=True)
            self.num_pids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)
        else:
            pass
        
    def _process_dir(self, dir_path, relabel=False):
        dataset = []
        camids = []
        is_special = ('cuhk03-np-detected' in self.target) or ('Duke' in self.target) or ('Market' in self.target) or ('mamt17' in self.target)

        if is_special:
            pattern = re.compile(r'_c(\d)')
        else:
            pattern = re.compile(r'cam_(\d)')
        
        img_idx = 0
        for pid_dir in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, pid_dir)):
                img_paths = glob.glob(f"{dir_path}/{pid_dir}/*")
                for img_path in img_paths:
                    # print(self.target, pattern, img_path)
                    camid = int(pattern.search(img_path).groups()[0])
                    if is_special:
                        camid -= 1
                    pid = int(pid_dir)
                    if relabel:
                        # CAP Dataset
                        dataset.append((img_path, pid, camid, img_idx))
                        # O2CAP Dataset
                        # dataset.append((img_path, pid, camid))
                        
                        camids.append(camid)
                        img_idx += 1
                    else:
                        dataset.append((img_path, pid, camid))
                        camids.append(camid)
            else:
                pass
        return dataset, camids
    
    def get_imagedata_info(self, data):
        pids, cams = [], []
        try:
            for _, pid, camid in data:
                pids += [pid]
                cams += [camid]
        except:
            for _, pid, camid, _ in data:
                pids += [pid]
                cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams
        
        
    