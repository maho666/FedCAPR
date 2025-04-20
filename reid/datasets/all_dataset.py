import glob
import re
import os
from pathlib import Path
from torch.utils.data.dataset import Dataset

class All_DS(Dataset):
    def __init__(self, target, root, **kwargs):
        super(All_DS, self).__init__()
        self.target = target
        if "_" in self.target:
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
        
        
    