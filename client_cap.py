import argparse
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bisect import bisect_right
import sys
import re
import numpy as np
import copy
import torch
import time
import torch._utils
from torch.utils.data import DataLoader

from reid.utils.data.preprocessor import Preprocessor, UnsupervisedTargetPreprocessor, ClassUniformlySampler
from reid.utils.data import transforms as T
from reid.loss import CAPMemory, CAPMemory_online, CAPMemory_simple, CAPMemory_part
from reid.models import stb_net
from reid.trainers import Trainer
from reid.utils.evaluation_metrics.retrieval import PersonReIDMAP
from reid.utils.meters import CatMeter
from reid.img_grouping import img_association
# from reid.part_img_grouping import img_association
from reid.supervised_img_grouping import sup_img_association

from easyfl.client.base import BaseClient
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.tracking import metric
from easyfl.tracking.evaluation import model_size
logger = logging.getLogger(__name__)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

class CAPClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0,
                 is_remote=False, local_port=23000, server_addr="localhost:22999", tracker_addr="localhost:12666"):
        super(CAPClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time,
                                                 is_remote, local_port, server_addr, tracker_addr)
        # logger.info(conf)
        self.cid = cid
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        # self.best_local_model = stb_net.MemoryBankModel(out_dim=2048)
        self.epoch_best_model = stb_net.MemoryBankModel(out_dim=2048)
        self.global_model = stb_net.MemoryBankModel(out_dim=2048)
        self.global_weight = 0
        self.data_weight = self.get_data_weight()
        # self.specific_layer_statdict = {k:v for k,v in self.best_local_model.state_dict().items() if re.findall("bottleneck",k)}
        self.train_acc_for_drawing = []
        self.record_acc = []
        self.best_acc = 0
        self.best_mAP = 0
        self.best_loss = 100
        transform_train = [
                T.Resize((256,128)),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop((256,128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.RandomErasing(EPSILON=0.5)
                ]
        transform_val = [
                T.Resize(size=(256,128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
        self.data_transforms = {
            'train': T.Compose(transform_train),
            'val': T.Compose(transform_val),
        }  
        cos_sim_loader = self.get_propagate_loader(conf)
        test_batchs = next(iter(cos_sim_loader))
        self.test_batch = test_batchs[0].to(device)

    def update_train_loader(self, conf, updated_label, proxy_labels, sample_position=7):
        new_train_samples = []
        for sample in self.train_data.data[self.cid]:
            lbl = updated_label[sample[3]]
            # px_lb = proxy_labels[sample[3]]
            if lbl != -1:
                assert(proxy_labels[sample[3]]>=0)
                new_sample = sample + (lbl, int(proxy_labels[sample[3]]))
                new_train_samples.append(new_sample) #(img_path, pid, cid, img_idx, p_label, proxy)
        target_train_loader = DataLoader(
            UnsupervisedTargetPreprocessor(new_train_samples,
                                            num_cam=self.train_data.dataset[self.cid].num_cams, transform=self.data_transforms['train'], has_pseudo_label=True),
            batch_size=conf.batch_size, num_workers=8, pin_memory=True, drop_last=True,
            sampler=ClassUniformlySampler(new_train_samples, class_position=sample_position, k=4))

        return target_train_loader, len(new_train_samples)

    def get_test_loader(self, conf):       
        query_target = f'{self.cid}_query'
        gallery_target = f'{self.cid}_gallery'

        query_loader = DataLoader(
            Preprocessor(self.test_data.data[query_target], transform=self.data_transforms['val']),
            batch_size=conf.batch_size, num_workers=0,
            shuffle=False, pin_memory=True)

        gallery_loader = DataLoader(
            Preprocessor(self.test_data.data[gallery_target], transform=self.data_transforms['val']),
            batch_size=conf.batch_size, num_workers=0,
            shuffle=False, pin_memory=True)
    
        return query_loader, gallery_loader
    
    def get_test_trainset_loader(self, conf):       
        target = f'{self.cid}_trainset'

        testtrain_loader = DataLoader(
            Preprocessor(self.test_data.data[target], transform=self.data_transforms['val']),
            batch_size=conf.batch_size, num_workers=8,
            shuffle=False, pin_memory=True)
    
        return testtrain_loader
    
    def get_propagate_loader(self, conf):
        propagate_loader = DataLoader(
            UnsupervisedTargetPreprocessor(self.train_data.data[self.cid],
                                            num_cam=self.train_data.dataset[self.cid].num_cams, transform=self.data_transforms['val']),
            batch_size=conf.batch_size, num_workers=8,
            shuffle=False, pin_memory=True)

        return propagate_loader
    

    def train(self, conf, device, global_memory_bank, client_ind):
        # current_round = len(self.train_acc_for_drawing)
        current_round = len(self.record_acc)
        logger.info("--------- training -------- cid: {}, round {}".format(self.cid, current_round))
        all_img_cams = torch.tensor(self.train_data.dataset[self.cid].camids)
        # Creat data loaders
        propagate_loader = self.get_propagate_loader(conf)
   
        # Create memory bank
        cap_memory = CAPMemory_simple(device, global_memory_bank, current_round, client_ind,
                               beta=0.07, alpha=0.2, all_img_cams=all_img_cams)
        cap_memory = cap_memory.to(device)
        # update global model
        self.global_model.load_state_dict(self.model.state_dict())
        self.global_model.to(device)
        # BN layer update
        # if current_round >= 6:       
        # self.model.load_state_dict(self.specific_layer_statdict, strict=False)
        # self.model.load_state_dict(self.local_state_dict)
        self.model.to(device)

        # Evaluate before local training
        self.test_model(conf, device, "global", 0, 0)

        # Calculate global feature
        self.global_model.eval()
        with torch.no_grad():
            old_feature = self.global_model(self.test_batch)

        # Optimizer
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            lr = conf.optimizer.lr
            if current_round >= 5:
                lr = lr * 0.1
            weight_decay = conf.optimizer.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.Adam(params)
        if current_round == 19:
            lr_scheduler = WarmupMultiStepLR(optimizer, [20,40], gamma=0.1, warmup_factor=0.01, warmup_iters=5)

        # Trainer
        trainer = Trainer(device, self.model, self.global_model, cap_memory)

        self.global_weight = self.record_model_weight(conf, current_round, -1, self.model)

        # Start training
        epoch_losses = []
        self.best_epoch_acc = 0
        self.best_epoch_mAP= 0
        self.best_epoch_loss = 100
        if current_round == 19:
            ep_num = 50
        else:
            ep_num = conf.local_epoch
        # for epoch in range(conf.local_epoch):
        for epoch in range(ep_num):
            # image grouping
            logger.info('Epoch {} image grouping:'.format(epoch+1))
            
            ### clustereps_bytest
            # if self.cid == "ilids":
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.45, rerank=True, k1=20, k2=6)
            # elif (self.cid == "prid") or ("cuhk03" in self.cid) or (self.cid == "viper") or (self.cid == "3dpes") or (self.cid == "cuhk01"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.4, rerank=True, k1=20, k2=6)
            # else:
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            ### clustereps_bytest_v1
            # if (self.cid == "viper") or (self.cid == "3dpes") or (self.cid == "ilids"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.4, rerank=True, k1=20, k2=6)
            # else:
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            ### clustereps_bytest_v2
            # if (self.cid == "3dpes") or (self.cid == "ilids") or (self.cid == "viper"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.35, rerank=True, k1=20, k2=6)
            # elif (self.cid == "cuhk01"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.35, rerank=True, k1=20, k2=6)
            # else:
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            ### clustereps_bytest_v3
            # if (self.cid == "3dpes") or (self.cid == "ilids") or (self.cid == "viper"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.3, rerank=True, k1=20, k2=6)
            # elif (self.cid == "cuhk01"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.3, rerank=True, k1=20, k2=6)
            # else:
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            ### multiple mpt
            # if self.data_weight > 10:
            #     mpt = conf.cluster.min_samples * (self.data_weight/10)
            # elif self.data_weight != 1:
            #     mpt = conf.cluster.min_samples * (self.data_weight/2)
            # else:
            #     mpt = conf.cluster.min_samples * self.data_weight
            # updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=mpt,
            #                                 eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            ### original dataset clustereps by test
            # if (self.cid == "ilids"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.4, rerank=True, k1=20, k2=6)
            # elif (self.cid == "viper") or (self.cid == "3dpes"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.4, rerank=True, k1=20, k2=6)
            # elif (self.cid == "cuhk01"):
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=0.45, rerank=True, k1=20, k2=6)
            # else:
            #     updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
            #                                     eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            ###### the normal one    
            # st_time = time.time()
            updated_label, init_intra_id_feat, proxy_labels = img_association(self.model, cap_memory, propagate_loader, self.data_weight, device, min_sample=conf.cluster.min_samples,
                                                eps=conf.cluster.thresh, rerank=True, k1=20, k2=6)
            # end_time = time.time()
            # if current_round == 0:
            #     delay = end_time - st_time
            #     logger.info("time: {}".format(delay))
            # elif epoch == (ep_num-1):
            #     delay = end_time - st_time
            #     logger.info("time: {}".format(delay))
            # else:
            #     continue
        
            # update train loader
            new_train_loader, loader_size = self.update_train_loader(conf, updated_label, proxy_labels, sample_position=5)
            num_batch = int(float(loader_size)/conf.batch_size)

            # train an epoch
            epoch_loss = trainer.train(epoch, new_train_loader, optimizer, num_batch=num_batch, init_intra_id_feat=init_intra_id_feat)
            epoch_losses.append(float(epoch_loss))

            self.track(metric.TRAIN_LOSS, epoch_loss)
            if current_round == 19:
                lr_scheduler.step()
            # evaluate
            self.test_model(conf, device, float(epoch_loss), current_round, epoch)
            # self.test_model(conf, device, float(epoch_loss), current_round, epoch)
            # test_trainset = ((self.cid == "prid") or (self.cid == "3dpes") or (self.cid == "ilids") or (self.cid == "viper"))
            # if test_trainset:
            #     self.test_trainset(conf, device)

            # _ = self.record_model_weight(conf, current_round, epoch, self.model)
        

        # save and update before send to global
        # state_dict = self.epoch_best_model.state_dict()
        # self.model.load_state_dict(state_dict)
        # self.model = copy.deepcopy(self.best_local_model)

        # local feature memory bank
        # self.update_memory_bank = cap_memory.percam_tempV.detach().clone()
        self.update_memory_bank = []
        for c in cap_memory.unique_cams:
            self.update_memory_bank.append(cap_memory.percam_memory[int(c)].detach().clone())
        self.update_memory_bank = torch.cat(self.update_memory_bank, dim=0) # real memorybank looks like the picture in the paper (num of proxy, 2048)
        # BN Layer
        # self.specific_layer_statdict = {k:v for k,v in state_dict.items() if re.findall("bottleneck",k)}
        
        # output figure and record
        self.train_accuracy = self.best_acc
        # self.train_acc_for_drawing.append(self.best_acc*100)
        self.record_acc.append(self.best_epoch_acc*100)
        self.draw_curve(conf)
        if current_round == 19:
            logger.info("Last Round Epoch Best --> mAP: {:4.2%}, Rank 1: {:4.2%}".format(self.best_epoch_mAP, self.best_epoch_acc))
            path = f"{conf.save_figure_path}/saved_model/{self.cid}"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.epoch_best_model.state_dict(), os.path.join(path, "best_local_model.pth"))

        # Get cosine similarity weight   
        self.model.eval()  
        with torch.no_grad():
            new_feature = self.model(self.test_batch)
        
        self.distance = 1 - torch.cosine_similarity(old_feature, new_feature)
        self.cos_distance = int(torch.mean(self.distance) * 1000)
        logger.info("cid: {}, cosine distance similarity {}".format(self.cid, torch.mean(self.distance)*100))

        # Get the difference between global model and local model by layer
        # self.model_diff(conf, current_round, self.model.state_dict(), self.global_model.state_dict())

    def model_diff(self, conf, current_round, local_param, global_param):
        diff_dict = {}
        for lp, gp in zip(local_param.items(), global_param.items()):
            assert lp[0] == gp[0]
            if ("weight" in lp[0]) or ("bias" in lp[0]):
                root = f"{conf.save_figure_path}/model_diff_record/{self.cid}"
                if not os.path.exists(root):
                    os.makedirs(root)
                path = f"{root}/{lp[0]}.txt"
                f = open(path, 'a')
                diff = (lp[1] - gp[1]).norm(2)
                diff_dict[lp[0]] = diff
                f.write(f"{str(diff)}\n")
                f.close()

        diff_dict = {k: v for k, v in sorted(diff_dict.items(), key=lambda item: item[1])}
        path = f"{conf.save_figure_path}/model_diff_record/{self.cid}/max_min.txt"
        f = open(path, 'a')
        f.write(f"round {current_round}\n")
        f.write(f"maximum: {list(diff_dict)[:10]}\n")
        f.write(f"minimum: {list(diff_dict)[-10:]}\n")
        f.close()

    def record_model_weight(self, conf, round, epoch, model):
        root = f"{conf.save_figure_path}/model_weight_record"
        if not os.path.exists(root):
            os.makedirs(root)
        path = os.path.join(root, f"{self.cid}.txt")
        f = open(path, 'a')
        wt_sum = 0
        for wt in model.parameters():
            weight = wt.data
            wt_sum += weight.norm(2)

        diff = abs(wt_sum - self.global_weight)

        if epoch == -1:
            f.write(f"round {round}  global_weight {str(wt_sum)}\n")  
            f.close()
        else:
            f.write(f"\tepoch {epoch}  wt {str(wt_sum)} diff {diff}\n")  
            f.close()

        return wt_sum


    def run_train(self, model, conf, global_memory_bank, client_ind):
        """Conduct training on clients.

        Args:
            model (nn.Module): Model to train.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            :obj:`UploadRequest`: Training contents. Unify the interface for both local and remote operations.
        """
        self.conf = conf
        if conf.track:
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid)

        self._is_train = True

        self.download(model)
        self.track(metric.TRAIN_DOWNLOAD_SIZE, model_size(model))

        self.decompression()

        self.pre_train()
        self.train(conf, self.device, global_memory_bank, client_ind)
        self.post_train()

        if conf.local_test:
            self.test_local(conf, self.device)

        self.track(metric.TRAIN_ACCURACY, self.train_accuracy)
        self.track(metric.TRAIN_LOSS, self.train_loss)
        self.track(metric.TRAIN_TIME, self.train_time)

        self.compression()

        self.track(metric.TRAIN_UPLOAD_SIZE, model_size(self.compressed_model))

        self.encryption()

        # return self.upload(), self.cos_distance
        return self.upload(), self.update_memory_bank, self.cos_distance

    def test(self, conf, device):
        logger.info("Evaluating_Test {}".format(self.cid))
        self.model.eval()
        self.model.to(device)

        # meters
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

        # init dataset
        query_loader, gallery_loader = self.get_test_loader(conf)
        loaders = [query_loader, gallery_loader]

        # compute query and gallery features
        with torch.no_grad():
            for loader_id, loader in enumerate(loaders):
                for data in loader:
                    images = data[0].to(device)
                    pids = data[2]
                    cids = data[3]
                    features = self.model(images)
                    # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)

        query_features = query_features_meter.get_val_numpy()
        gallery_features = gallery_features_meter.get_val_numpy()

        # compute mAP and rank@k
        result = PersonReIDMAP(
            query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
            gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy(), dist='cosine')

        self.test_accuracy = result.CMC[0]
        self._upload_holder = server_pb.UploadContent(
            data=codec.marshal(server_pb.Performance(accuracy=result.CMC[0], loss=0)),  # loss not applicable
            type=common_pb.DATA_TYPE_PERFORMANCE,
            data_size=len(self.test_data.data[f"{self.cid}_query"]),
        )
        logger.info("mAP: {:4.2%}, Rank 1: {:4.2%}".format(result.mAP, result.CMC[0]))
        
    def test_model(self, conf, device, loss, round, ep):
        # logger.info("Evaluating_Train {}".format(self.cid))
        self.model.eval()
        self.model.to(device)

        # meters
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

        # init dataset
        query_loader, gallery_loader = self.get_test_loader(conf)
        loaders = [query_loader, gallery_loader]

        # compute query and gallery features    
        with torch.no_grad():
            for loader_id, loader in enumerate(loaders):
                for data in loader:
                    images = data[0].to(device)
                    pids = data[2]
                    cids = data[3]
                    features = self.model(images)
                    # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)


        query_features = query_features_meter.get_val_numpy()
        gallery_features = gallery_features_meter.get_val_numpy()

        # compute mAP and rank@k
        result = PersonReIDMAP(
            query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
            gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy(), dist='cosine')

        del query_features_meter, gallery_features_meter
        del query_features, gallery_features

        if (round != 19) and (ep == 4):
            self.train_acc_for_drawing.append(result.CMC[0] * 100)
        if (round == 19) and (ep == 49):
            self.train_acc_for_drawing.append(result.CMC[0] * 100)           
        
        if loss == "global":
            logger.info("Initial model --> mAP: {:4.2%}, Rank 1: {:4.2%}".format(result.mAP, result.CMC[0]))
        else:
            # local best from rank1 acc
            is_best = result.CMC[0] >= self.best_acc
            is_epoch_best = result.CMC[0] >= self.best_epoch_acc
            if is_best:
                # self.best_local_model = copy.deepcopy(self.model)
                # path = f"{conf.save_figure_path}/saved_model/{self.cid}"
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # torch.save(self.best_local_model.state_dict(), os.path.join(path, "best_lcoal_model.pth"))
                self.best_acc = result.CMC[0]
                self.best_mAP = result.mAP
            if is_epoch_best:
                self.best_epoch_acc = result.CMC[0]
                self.best_epoch_mAP = result.mAP
                self.epoch_best_model.load_state_dict(self.model.state_dict())

            # if ((round+1) % 5) == 0:
            #     path = f"{conf.save_figure_path}/saved_model/{self.cid}/lcoal_model_r{round}.pth"
            #     torch.save(self.model.state_dict(), path)
            logger.info("\tmAP: {:4.2%}, Rank 1: {:4.2%}, Loss: {:4.2%}".format(result.mAP, result.CMC[0], loss))


    def test_trainset(self, conf, device):
        self.model.eval()
        self.model.to(device)

        # meters
        query_train_features_meter, query_train_pids_meter, query_train_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_train_features_meter, gallery_train_pids_meter, gallery_train_cids_meter = CatMeter(), CatMeter(), CatMeter()

        # init dataset
        query_train_loader = self.get_test_trainset_loader(conf)
        gallery_train_loader = self.get_test_trainset_loader(conf)
        loaders = [query_train_loader, gallery_train_loader]

        # compute query_train and gallery_train features
        with torch.no_grad():
            for loader_id, loader in enumerate(loaders):
                for data in loader:
                    images = data[0].to(device)
                    pids = data[2]
                    cids = data[3]
                    features = self.model(images)
                    # save as query_train features
                    if loader_id == 0:
                        query_train_features_meter.update(features.data)
                        query_train_pids_meter.update(pids)
                        query_train_cids_meter.update(cids)
                    # save as gallery_train features
                    elif loader_id == 1:
                        gallery_train_features_meter.update(features.data)
                        gallery_train_pids_meter.update(pids)
                        gallery_train_cids_meter.update(cids)

        query_train_features = query_train_features_meter.get_val_numpy()
        gallery_train_features = gallery_train_features_meter.get_val_numpy()

        # compute mAP and rank@k
        result = PersonReIDMAP(
            query_train_features, query_train_cids_meter.get_val_numpy(), query_train_pids_meter.get_val_numpy(),
            gallery_train_features, gallery_train_cids_meter.get_val_numpy(), gallery_train_pids_meter.get_val_numpy(), dist='cosine')

        del query_train_features_meter, gallery_train_features_meter
        del query_train_features, gallery_train_features

        logger.info("\ttrainset --> mAP: {:4.2%}, Rank 1: {:4.2%}".format(result.mAP, result.CMC[0]))
                

    def draw_curve(self, conf):    
        # Draw acc for each round
        plt.figure()
        x_epoch = list(range(len(self.train_acc_for_drawing)))
        plt.plot(x_epoch, self.train_acc_for_drawing, 'ro-', label='Rank1 Acc')
        plt.title(f"{self.cid}_lastepoch")
        for i, acc in enumerate(self.train_acc_for_drawing):
            plt.text(i, acc, round(acc, 2))
        plt.legend()
        plt.savefig(os.path.join(conf.save_figure_path, f'{self.cid}_lastep.png'))
        plt.close('all')

        # Draw epoch best acc
        plt.figure()
        x_epoch = list(range(len(self.record_acc)))
        plt.plot(x_epoch, self.record_acc, 'ro-', label='Rank1 Acc')
        plt.title(f"{self.cid}_epochbest")
        for i, acc in enumerate(self.record_acc):
            plt.text(i, acc, round(acc, 2))
        plt.legend()
        plt.savefig(os.path.join(conf.save_figure_path, f'{self.cid}_epochbest.png'))
        plt.close('all')

    def get_data_weight(self):
        if self.cid == "ilids":
            weight = 60
        elif self.cid == "prid":
            weight = 4
        elif self.cid == "viper":
            weight = 20
        elif self.cid == "cuhk01":
            weight = 6
        elif self.cid == "3dpes":
            weight = 30
        elif "cuhk03" in self.cid:
            weight = 2
        else:
            weight = 1
        
        return weight