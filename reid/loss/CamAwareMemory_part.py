from xml.dom import INDEX_SIZE_ERR
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
import copy
import logging
import sys
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


class ExemplarMemory(Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


class CAPMemory_part(nn.Module):
    def __init__(self, device, global_memory_bank, current_round=0, client_ind=0, beta=0.05, alpha=0.01, all_img_cams='', crosscam_epoch=0, bg_knn=50):
        super(CAPMemory_part, self).__init__()
        self.device = device
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature factor
        self.all_img_cams = all_img_cams.to(device)
        self.unique_cams = torch.unique(self.all_img_cams)
        self.all_pseudo_label = ''
        self.crosscam_epoch = crosscam_epoch
        self.bg_knn = bg_knn
        self.global_memory_bank = copy.deepcopy(global_memory_bank)
        self.client_memory_lenth = 0
        logger.info("check Gti2 online")
        self.used_local_negative = 0
        self.local_negative = 0
        self.denominator = 0
        self.proxy_label_dict = {}
        self.proxy_cam_dict = {}

        if current_round > 0:
            logger.info("client_index: {}".format(client_ind))
            del self.global_memory_bank[client_ind]
            self.global_memory_bank = torch.cat(self.global_memory_bank).to(device)
            self.other_memory_lenth = self.global_memory_bank.shape[0]
            # logger.info("other_memory_lenth: {}".format(self.other_memory_lenth))

        self.current_round = current_round
    
    def forward(self, features, global_features, plabel, proxy, cams=None, epoch=None,
                batch_ind=-1, init_intra_id_feat=''):

        loss = torch.tensor([0.]).to(self.device)
        self.init_intra_id_feat = init_intra_id_feat

        loss = self.loss_using_pseudo_percam_proxy(features, global_features, plabel, proxy, cams, batch_ind, epoch)

        return loss


    def loss_using_pseudo_percam_proxy(self, features, global_features, plabel, proxy, cams, batch_ind, epoch):
        ## set 6 (set 1 + online) should compare with GTi2 base_online_moreneg ###
        ## (set6(online)offline associated loss + offline online loss with same coding sytle)###
        ############################################################################
        ## o2cap set : associate loss use offline(before update) memory ############ 
        ##             online loss use online(updated) memory           ############ 
        ############################################################################
        if batch_ind == 0:
            # initialize proxy memory
            self.percam_memory = []
            self.client_memory_lenth = 0
            self.cam_proxy_num = [0]
            for cc in self.unique_cams:
                if len(self.init_intra_id_feat) > 0:
                    proto_memory = self.init_intra_id_feat[cc]
                    proto_memory = proto_memory.to(self.device)
                    self.percam_memory.append(proto_memory.detach()) # len()=cam_num, percam_memroybank
                    self.client_memory_lenth += proto_memory.shape[0]
                    self.cam_proxy_num.append(self.client_memory_lenth)
            logger.info("client_memory_lenth: {}".format(self.client_memory_lenth))
        
        percam_tempV = []
        for ii in self.unique_cams:
            percam_tempV.append(self.percam_memory[ii].detach().clone()) #(num of label, 2048)
        percam_tempV = torch.cat(percam_tempV, dim=0).to(self.device) # real memorybank looks like the picture in the paper (num of proxy, 2048)

        loss = torch.tensor([0.]).to(self.device)
        # calculate from data in a batch
        for cc in torch.unique(cams):
            inds = torch.nonzero(cams == cc).squeeze(-1) # index of traget camera
            percam_targets = plabel[inds]
            percam_feat = features[inds] # (num of same cam, 2048)
            percam_proxy = proxy[inds]

            # intra-camera loss
            mapped_targets = [temp_proxy - self.cam_proxy_num[cc] for temp_proxy in percam_proxy]
            mapped_targets = torch.tensor(mapped_targets).to(self.device)
            percam_inputs = ExemplarMemory.apply(percam_feat[:, :2048], mapped_targets, self.percam_memory[cc][:, :2048], self.alpha)
            percam_inputs /= self.beta  # similarity score before softmax
            percam_inputs1 = ExemplarMemory.apply(percam_feat[:, 2048:4096], mapped_targets, self.percam_memory[cc][:, 2048:4096], self.alpha)
            percam_inputs1 /= self.beta  # similarity score before softmax
            # percam_inputs2 = ExemplarMemory.apply(percam_feat[:, 4096:], mapped_targets, self.percam_memory[cc][:, 4096:], self.alpha)
            # percam_inputs2 /= self.beta  # similarity score before softmax
            loss += 0.6 * F.cross_entropy(percam_inputs, mapped_targets)
            loss += 0.6 * F.cross_entropy(percam_inputs1, mapped_targets)
            # loss += 0.6 * F.cross_entropy(percam_inputs2, mapped_targets)

            # global loss + globel memory loss (local2global) split
            associate_loss = 0
            target_inputs = percam_feat[:, :2048].mm(percam_tempV[:, :2048].t())
            target_inputs1 = percam_feat[:, 2048:4096].mm(percam_tempV[:, 2048:4096].t())
            # target_inputs2 = percam_feat[:, 4096:].mm(percam_tempV[:, 4096:].t())
            offline_temp_sims = target_inputs.detach().clone() # (len(percam_feat), len(self.concate_intra_class))
            offline_temp_sims1 = target_inputs1.detach().clone() # (len(percam_feat), len(self.concate_intra_class))
            # offline_temp_sims2 = target_inputs2.detach() # (len(percam_feat), len(self.concate_intra_class))
            target_inputs /= self.beta                 # (len(percam_feat), len(proxy_num + global_proxy_num))
            target_inputs1 /= self.beta                 # (len(percam_feat), len(proxy_num + global_proxy_num))
            # target_inputs2 /= self.beta                 # (len(percam_feat), len(proxy_num + global_proxy_num))
            for k in range(len(percam_feat)):
                ori_asso_ind = self.proxy_label_dict[int(percam_targets[k])]
                ################################################
                offline_temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(offline_temp_sims[k])[1][-self.bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                offline_temp_sims1[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind1 = torch.sort(offline_temp_sims1[k])[1][-self.bg_knn:]
                concated_input1 = torch.cat((target_inputs1[k, ori_asso_ind], target_inputs1[k, sel_ind1]), dim=0)
                # offline_temp_sims2[k, ori_asso_ind] = -10000.0  # mask out positive
                # sel_ind2 = torch.sort(offline_temp_sims2[k])[1][-self.bg_knn:]
                # concated_input2 = torch.cat((target_inputs2[k, ori_asso_ind], target_inputs2[k, sel_ind2]), dim=0)
                ################################################
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(self.device)
                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                associate_loss += -1 * (F.log_softmax(concated_input1.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                # associate_loss += -1 * (F.log_softmax(concated_input2.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
            loss += 0.7 * associate_loss / len(percam_feat)

        percam_tempV = []
        for ii in self.unique_cams:
            percam_tempV.append(self.percam_memory[ii].detach().clone()) #(num of label, 2048)
        percam_tempV = torch.cat(percam_tempV, dim=0).to(self.device) # real memorybank looks like the picture in the paper (num of proxy, 2048)
        
        for cc in torch.unique(cams):
            inds = torch.nonzero(cams == cc).squeeze(-1) # index of traget camera
            online_loss = 0
            percam_feat = features[inds] # (num of same cam, 2048)
            #################################################################################
            target_inputs = percam_feat[:, :2048].mm(percam_tempV[:, :2048].t())
            target_inputs1 = percam_feat[:, 2048:4096].mm(percam_tempV[:, 2048:4096].t())
            # target_inputs2 = percam_feat[:, 4096:].mm(percam_tempV[:, 4096:].t())
            online_temp_sims = target_inputs.detach().clone()
            online_temp_sims1 = target_inputs1.detach().clone()
            # online_temp_sims2 = target_inputs2.detach()
            target_inputs /= self.beta
            target_inputs1 /= self.beta
            # target_inputs2 /= self.beta
            #################################################################################
            for k in range(len(percam_feat)):
                all_cam_tops = []
                all_cam_tops1 = []
                # all_cam_tops2 = []
                for c in self.unique_cams:
                    proxy_inds = self.proxy_cam_dict[int(c)]
                    maxInd = online_temp_sims[k, proxy_inds].argmax()
                    maxInd1 = online_temp_sims1[k, proxy_inds].argmax()
                    # maxInd2 = online_temp_sims2[k, proxy_inds].argmax()
                    all_cam_tops.append(proxy_inds[maxInd])
                    all_cam_tops1.append(proxy_inds[maxInd1])
                    # all_cam_tops2.append(proxy_inds[maxInd2])
                all_cam_tops = torch.tensor(all_cam_tops).to(self.device)
                all_cam_tops1 = torch.tensor(all_cam_tops1).to(self.device)
                # all_cam_tops2 = torch.tensor(all_cam_tops2).to(self.device)
                online_sel_ind = torch.argsort(online_temp_sims[k, all_cam_tops])[-3:]
                online_sel_ind1 = torch.argsort(online_temp_sims1[k, all_cam_tops1])[-3:]
                # online_sel_ind2 = torch.argsort(online_temp_sims2[k, all_cam_tops2])[-3:]
                pos_lenth = online_sel_ind.size(0)
                online_temp_sims[k, all_cam_tops[online_sel_ind]] = 10000
                online_temp_sims1[k, all_cam_tops1[online_sel_ind1]] = 10000
                # online_temp_sims2[k, all_cam_tops2[online_sel_ind2]] = 10000
                top_inds = torch.sort(online_temp_sims[k])[1][-30-pos_lenth:]
                top_inds1 = torch.sort(online_temp_sims1[k])[1][-30-pos_lenth:]
                # top_inds2 = torch.sort(online_temp_sims2[k])[1][-30-pos_lenth:]
                all_lenth = top_inds.size(0)
                sel_input = target_inputs[k, top_inds]
                sel_input1 = target_inputs1[k, top_inds1]
                # sel_input2 = target_inputs2[k, top_inds2]
                target = torch.zeros(all_lenth, dtype=sel_input.dtype).to(self.device)
                target[-pos_lenth:] = 1.0 / pos_lenth
                online_loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * target.unsqueeze(0)).sum()
                online_loss += -1.0 * (F.log_softmax(sel_input1.unsqueeze(0), dim=1) * target.unsqueeze(0)).sum()
                # online_loss += -1.0 * (F.log_softmax(sel_input2.unsqueeze(0), dim=1) * target.unsqueeze(0)).sum()
            
            loss += 0.7 * online_loss / len(percam_feat)

        ## feat reg loss
        if self.current_round > 0:
            percam_global_feat = global_features[inds]
            distance = 1 - torch.cosine_similarity(percam_feat, percam_global_feat)
            # loss += 0.39 * torch.mean(distance)
            loss += 0.26 * torch.mean(distance)
        
        return loss






        