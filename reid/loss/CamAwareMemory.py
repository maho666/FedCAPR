import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
import copy
import logging
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


class CAPMemory(nn.Module):
    def __init__(self, device, global_memory_bank, current_round=0, client_ind=0, beta=0.05, alpha=0.01, all_img_cams='', crosscam_epoch=0, bg_knn=50):
        super(CAPMemory, self).__init__()
        self.device = device
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature factor
        self.all_img_cams = torch.tensor(all_img_cams).to(device)
        self.unique_cams = torch.unique(self.all_img_cams)
        self.all_pseudo_label = ''
        self.crosscam_epoch = crosscam_epoch
        self.bg_knn = bg_knn
        self.global_memory_bank = copy.deepcopy(global_memory_bank)
        self.client_memory_lenth = 0
        logger.info("check Gti2 reg015")
        self.used_local_negative = 0
        self.local_negative = 0
        self.denominator = 0

        if current_round > 0:
            logger.info("client_index: {}".format(client_ind))
            del self.global_memory_bank[client_ind]
            self.global_memory_bank = torch.cat(self.global_memory_bank).to(device)
            self.other_memory_lenth = self.global_memory_bank.shape[0]
            # logger.info("other_memory_lenth: {}".format(self.other_memory_lenth))

        self.current_round = current_round
    
    def forward(self, features, global_features, targets, cams=None, epoch=None, all_pseudo_label='',
                batch_ind=-1, init_intra_id_feat=''):

        loss = torch.tensor([0.]).to(self.device)
        self.all_pseudo_label = all_pseudo_label
        self.init_intra_id_feat = init_intra_id_feat

        loss = self.loss_using_pseudo_percam_proxy(features, global_features, targets, cams, batch_ind, epoch)

        return loss


    def loss_using_pseudo_percam_proxy(self, features, global_features, targets, cams, batch_ind, epoch):
        if batch_ind == 0:
            # initialize proxy memory
            self.percam_memory = []
            self.memory_class_mapper = []
            self.concate_intra_class = []
            self.client_memory_lenth = 0
            self.used_local_negative = 0
            self.local_negative = 0
            self.denominator = 0
            for cc in self.unique_cams:
                percam_ind = torch.nonzero(self.all_img_cams == cc).squeeze(-1) # index of same camera
                uniq_class = torch.unique(self.all_pseudo_label[percam_ind]) # classes(label) of selected camera
                uniq_class = uniq_class[uniq_class >= 0] # a list of percam_id
                self.concate_intra_class.append(uniq_class) # len() = cam_num
                cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))} # re-label (from 0)
                self.memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

                if len(self.init_intra_id_feat) > 0:
                    proto_memory = self.init_intra_id_feat[cc]
                    proto_memory = proto_memory.to(self.device)
                    self.percam_memory.append(proto_memory.detach()) # len()=cam_num, percam_memroybank
                    self.client_memory_lenth += proto_memory.shape[0]
            self.concate_intra_class = torch.cat(self.concate_intra_class) # (cam_num*origin_class_per_cam,)
            logger.info("client_memory_lenth: {}".format(self.client_memory_lenth))


        percam_tempV = []
        for ii in self.unique_cams:
            percam_tempV.append(self.percam_memory[ii].detach().clone()) #(num of label, 2048)
        percam_tempV = torch.cat(percam_tempV, dim=0).to(self.device) # real memorybank looks like the picture in the paper (num of proxy, 2048)


        loss = torch.tensor([0.]).to(self.device)
        # calculate from data in a batch
        for cc in torch.unique(cams):
            inds = torch.nonzero(cams == cc).squeeze(-1) # index of traget camera
            percam_targets = self.all_pseudo_label[targets[inds]]
            percam_feat = features[inds] # (num of same cam, 2048)

            # intra-camera loss
            mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).to(self.device)
            percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            percam_inputs /= self.beta  # similarity score before softmax
            loss += 0.6 * F.cross_entropy(percam_inputs, mapped_targets)
         

            # # mix offline loss and global_memory_loss
            # associate_loss = 0
            # if self.current_round < 10:
            #     target_inputs = percam_feat.mm(percam_tempV.t().clone())
            # else:
            #     all_memory_bank = torch.cat([percam_tempV, self.global_memory_bank]).to(self.device)
            #     target_inputs = percam_feat.mm(all_memory_bank.t().clone())
            # temp_sims = target_inputs.detach().clone() # (len(percam_feat), len(self.concate_intra_class))
            # target_inputs /= self.beta                 # (len(percam_feat), len(proxy_num + global_proxy_num))

            # for k in range(len(percam_feat)):
            #     ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)  # ori_asso_ind.shape = (num of same label,)
            #     temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            #     sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
            #     concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
            #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(self.device)
            #     concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
            #     associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
            #     self.local_negative += int(sum(i<self.client_memory_lenth for i in sel_ind))
            #     self.denominator += 50

            # if self.current_round < 10: 
            #     loss += 0.7 * associate_loss / len(percam_feat)
            # else:
            #     loss += 1 * associate_loss / len(percam_feat)

            # global loss + globel memory loss (local2global) split
            associate_loss = 0
            online_loss = 0
            target_inputs = percam_feat.mm(percam_tempV.t().clone())
            offline_temp_sims = target_inputs.detach().clone() # (len(percam_feat), len(self.concate_intra_class))
            online_temp_sims = target_inputs.detach().clone() # (len(percam_feat), len(self.concate_intra_class))
            target_inputs /= self.beta                 # (len(percam_feat), len(proxy_num + global_proxy_num))

            # if self.current_round >= 10:
            #     global_associate_loss = 0
            #     all_memory_bank = torch.cat([percam_tempV, self.global_memory_bank]).to(self.device)
            #     global_target_inputs = percam_feat.mm(all_memory_bank.t().clone())
            #     global_temp_sims = global_target_inputs.detach().clone() # (len(percam_feat), len(self.concate_intra_class))
            #     global_target_inputs /= self.beta                 # (len(percam_feat), len(proxy_num + global_proxy_num))

            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)  # ori_asso_ind.shape = (num of same label,)
                offline_temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(offline_temp_sims[k])[1][-self.bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(self.device)
                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                
                ## calculate online loss in O2CAP
                all_cam_tops = []
                ind = 0
                for c in self.unique_cams:
                    accum = self.percam_memory[c].size(0)
                    max = ind + accum
                    maxInd = online_temp_sims[k, ind:max].argmax()
                    cam_top_ind = ind + maxInd
                    all_cam_tops.append(cam_top_ind)
                    ind += accum 
                all_cam_tops = torch.tensor(all_cam_tops).to(self.device)
                online_sel_ind = torch.argsort(online_temp_sims[k, all_cam_tops])[-3:]
                lenth = online_sel_ind.size(0)
                online_temp_sims[k, all_cam_tops[online_sel_ind]] = 10000
                top_inds = torch.sort(online_temp_sims[k])[1][-5-lenth:]
                sel_input = target_inputs[k, top_inds]
                target = torch.zeros(5+lenth, dtype=sel_input.dtype).to(self.device)
                target[-lenth:] = 1.0 / lenth
                online_loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * target.unsqueeze(0)).sum()

                # if self.current_round >= 10:
                #     global_temp_sims[k, :self.client_memory_lenth] = -10000.0  # mask out positive
                #     global_sel_ind = torch.sort(global_temp_sims[k])[1][-5:]
                #     global_concated_input = torch.cat((global_target_inputs[k, ori_asso_ind], global_target_inputs[k, global_sel_ind]), dim=0)
                #     global_concated_target = torch.zeros((len(global_concated_input)), dtype=global_concated_input.dtype).to(self.device)
                #     global_concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                #     global_associate_loss += -1 * (F.log_softmax(global_concated_input.unsqueeze(0), dim=1) * global_concated_target.unsqueeze(0)).sum()
                
            loss += 0.7 * associate_loss / len(percam_feat)
            loss += 0.7 * online_loss / len(percam_feat)
            # if self.current_round >= 10:
            #     loss += 0.7 * global_associate_loss / len(percam_feat)
    

            ## feat reg loss
            # percam_global_feat = global_features[inds]
            # distance = 1 - torch.cosine_similarity(percam_feat, percam_global_feat)
            # loss += 0.15 * torch.mean(distance)
            # loss += 0.09 * torch.mean(distance)
        # self.used_local_negative = self.local_negative / self.denominator

        return loss






        