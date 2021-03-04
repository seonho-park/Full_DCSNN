import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

# class Loss_DCSNN2(nn.Module):
#     def __init__(self, device, lam_recon=10., lam_reg = 0.1, lam_local = 1., s = 1.):
#         super().__init__()
#         self.lam_recon = lam_recon
#         self.lam_reg = lam_reg
#         self.lam_local = lam_local
#         self.device = device
#         self.s = s
#         self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

#     def forward(self, global_features, local_features, local_features_recon, local_aggre_features, D_global, D_local, S):
#         omega_x = global_features.mm(Variable(D_global,requires_grad=False).t())/self.s # logit values

#         # global loss
#         global_loss = self.BCEWithLogitsLoss(omega_x, S)
#         norm = global_features.norm(dim=1)
#         reg_global = (norm - torch.ones_like(norm).cuda()).pow(2).mean()

#         # reconstruction loss
#         recon_loss = 1/local_features.flatten().size(0)*torch.norm(local_features.flatten()-local_features_recon.flatten())

#         # local aggregation loss
#         local_omega_x = local_aggre_features.mm(Variable(D_local,requires_grad=False).t())/self.s
#         local_loss = self.BCEWithLogitsLoss(local_omega_x, S)
#         norm = local_aggre_features.norm(dim=1)
#         reg_local = (norm - torch.ones_like(norm).cuda()).pow(2).mean()
        
#         # total loss
#         loss =  global_loss + self.lam_local*local_loss + self.lam_recon*recon_loss + self.lam_reg * (reg_global + reg_local)

#         return loss, global_loss, local_loss, recon_loss, reg_global, reg_local

def _compute_homography_est_loss(h4_est, h4):
    return F.mse_loss(h4_est.view(1,-1), h4.view(1,-1).float())

def _compute_matching_loss(ms,kp,kpw,h,gt_radious):
    kp = kp.flip(1) # flip to x,y coordinates from h,w 
    kp = torch.cat((kp, kp.new_ones((kp.size(0),1))), axis=1).float()
    kpw_ = torch.mm(h.float(), kp.t()).t()
    kpw_ /= kpw_[:,2].view(kpw_.size(0),1)
    kpw_ = kpw_[:,:2]
    kpw_ = kpw_.flip(1) # flip back to h,w coordinates
    # kp = kp[:,:2]
    dist = torch.norm(kpw_.view(kpw_.size(0),1,2) - kpw.view(1,kpw.size(0),2), p=2, dim=-1) # nkp (k) X nkpw (q)
    s = (dist<=gt_radious).float()
    s_pred = ms[:-1,:-1]

    s_pred_k = ms[:-1,-1]
    s_pred_q = ms[-1,:-1]

    s_k = torch.where(s.sum(1) == 0, s.new_tensor(1.), s.new_tensor(0.))
    s_q = torch.where(s.sum(0) == 0, s.new_tensor(1.), s.new_tensor(0.))

    # loss =  - (s_k*s_pred_k.log()).sum() - (s_q*s_pred_q.log()).sum()
    loss_term1 = -(s*s_pred).sum() 
    loss_term2 = - (s_k*s_pred_k).sum()
    loss_term3 = - (s_q*s_pred_q).sum()
    # if (s*s_pred).sum() > 0.:
    #     loss -= (s*s_pred.log()).sum()
    # print(loss_term1.item(), loss_term2.item(),loss_term3.item())
    # if np.isnan(loss_term1.item()):
    #     aa = 1
    loss = loss_term1 + loss_term2 + loss_term3
    return loss
    # return F.binary_cross_entropy(s_pred, s) +F.binary_cross_entropy(s_pred_k, s_k)+ \
                # F.binary_cross_entropy(s_pred_q, s_q)

def compute_matcher_loss(matching_scores, h4_ests, H, H4, kpts_k, kpts_q, lam, device, gt_radious=4.):
    nimg = len(matching_scores)
    matching_loss = torch.zeros(1).float().to(device)
    homoest_loss = torch.zeros(1).float().to(device)

    for ms, kp, kpw, h, h4_est, h4 in zip(matching_scores, kpts_k, kpts_q, H, h4_ests, H4):
        matching_loss += _compute_matching_loss(ms,kp,kpw,h,gt_radious)
        homoest_loss += _compute_homography_est_loss(h4_est, h4)
    matching_loss /= nimg
    homoest_loss /= nimg
    homoest_loss*=lam

    return matching_loss + homoest_loss, matching_loss, homoest_loss



def descriptor_loss(net_out, H, positive_margin=1., negative_margin=0.2):
    lds, lds_warp = net_out['localdesc_k'], net_out['localdesc_q']
    kpts, kpts_warp = net_out['kpts_k'], net_out['kpts_q']
    assert len(kpts) == len(kpts_warp) & len(lds) == len(lds_warp) & len(lds) == H.size(0)
    nimg = len(kpts)
    loss = lds[0].new_zeros(1)

    for ld, ldw, kp, kpw, h in zip(lds, lds_warp, kpts, kpts_warp, H):
        kp = kp.flip(1) # flip to x,y coordinates from h,w 
        kp = torch.cat((kp, kp.new_ones((kp.size(0),1))), axis=1).float()
        
        kpw_ = torch.mm(h.float().to(kp), kp.t()).t()
        kpw_ = kpw_/kpw_[:,2].view(-1,1)
        kpw_ = kpw_[:,:2]
        kp = kp[:,:2]

        kpw_ = kpw_.flip(1) # flip back to h,w coordinates

        # kpw_ = torch.where(kpw_<0, kpw_.new_tensor(0), kpw_)
        # kpw_ = torch.where(kpw_>223., kpw_.new_tensor(223), kpw_)
        
        dist = torch.norm(kpw_.view(kpw_.size(0),1,2) - kpw.view(1,kpw.size(0),2), p=2, dim=-1) # nkpw X nkpw_
        s = (dist<=8.).float()
        # print(s.sum().item())
        ld_mat = torch.einsum('dl,dw->lw', ld, ldw)
        loss_p = torch.max(torch.zeros_like(ld_mat), positive_margin-ld_mat*s)*s
        loss_p = loss_p.sum()/torch.nonzero(s,as_tuple=False).size(0)
        loss_m = torch.max(torch.zeros_like(ld_mat), ld_mat*(1.-s) - negative_margin)
        loss_m = loss_m.sum()/torch.nonzero(1-s,as_tuple=False).size(0)
        loss += loss_p + loss_m
    
    loss /= nimg
    # if torch.isnan(loss):
    #     aa = 1
        
    return loss


class Loss_DCSNN(nn.Module):
    def __init__(self, device, lamda = [0.1, 10, 0.001], temp = 0.5, self_learning=True):
        super().__init__()
        self.lam_reg, self.lam_kp, self.lam_ld = lamda
        self.device = device
        self.T = temp
        self.self_learning = self_learning

    # def forward(self, global_feature_q, global_feature_k, key_queue, S, kp_gt, kp_out, kp_warp_gt, kp_warp_out):
    def forward(self, key_queue, adj_mat, kpts_gt_k, kpts_gt_q, net_out, H):
        globaldesc_q = net_out['globaldesc_q']
        globaldesc_k = net_out['globaldesc_k']
        kpts_q = net_out['scores_map_q']
        kpts_k = net_out['scores_map_k']
        
        # regularization
        norm = globaldesc_q.norm(dim=1)
        reg_global = (norm - torch.ones_like(norm).cuda()).pow(2).mean()
        
        # global loss
        omega_x = globaldesc_q.mm(Variable(key_queue,requires_grad=False))/self.T # logit values
        global_loss = F.binary_cross_entropy_with_logits(omega_x, adj_mat)
        
        # self learning
        if self.self_learning:
            norm_k = globaldesc_k.norm(dim=1)
            reg_global_self = (norm_k - torch.ones_like(norm_k).cuda()).pow(2).mean()
            reg_global += reg_global_self

            self_omega_x = (globaldesc_q*Variable(globaldesc_k,requires_grad=False)).sum(dim=1)/self.T
            label = torch.ones_like(self_omega_x) # self label
            global_loss_self = F.binary_cross_entropy_with_logits(self_omega_x, label)
            global_loss += global_loss_self

        # kp_example = kpts_k[0].cpu().detach().numpy()
        # from PIL import Image
        # kp_example = Image.fromarray(kp_example, mode='L')

        # kpgt_example = kpts_gt_k.squeeze()[0].cpu().detach().numpy()
        # from PIL import Image
        # kpgt_example = Image.fromarray(kpgt_example, mode='L')

        # keypoint - element-wise bce
        # kp_bce = F.binary_cross_entropy_with_logits(kpts_q, kpts_gt_q.squeeze()) \
        #        + F.binary_cross_entropy_with_logits(kpts_k, kpts_gt_k.squeeze())
        # kp_bce = F.binary_cross_entropy(torch.sigmoid(kpts_q), kpts_gt_q.squeeze()) \
        #        + F.binary_cross_entropy(torch.sigmoid(kpts_k), kpts_gt_k.squeeze())
        kp_loss = F.binary_cross_entropy(kpts_q, kpts_gt_q.squeeze()) \
               + F.binary_cross_entropy(kpts_k, kpts_gt_k.squeeze())

        ld_loss = descriptor_loss(net_out, H) # local descriptor loss
            
        # total loss
        reg_global *= self.lam_reg
        kp_loss *= self.lam_kp
        ld_loss *= self.lam_ld
        loss =  global_loss + reg_global + kp_loss + ld_loss

        return loss, global_loss, reg_global, kp_loss, ld_loss



def setup_loss(method, device, **kwargs):
    method = method.lower()
    # if method in ['dpsh']:
    #     criterion = Loss_DPSH(device, **kwargs)
    # elif method in ['dhn']:
    #     criterion = Loss_DHN(device, **kwargs)
    # elif method in ['dhnnl2']:
    #     criterion = Loss_DHNNL2(device, **kwargs)
    # elif method in ['dcsnn']:
    #     criterion = Loss_DCSNN(device, **kwargs)
    # if method in ['dcsnn2']:
    #     criterion = Loss_DCSNN2(device, **kwargs)
    # else:
    #     raise ValueError('Unknown method type!')
    criterion = Loss_DCSNN(device, **kwargs)
    return criterion