import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Loss_DCSNN2(nn.Module):
    def __init__(self, device, lam_recon=10., lam_reg = 0.1, lam_local = 1., s = 1.):
        super().__init__()
        self.lam_recon = lam_recon
        self.lam_reg = lam_reg
        self.lam_local = lam_local
        self.device = device
        self.s = s
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    def forward(self, global_features, local_features, local_features_recon, local_aggre_features, D_global, D_local, S):
        omega_x = global_features.mm(Variable(D_global,requires_grad=False).t())/self.s # logit values

        # global loss
        global_loss = self.BCEWithLogitsLoss(omega_x, S)
        norm = global_features.norm(dim=1)
        reg_global = (norm - torch.ones_like(norm).cuda()).pow(2).mean()

        # reconstruction loss
        recon_loss = 1/local_features.flatten().size(0)*torch.norm(local_features.flatten()-local_features_recon.flatten())

        # local aggregation loss
        local_omega_x = local_aggre_features.mm(Variable(D_local,requires_grad=False).t())/self.s
        local_loss = self.BCEWithLogitsLoss(local_omega_x, S)
        norm = local_aggre_features.norm(dim=1)
        reg_local = (norm - torch.ones_like(norm).cuda()).pow(2).mean()
        
        # total loss
        loss =  global_loss + self.lam_local*local_loss + self.lam_recon*recon_loss + self.lam_reg * (reg_global + reg_local)

        return loss, global_loss, local_loss, recon_loss, reg_global, reg_local


class Loss_DCSNN(nn.Module):
    def __init__(self, device, lam_reg = 0.1, temp = 1.):
        super().__init__()
        self.lam_reg = lam_reg
        self.device = device
        self.T = temp

    def forward(self, global_features1, global_features2, D_global, S):
        omega_x = global_features1.mm(Variable(D_global,requires_grad=False).t())/self.T # logit values

        self_omega_x = (global_features1*Variable(global_features2,requires_grad=False)).sum(axis=1)/self.T
        label = torch.ones_like(self_omega_x)
        
        # global loss
        global_loss = F.binary_cross_entropy_with_logits(omega_x, S)
        global_loss2 = F.binary_cross_entropy_with_logits(self_omega_x, label)
        norm = global_features1.norm(dim=1)
        reg_global = (norm - torch.ones_like(norm).cuda()).pow(2).mean()

        # total loss
        loss =  global_loss + global_loss2 + self.lam_reg * reg_global

        return loss, global_loss, reg_global


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