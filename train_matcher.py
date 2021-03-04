import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import argparse
import gc
import json
import time
import numpy as np
import os
from datetime import datetime
import time
import scipy.sparse as sparse

import model
import loader_grd
from losses import compute_matcher_loss
import utils
import configs
import pqcode

from superglue import SuperGlue

def train(epoch, lf_generator, matcher, trainloader, optimizer, device, logger):
    lf_generator.eval()
    matcher.train()
    train_loss, matching_loss_total, h_loss_total = 0.,0.,0.
    elapsed_time_total = 0.
    for batch_idx, (img_k, img_q, kpts_gt_k, kpts_gt_q, idx, H, H4) in enumerate(trainloader):
        t0 = time.time()
        img_k, img_q, kpts_gt_k, kpts_gt_q = img_k.to(device), img_q.to(device), kpts_gt_k.to(device), kpts_gt_q.to(device)
        H, H4 = H.to(device), H4.to(device)

        # optimizer.zero_grad()
        if batch_idx%2 == 0:
            optimizer.zero_grad()
        
        with torch.no_grad():
            kpts_q, ld_q, scores_q, _ = lf_generator(img_q)
            kpts_k, ld_k, scores_k, _ = lf_generator(img_k)

        matching_scores_out, h4_ests = matcher(kpts_k, ld_k, scores_k, kpts_q, ld_q, scores_q, img_k, img_q)
        loss, matching_loss, h_loss = compute_matcher_loss(matching_scores_out, h4_ests, H, H4, kpts_k, kpts_q, args.lamhomo, device)

        loss.backward()
        # optimizer.step()
        if batch_idx%2 == 1:
            optimizer.step()

        train_loss += loss.item()
        matching_loss_total += matching_loss.item()
        h_loss_total += h_loss.item()
        elapsed_time_total += time.time()-t0
        print('  Training... Epoch:%4d | Iter:%4d/%4d | Mean Loss:%.4f | Matching Loss:%.4f | Homography Loss:%.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),matching_loss_total/(batch_idx+1),\
                h_loss_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)), end = '\r')
        del matching_scores_out, h4_ests, kpts_q, ld_q, scores_q, kpts_k,  ld_k, scores_k, _
        torch.cuda.empty_cache()
        gc.collect()
    
    print('')
    logger.write_only('  Training... Epoch:%4d | Iter:%4d/%4d | Mean Loss:%.4f | Matching Loss:%.4f | Homography Loss:%.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),matching_loss_total/(batch_idx+1),\
                h_loss_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)))



def main():
    logger, result_dir, dir_name = utils.config_backup_get_log(args, "matcher", __file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    chpt_name = 'matcher_%s_%s'%(args.dataname, args.method)
    print('Checkpoint name:', chpt_name)

    # load data
    assert args.dataname in ['Haywrd','ykdelB']
    dataconfig = configs.dataconfig[args.sartype.lower()]
    traindata = dataconfig[args.dataname]['traindata']
    testdata = dataconfig[args.dataname]['testdata']
    
    trainloader = loader_grd.setup_trainloader(traindata, args)
    # testloader = loader_grd.setup_testloader(traindata, args)
    # testloader_real = loader_grd.setup_testloader(testdata, args)
    gtmat = loader_grd.load_gtmat(args.sartype, dataconfig[args.dataname]['gtmat'])
    
    # setup descriptor
    state_dict = torch.load(args.descriptor_name)
    args_load = state_dict['args']
    descriptor = model.dcsnn_full(args_load["nglobal"], 256, pretrained=True, K=args_load['moco_k'], m=args_load['moco_m'],\
         T = args_load['temp'], no_moco=args_load['no_moco'], ntrain=len(traindata)).to(device)
    descriptor.load_state_dict(state_dict['net']) # load chpt
    lf_generator = descriptor.lf_generator

    # setup matcher
    matcher = SuperGlue(sinkhorn_iterations=args.sinkhorn_iter, match_threshold=args.match_threshold).to(device)
    # optimizer = torch.optim.SGD(matcher.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.)
    optimizer = torch.optim.Adam(matcher.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20], gamma=0.1)

    print('==> Start training ..')   
    start = time.time()
    best_mAP = -1

    state_dict = torch.load('./chpt/matcher_Haywrd_tanta_attempt1.pth')
    args_load = state_dict['args']
    # matcher.load_state_dict(state_dict['net'])

    for epoch in range(args.maxepoch):
        train(epoch, lf_generator, matcher, trainloader, optimizer, device, logger)
    
    state = {'method': args.method, 'net': matcher.state_dict(), 'args': vars(args)}
    if result_dir is not None:
        torch.save(state, './%s/%s.pth'%(result_dir, chpt_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--descriptor_name', type=str, default='./chpt/descriptor_Haywrd_tanta_attempt4.pth', help='trained descriptor name along with path')
    parser.add_argument('--sartype', type=str, default='grd', help='SAR image type mlc (previous)|grd (current)')
    parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')
    parser.add_argument('--method', type=str, default='tanta', help='learning method dpsh|dhn|dhnnl2|dcsnn')
    parser.add_argument('--bstrain', type=int, default=4, help='batch size for training')

    parser.add_argument('--bstest', type=int, default=16, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=21, help='the number of epoches')
    
    # parser.add_argument('--nglobal', type=int, default=128, help='global desc. feature dimension')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lamhomo', type=float, default=0., help='coefficient for homography estimation term')

    parser.add_argument('--sinkhorn_iter', type=int, default=100, help='sinkhorn iteration')
    parser.add_argument('--match_threshold', type=float, default=0.2, help=' ')

    # parser.add_argument('--moco-m', type=float, default=0.9999, help='moco momentum of updating key encoder (default: 0.999)')
    # parser.add_argument('--moco-k', type=int, default=1024, help='queue size; number of negative keys (default: 1024)')

    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    parser.add_argument('--batchout', action='store_true', help='batch out')

    parser.add_argument('--no-moco', action='store_true', help='turn off moco')
    parser.add_argument('--no-homography', action='store_true', help='turn off homography')
    parser.add_argument('--no-selflearning', action='store_true', help='turn off self learning')
    args = parser.parse_args()
    # args.no_moco= False
    # args.no_homography = False
    # args.no_selflearning = True

    assert args.sartype.lower() in ['grd','mlc']
    main()    
 