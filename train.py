import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import argparse
import json
import time
import numpy as np
import os
from datetime import datetime
import time

import model
import loader
import losses
import utils
from configs import *
import pqcode


def compute_mAP(Xtrain, gtmat, testloader_real, device, net, verbose=False):
    Xtrain = Xtrain/np.linalg.norm(Xtrain, axis=1).reshape(-1,1)
    pq = pqcode.PQ(M=16, Ks=256, verbose=False)    
    pq.fit(Xtrain, iter=200, seed=123)
    Xtrain_encode = pq.encode(Xtrain) # encode to PQ code
    mAP_vals = []
    with torch.no_grad():
        for i, (imgs, idx) in enumerate(testloader_real):
            imgs = imgs.to(device)
            idx = idx.cpu().numpy()
            features = net(imgs).detach().cpu().numpy()

            for ii in range(features.shape[0]):
                idx_query = idx[ii]
                x_query = features[ii]
                x_query = x_query/np.linalg.norm(x_query)
                dist = pq.dtable2(x_query).adist(Xtrain_encode)
                sorted_dist_idx = np.argsort(dist)[::-1] # descending order
                retrieved_idxs = sorted_dist_idx[:10]
                map_val = np.zeros(10)
                for j, retrieved_idx in enumerate(retrieved_idxs):
                    map_val[j] = gtmat[idx_query, retrieved_idx]
                map_val = np.asarray([map_val[:j+1].mean() for j in range(10)])
                mAP_vals.append(map_val.mean())
                if verbose:
                    print('mAP: %.4f'%(mAP_vals[-1]))
    
    mAP = np.asarray(mAP_vals).mean()
    return mAP


def train(epoch, net, trainloader, criterion, optimizer, device, D_global, logger, bsanchor):
    train_loss = 0.
    global_loss_total = 0.
    reg_global_total = 0.
    elapsed_time_total = 0.

    net.train() # train mode
    dataset = trainloader.dataset
    ntrain = len(dataset)

    for batch_idx, (imgs1, imgs2, idxs) in enumerate(trainloader):
        t0 = time.time()
        imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
        optimizer.zero_grad()
        features1, features2 = net(imgs1, imgs2)
        # features1 = net(imgs1)
        # features2 = net(imgs2).detach()
        features2_ = features2.clone().detach()
        
        # update Ds
        for i, idx in enumerate(idxs):
            D_global[idx,:] = features2_[i,:]
        A = dataset.get_sim_mat(idxs,device) # adjacency matrix

        # get stochastic U and S
        anchor_idxs = np.random.permutation(ntrain)[:bsanchor]
        D_global_ = D_global[anchor_idxs,:]
        A_ = A[:,anchor_idxs]

        loss, global_loss, reg_global = criterion(features1, features2, D_global_, A_)
        loss.backward()
        train_loss += loss.item()
        global_loss_total += global_loss.item()
        reg_global_total += reg_global.item()
        
        optimizer.step()

        elapsed_time_total += time.time()-t0
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f | Global Loss: %.4f | Reg Global Loss: %.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),global_loss_total/(batch_idx+1),\
                reg_global_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)), end = '\r')
    print('')
    logger.write_only('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f | Global Loss: %.4f | Reg Global Loss: %.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),global_loss_total/(batch_idx+1),\
                reg_global_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)))

        
def test(net, gtmat, testloader, testloader_real, device, logger):
    net.eval()
    X = []
    idxs = []
    with torch.no_grad():
        for i, (imgs, idx) in enumerate(testloader):
            imgs = imgs.to(device)
            features = net(imgs)
            X.append(features.detach().cpu().numpy())
            idxs.append(idx.detach().cpu().numpy().squeeze())
    X = np.concatenate(X)

    mAP = compute_mAP(X, gtmat, testloader_real, device, net)
    return mAP

def main():
    if not os.path.exists('./chpt'):
        os.mkdir('./chpt')

    if not os.path.exists('./results'):
        os.mkdir('./results')

    logger, result_dir, dir_name = utils.config_backup_get_log(args,__file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed
    
    chpt_name = '%s_%s_%d'%(args.dataname, args.method, args.nglobal)
    print('Checkpoint name:', chpt_name)

    # setup architecture
    net = model.dcsnn_full(args.nglobal, pretrained=True, K=args.bstrain*100, m=args.moco_m, T = args.temp).to(device)

    # setup criterion and optimizer
    criterion = losses.setup_loss(args.method, device, lam_reg = args.lamreg, temp = args.temp)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)   
    
    # load data
    assert args.dataname in ['Haywrd','ykdelB']
    traindata = dataconfig[args.dataname]['traindata']
    testdata = dataconfig[args.dataname]['testdata']
    gtmat = np.load(dataconfig[args.dataname]['gtmat']) # load groundtruth matrix
    trainloader = loader.setup_trainloader(traindata, args)
    testloader = loader.setup_testloader(traindata, args)
    testloader_real = loader.setup_testloader(testdata, args)
    
    # initialize the anchor matrix U
    D_global = torch.zeros((len(trainloader.dataset), args.nglobal), device = device) 
    net.eval()
    with torch.no_grad():
        for imgs1, imgs2, idxs in trainloader:
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
            features1, _ = net(imgs1, imgs2)
            features1 = features1.detach()
            
            for i, idx in enumerate(idxs):
                D_global[idx,:] = features1[i,:]

    print('==> Start training ..')   
    start = time.time()
    best_mAP = -1
    
    for epoch in range(args.maxepoch):
        train(epoch, net, trainloader, criterion, optimizer, device, D_global, logger, args.bsanchor)
        scheduler.step() # update optimizer lr
        if epoch%10==0:
            mAP = test(net, gtmat, testloader, testloader_real, device, logger)
            logger.write("mAP: %0.6f"%(mAP))
            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch
                state = {'method': args.method, 'net': net.state_dict(), 'mAP': mAP, 'args': vars(args)}
                torch.save(state, './%s/%s.pth'%(result_dir, chpt_name))


    if args.batchout:
        with open('temp_result.txt', 'w') as f:
            f.write("%10.8f\n"%(best_mAP))
            f.write("%d"%(best_epoch))


    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.write("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')
    parser.add_argument('--method', type=str, default='dcsnn2', help='learning method dpsh|dhn|dhnnl2|dcsnn')
    parser.add_argument('--bstrain', type=int, default=32, help='batch size for training')
    parser.add_argument('--bsanchor', type=int, default=1024, help='batch size for anchor matrix')
    parser.add_argument('--bstest', type=int, default=64, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=110, help='the number of epoches')
    
    parser.add_argument('--nglobal', type=int, default=128, help='global desc. feature dimension')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lamreg', type=float, default=0.1, help='coefficient for regularization')
    parser.add_argument('--temp', type=float, default=0.5, help='logit temperature')
    parser.add_argument('--moco-m', type=float, default=0.9, help='moco momentum of updating key encoder (default: 0.999)')

    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    parser.add_argument('--batchout', action='store_true', help='batch out')
    
    args = parser.parse_args()
    main()    
 