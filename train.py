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
import scipy.sparse as sparse

import model
import loader_grd
import losses
import utils
import configs
import pqcode


def compute_mAP(Xtrain, gtmat, testloader_real, device, net, verbose=False):
    Xtrain = Xtrain/np.linalg.norm(Xtrain, axis=1).reshape(-1,1)
    pq = pqcode.PQ(M=16, Ks=256, verbose=False)    
    pq.fit(Xtrain, iter=200, seed=123)
    Xtrain_encode = pq.encode(Xtrain) # encode to PQ code
    mAP_vals = []
    mAP_vals_l2norm = []
    mAP_vals_cs = []
    mAP_l2norm = 0.
    mAP_cs = 0.

    with torch.no_grad():
        for i, (imgs, idx) in enumerate(testloader_real):
            imgs = imgs.to(device)
            idx = idx.cpu().numpy()
            # kp_out, features = net(imgs)
            features = net(imgs)
            features = features.detach().cpu().numpy()

            for ii in range(features.shape[0]):
                idx_query = idx[ii]
                x_query = features[ii]
                x_query = x_query/np.linalg.norm(x_query)
                dist = pq.dtable2(x_query).adist(Xtrain_encode)
                sorted_dist_idx = np.argsort(dist)[::-1] # descending order (cosine similarity measure)
                retrieved_idxs = sorted_dist_idx[:10]
                
                map_val = np.zeros(10)
                for j, retrieved_idx in enumerate(retrieved_idxs):
                    map_val[j] = gtmat[idx_query, retrieved_idx]
                map_val = np.asarray([map_val[:j+1].mean() for j in range(10)])
                mAP_vals.append(map_val.mean())

                # # exact distance
                # dist = np.linalg.norm(x_query-Xtrain, axis=1) # l2 norm distance
                # sorted_dist_idx = np.argsort(dist) # ascending order
                # retrieved_idxs = sorted_dist_idx[:10]
                # map_val = np.zeros(10)
                # for j, retrieved_idx in enumerate(retrieved_idxs):
                #     map_val[j] = gtmat[idx_query, retrieved_idx]
                # map_val = np.asarray([map_val[:j+1].mean() for j in range(10)])
                # mAP_vals_l2norm.append(map_val.mean())

                # # exact distance - cosine similarity
                # dist = np.matmul(x_query, Xtrain.transpose())
                # sorted_dist_idx = np.argsort(dist)[::-1] # descending order (cosine similarity measure)
                # retrieved_idxs = sorted_dist_idx[:10]
                # map_val = np.zeros(10)
                # for j, retrieved_idx in enumerate(retrieved_idxs):
                #     map_val[j] = gtmat[idx_query, retrieved_idx]
                # map_val = np.asarray([map_val[:j+1].mean() for j in range(10)])
                # mAP_vals_cs.append(map_val.mean())

                if verbose:
                    print('mAP: %.4f'%(mAP_vals[-1]))
            # del features, kp_out
            del features
    
    mAP = np.asarray(mAP_vals).mean()
    # mAP_l2norm = np.asarray(mAP_vals_l2norm).mean()
    # mAP_cs = np.asarray(mAP_vals_cs).mean()
    return mAP, mAP_l2norm, mAP_cs


def train(epoch, net, trainloader, criterion, optimizer, device, logger):
    train_loss = 0.
    global_loss_total = 0.
    reg_global_total = 0.
    elapsed_time_total = 0.

    net.train() # train mode
    dataset = trainloader.dataset
    ntrain = len(dataset)

    for batch_idx, (img_warp, img, idxs) in enumerate(trainloader):
        if idxs.size(0) != args.bstrain:
            continue
        t0 = time.time()
        img_warp, img = img_warp.to(device), img.to(device)
        optimizer.zero_grad()
        mem1 = torch.cuda.memory_allocated()
        # kp_out, feature_out = net(img_warp, img) # img_q:img_warp, img_k:img
        # feature_q, feature_k = feature_out
        feature_q, feature_k = net(img_warp, img)
        mem2 = torch.cuda.memory_allocated()
        # feature_k_ = feature_k.cpu()
        
        # update Ds
        # for i, idx in enumerate(idxs):
        #     D_global[idx,:] = feature_k_[i,:]
        
        # get stochastic U and S
        # anchor_idxs = np.random.permutation(ntrain)[:bsanchor]
        # D_global_ = D_global[anchor_idxs,:].to(device)
        # A_ = A[:,anchor_idxs].to(device)
        A = dataset.get_sim_mat(idxs,'cpu') # adjacency matrix
        A_ = A[:,net.queue_idx.squeeze()].to(device)
        
        loss, global_loss, reg_global = criterion(feature_q, feature_k, net.queue, A_)
        loss.backward()
        train_loss += loss.item()
        global_loss_total += global_loss.item()
        reg_global_total += reg_global.item()
        
        optimizer.step()

        net.dequeue_and_enqueue(feature_k, idxs)
        mem3 = torch.cuda.memory_allocated()
        

        elapsed_time_total += time.time()-t0
        
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f | Global Loss: %.4f | Reg Global Loss: %.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),global_loss_total/(batch_idx+1),\
                reg_global_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)), end = '\r')
        mem4 = torch.cuda.memory_allocated()
        # del loss, kp_out, feature_out
        del loss, feature_k, feature_q
        mem5 = torch.cuda.memory_allocated()
        # print("\n", mem1, mem2, mem3, mem4, mem5)
    print('')
    logger.write_only('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f | Global Loss: %.4f | Reg Global Loss: %.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),global_loss_total/(batch_idx+1),\
                reg_global_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)))

        
def test(net, gtmat, testloader, testloader_real, device, logger):
    net.eval()
    X = []
    t0 = time.time()
    # idxs = []
    with torch.no_grad():
        for i, (img, idx) in enumerate(testloader):
            img = img.to(device)
            # kp_out, feature = net(img)
            feature = net(img)
            X.append(feature.detach().cpu().numpy())
            # idxs.append(idx.detach().cpu().numpy().squeeze())
            # del kp_out, feature
            del feature 
    X = np.concatenate(X)
    print("Test Elapsed Time 1 :", time.time()-t0)
    mAP = compute_mAP(X, gtmat, testloader_real, device, net)
    print("test", torch.cuda.memory_allocated())
    print("Test Elapsed Time 2 :", time.time()-t0)
    del X

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


    # load data
    assert args.dataname in ['Haywrd','ykdelB']
    dataconfig = configs.dataconfig[args.sartype.lower()]
    traindata = dataconfig[args.dataname]['traindata']
    testdata = dataconfig[args.dataname]['testdata']
    
    trainloader = loader_grd.setup_trainloader(traindata, args)
    testloader = loader_grd.setup_testloader(traindata, args)
    testloader_real = loader_grd.setup_testloader(testdata, args)
    gtmat = loader_grd.load_gtmat(args.sartype, dataconfig[args.dataname]['gtmat'])
    
    # setup architecture
    net = model.dcsnn_full(args.nglobal, pretrained=True, K=args.moco_k, m=args.moco_m, T = args.temp, no_moco=args.no_moco, ntrain=len(traindata)).to(device)

    # setup criterion and optimizer
    criterion = losses.setup_loss(args.method, device, lam_reg = args.lamreg, temp = args.temp, self_learning = not args.no_selflearning)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.)
    # optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay=1e-4) # TODO: need experiments
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)   
    
    # initialize the anchor matrix U
    # D_global = torch.zeros((len(trainloader.dataset), args.nglobal), device = "cpu")
    # net.eval()
    # with torch.no_grad():
    #     for img, idxs in testloader:
    #         img = img.to(device)
    #         # feature = net.moco(img).detach().cpu()
    #         feature = net(img).detach().cpu()
            
    #         for i, idx in enumerate(idxs):
    #             D_global[idx,:] = feature[i,:]
    #     del feature
    # print(torch.cuda.memory_allocated())
    print('==> Start training ..')   
    start = time.time()
    best_mAP = -1

    for epoch in range(args.maxepoch):
        train(epoch, net, trainloader, criterion, optimizer, device, logger)
        scheduler.step() # update optimizer lr
        if epoch%10==0:
            mAP, mAP_l2, mAP_cs = test(net, gtmat, testloader, testloader_real, device, logger)
            logger.write("mAP: %0.6f | mAP l2(exact): %0.6f | mAP cs(exact): %0.6f"%(mAP, mAP_l2, mAP_cs))
            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch
                state = {'method': args.method, 'net': net.state_dict(), 'mAP': mAP, 'args': vars(args)}
                if result_dir is not None:
                    torch.save(state, './%s/%s.pth'%(result_dir, chpt_name))

    logger.write("Best mAP: %.6f (Epoch: %d)"%(best_mAP, best_epoch))

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
    parser.add_argument('--sartype', type=str, default='grd', help='SAR image type mlc (previous)|grd (current)')
    parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')
    parser.add_argument('--method', type=str, default='dcsnn2', help='learning method dpsh|dhn|dhnnl2|dcsnn')
    parser.add_argument('--bstrain', type=int, default=32, help='batch size for training')
    # parser.add_argument('--bsanchor', type=int, default=1024, help='batch size for anchor matrix')
    parser.add_argument('--bstest', type=int, default=64, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=200, help='the number of epoches')
    
    parser.add_argument('--nglobal', type=int, default=128, help='global desc. feature dimension')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lamreg', type=float, default=0.1, help='coefficient for regularization')
    parser.add_argument('--temp', type=float, default=0.5, help='logit temperature')
    parser.add_argument('--moco-m', type=float, default=0.9999, help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-k', type=int, default=1024, help='queue size; number of negative keys (default: 1024)')

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
 