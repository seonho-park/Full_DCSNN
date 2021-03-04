import gc
import torch
import argparse
import json
import time
import os
from datetime import datetime
import time

import model
import loader_grd
import losses
import utils
import configs
from test import test


def train(epoch, net, trainloader, criterion, net_optimizer, device, logger):
    train_loss = 0.
    elapsed_time_total = 0.
    global_loss_total, reg_global_total, kp_bce_total, localdesc_loss_total = 0., 0., 0., 0.

    net.train() # train mode
    dataset = trainloader.dataset
    # mem1 = torch.cuda.memory_allocated()
    for batch_idx, (img_k, img_q, kpts_gt_k, kpts_gt_q, idx, H, H4) in enumerate(trainloader):
        if idx.size(0) != args.bstrain:
            continue
        t0 = time.time()
        img_k, img_q, kpts_gt_k, kpts_gt_q = img_k.to(device), img_q.to(device), kpts_gt_k.to(device), kpts_gt_q.to(device)
        adj_mat = dataset.get_sim_mat(idx,'cpu')[:,net.moco.queue_idx.squeeze()].to(device) # adjacency matrix

        net_optimizer.zero_grad()
        # if batch_idx%2 == 0:
        #     net_optimizer.zero_grad()
        
        net_out = net(img_q, img_k)
        
        loss, global_loss, reg_global, kp_bce, localdesc_loss = criterion(net.moco.queue, adj_mat, kpts_gt_k, kpts_gt_q, net_out, H)

        # print(global_loss.item(), reg_global.item(), kp_bce.item(), localdesc_loss.item())

        loss.backward()
        train_loss += loss.item()
        global_loss_total += global_loss.item()
        reg_global_total += reg_global.item()
        kp_bce_total += kp_bce.item()
        localdesc_loss_total += localdesc_loss.item()
        # if batch_idx%2 == 1:
        #     net_optimizer.step()
        net_optimizer.step()

        net.moco.dequeue_and_enqueue(net_out['globaldesc_k'], idx)

        elapsed_time_total += time.time()-t0

        print('  Training... Epoch:%4d | Iter:%4d/%4d | Mean Loss:%.4f | Global Loss:%.4f | Reg Global Loss:%.4f | KP BCE:%.4f | Local Loss:%.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),global_loss_total/(batch_idx+1),\
                reg_global_total/(batch_idx+1), kp_bce_total/(batch_idx+1), localdesc_loss_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)), end = '\r')
        del net_out
        torch.cuda.empty_cache()
        gc.collect()

    print('')
    logger.write_only('  Training... Epoch: %4d | Iter: %4d/%4d | Mean Loss: %.4f | Global Loss: %.4f | Reg Global Loss: %.4f | KPBCE: %.4f | Local Loss:%.4f | Elapsed TIme: %.4f'%\
            (epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1),global_loss_total/(batch_idx+1),\
                reg_global_total/(batch_idx+1), kp_bce_total/(batch_idx+1), localdesc_loss_total/(batch_idx+1), elapsed_time_total/(batch_idx+1)))


def main():
    logger, result_dir, dir_name = utils.config_backup_get_log(args, "descriptor", __file__)
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed
    
    chpt_name = 'descriptor_%s_%s_%d_%d'%(args.dataname, args.method, args.nglobal, args.nlocal)
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
    net = model.dcsnn_full(args.nglobal, args.nlocal, pretrained=True, K=args.moco_k, m=args.moco_m, T = args.temp, no_moco=args.no_moco, ntrain=len(traindata)).to(device)
    # matcher = SuperGlue(sinkhorn_iterations=100, match_threshold=0.2).to(device)

    # setup criterion and optimizer
    criterion = losses.setup_loss(args.method, device, lamda = (args.lamreg,args.lamkp,args.lamld),\
                                  temp = args.temp, self_learning = not args.no_selflearning)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,25], gamma=0.1)
    
    print('==> Start training ..')   
    start = time.time()
    best_mAP = -1

    for epoch in range(args.maxepoch):
        train(epoch, net, trainloader, criterion, optimizer, device, logger)
        # print("after train \n", torch.cuda.memory_allocated())
        scheduler.step() # update optimizer lr
        if epoch%10==0:
            mAP, _ = test(gtmat, testloader, testloader_real, device, net)
            logger.write("mAP: %0.6f"%(mAP))
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
    parser.add_argument('--method', type=str, default='tanta', help='learning method dpsh|dhn|dhnnl|tanta')
    parser.add_argument('--bstrain', type=int, default=8, help='batch size for training')
    # parser.add_argument('--bsanchor', type=int, default=1024, help='batch size for anchor matrix')
    parser.add_argument('--bstest', type=int, default=16, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--maxepoch', type=int, default=21, help='the number of epoches')
    
    parser.add_argument('--nglobal', type=int, default=128, help='global desc. feature dimension')
    parser.add_argument('--nlocal', type=int, default=256, help='local feature dimension')

    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lamreg', type=float, default=0.1, help='coefficient for regularization')
    parser.add_argument('--lamkp', type=float, default=10., help='coefficient for keypoint')
    parser.add_argument('--lamld', type=float, default=1., help='coefficient for local feature')

    parser.add_argument('--temp', type=float, default=0.5, help='logit temperature')
    parser.add_argument('--moco-m', type=float, default=0.9999, help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-k', type=int, default=1024, help='queue size; number of negative keys (default: 1024)')

    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    parser.add_argument('--batchout', action='store_true', help='batch out')

    parser.add_argument('--no-moco', action='store_true', help='turn off moco')
    parser.add_argument('--no-homography', action='store_true', help='turn off homography')
    parser.add_argument('--no-selflearning', action='store_true', help='turn off self learning')
    args = parser.parse_args()

    assert args.sartype.lower() in ['grd','mlc']
    main()    
 