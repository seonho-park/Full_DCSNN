import argparse
import numpy as np
import time
import torch

import configs
import pqcode
import model
import utils
import loader_grd
from superglue import SuperGlue

def _compute_mAP(Xtrain, gtmat, testloader, testloader_real, device, descriptor, matcher):
    Xtrain = Xtrain/np.linalg.norm(Xtrain, axis=1).reshape(-1,1)
    pq = pqcode.PQ(M=16, Ks=256, verbose=False)    
    pq.fit(Xtrain, iter=200, seed=123)
    Xtrain_encode = pq.encode(Xtrain) # encode to PQ code
    mAP_vals = []
    # mAP_vals_l2norm = []
    # mAP_vals_cs = []
    # mAP_l2norm = 0.
    # mAP_cs = 0.

    with torch.no_grad():
        for i, (imgs, _, idx) in enumerate(testloader_real):
            imgs = imgs.to(device)
            idx = idx.cpu().numpy()
            globaldesc = descriptor.moco(imgs)[0]
            features = globaldesc.detach().cpu().numpy()

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
                print(map_val)
                map_val = np.asarray([map_val[:j+1].mean() for j in range(10)])
                mAP_vals.append(map_val.mean())
                
                if matcher is not None:
                    img_q = testloader_real.dataset[idx_query][0].to(device) # query
                    kpts_q, ld_q, scores_q, _ = descriptor.lf_generator(img_q.unsqueeze(0))
                    for ik in retrieved_idxs:
                        img_k = testloader.dataset[ik][0].to(device)
                        kpts_k, ld_k, scores_k, _ = descriptor.lf_generator(img_k.unsqueeze(0))
                        
                        matching_scores, h4_ests = matcher(kpts_k, ld_k, scores_k, kpts_q, ld_q, scores_q, \
                                                            img_k.unsqueeze(0), img_q.unsqueeze(0))
                        matching_scores2, h4_ests = matcher(kpts_k, ld_k, scores_k, kpts_k, ld_k, scores_k, \
                                                            img_k.unsqueeze(0), img_k.unsqueeze(0))
                        # matching_out = matcher.get_matching_result(matching_scores[0])
                        score = (matching_scores[0][:-1,:-1]>0.8).float().sum()
                        score2 = (matching_scores2[0][:-1,:-1]>0.8).float().sum()
                        print(score, score2)

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

                # if verbose:
                #     print('mAP: %.4f'%(mAP_vals[-1]))
            # del features, kp_out
            # del features
    
    mAP = np.asarray(mAP_vals).mean()
    # mAP_l2norm = np.asarray(mAP_vals_l2norm).mean()
    # mAP_cs = np.asarray(mAP_vals_cs).mean()
    return mAP


def test(gtmat, testloader, testloader_real, device, descriptor, matcher = None):
    descriptor.eval()
    if matcher is not None:
        matcher.eval()
    X = []
    t0 = time.time()
    # idxs = []
    with torch.no_grad():
        for i, (img, _, idx) in enumerate(testloader):
            img = img.to(device)
            global_desc = descriptor.moco(img)[0]
            X.append(global_desc.detach().cpu().numpy())

    X = np.concatenate(X)
    print("Test Elapsed Time 1 :", time.time()-t0)
    mAP, mAP_after = _compute_mAP(X, gtmat, testloader, testloader_real, device, descriptor, matcher)
    print("Test Elapsed Time 2 :", time.time()-t0)
    del X

    return mAP, mAP_after

def main():
    device = utils.get_device()
    utils.set_seed(args.seed, device) # set random seed

    # load data
    assert args.dataname in ['Haywrd','ykdelB']
    dataconfig = configs.dataconfig[args.sartype.lower()]
    traindata = dataconfig[args.dataname]['traindata']
    testdata = dataconfig[args.dataname]['testdata']
    
    trainloader = loader_grd.setup_trainloader(traindata, args)
    testloader = loader_grd.setup_testloader(traindata, args)
    testloader_real = loader_grd.setup_testloader(testdata, args)
    gtmat = loader_grd.load_gtmat(args.sartype, dataconfig[args.dataname]['gtmat'])

    state_dict = torch.load(args.descriptor_name)
    args_load = state_dict['args']
    descriptor = model.dcsnn_full(args_load["nglobal"], 256, pretrained=True, K=args_load['moco_k'], m=args_load['moco_m'],\
         T = args_load['temp'], no_moco=args_load['no_moco'], ntrain=len(traindata)).to(device)
    descriptor.load_state_dict(state_dict['net']) # load chpt

    state_dict = torch.load(args.matcher_name)
    args_load = state_dict['args']
    matcher = SuperGlue(sinkhorn_iterations=args_load['sinkhorn_iter'], match_threshold=args_load['match_threshold']).to(device)
    matcher.load_state_dict(state_dict['net'])

    mAP, mAP_after, mAP_l2, mAP_cs = test(gtmat, testloader, testloader_real, device, descriptor, matcher) # before & after reranking


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-d', '--descriptor_name', type=str, default='./chpt/descriptor_Haywrd_tanta_attempt4.pth', help='trained descriptor name along with path')
    parser.add_argument('-m', '--matcher_name', type=str, default='./chpt/matcher_Haywrd_tanta_attempt1.pth', help='trained descriptor name along with path')
    parser.add_argument('--bstrain', type=int, default=8, help='batch size for training')
    parser.add_argument('--bstest', type=int, default=16, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=2, help='the number of workers used in DataLoader')
    parser.add_argument('--sartype', type=str, default='grd', help='SAR image type mlc (previous)|grd (current)')
    parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')

    args = parser.parse_args()
    main()


