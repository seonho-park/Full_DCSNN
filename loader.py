# reference : https://github.com/mez/deep_homography_estimation

import _pickle as pickle
import torch
import torch.utils.data as torchdata
import numpy as np
import scipy.sparse as sparse
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from random import uniform
import os
import random
import cv2
    

class SAR_DataSet(torchdata.Dataset):    
    def __init__(self, img_path, data, desc, kp, desc_sift, kp_sift, patch_size, stride):
        self.img_path = img_path
        self.patch_size = patch_size
        self.stride = stride
        self.data = data
        self.desc = desc
        self.kp = kp
        self.desc_sift = desc_sift
        self.kp_sift = kp_sift

    def get_pixel_coords(self, idx):
        return self.data[idx]['coordinate']
        

    def get_georeference(self, idx):
        ref = self.data[idx]['georeference']
        return ref
    
    def get_kp_desc(self, idx):
        data_i = self.data[idx]
        coord = data_i['coordinate']
        j = coord[0]
        i = coord[1]
        xidx = np.where((self.kp[:,0]>=i) & (self.kp[:,0]<i+self.patch_size))[0]
        yidx = np.where((self.kp[:,1]>=j) & (self.kp[:,1]<j+self.patch_size))[0]
        idx = np.intersect1d(xidx,yidx)
        kp_ = self.kp[idx,:]
        kp_ = kp_ - np.asarray([i,j]) # add offset
        desc_ = self.desc[idx,:]
        return kp_, desc_


    def get_kp_desc_sift(self, idx):
        data_i = self.data[idx]
        coord = data_i['coordinate']
        j = coord[0]
        i = coord[1]
        xidx = np.where((self.kp_sift[:,0]>=i) & (self.kp_sift[:,0]<i+self.patch_size))[0]
        yidx = np.where((self.kp_sift[:,1]>=j) & (self.kp_sift[:,1]<j+self.patch_size))[0]
        idx = np.intersect1d(xidx,yidx)
        kp_ = self.kp_sift[idx,:]
        kp_ = kp_ - np.asarray([i,j]) # add offset
        desc_ = self.desc_sift[idx,:]
        return kp_, desc_


    def get_image(self,idx):
        return np.load(os.path.join(self.img_path, "%04d.npy"%(idx)))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = np.load(os.path.join(self.img_path, "%04d.npy"%(idx)))
        patch = Image.fromarray(patch)
        patch = transforms.functional.to_tensor(patch)
        return patch, idx


rho          = 32
patch_size   = 224
top_point    = (32,32)
left_point   = (patch_size+32, 32)
bottom_point = (patch_size+32, patch_size+32)
right_point  = (32, patch_size+32)
four_points = [top_point, left_point, bottom_point, right_point]


class SAR_TrainDataSet(SAR_DataSet):
    def __init__(self, img_path, data, a_mat, desc, kp, desc_sift, kp_sift, patch_size=800, stride=200):
        super(SAR_TrainDataSet,self).__init__(img_path, data, desc, kp, desc_sift, kp_sift, patch_size, stride)
        self.sim_mat = a_mat

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_path, "%04d.npy"%(idx)))

        # warp 1
        img1 = self.warp_image(img)
        img1 = Image.fromarray(img1)
        img1 = transforms.functional.to_tensor(img1)

        # warp 2
        img2 = self.warp_image(img)
        img2 = Image.fromarray(img2)
        img2 = transforms.functional.to_tensor(img2)

        return img1, img2, idx

    def warp_image(self, img):
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-rho,rho), point[1]+random.randint(-rho,rho)))
        H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )
        H_inverse = np.linalg.inv(H)
        warped_image = cv2.warpPerspective(img, H_inverse, (224,224))
        return warped_image

    def get_sim_mat(self, idxs, device):
        sim_mat = np.zeros((idxs.shape[0], self.sim_mat.get_shape()[0]))
        for (i,), idx in np.ndenumerate(idxs):
            sim_mat[i,:] = self.sim_mat.getrow(idx).todense().squeeze()
        
        return torch.FloatTensor(sim_mat).to(device)
        


def setup_trainloader(data_files, args):
    data_file, img_file = data_files
    data, desc, kp, desc_sift, kp_sift, patch_size, stride = pickle.load(open("%s.pkl"%data_file,'rb'))
    a_mat = sparse.load_npz("%s.npz"%data_file)
    trainset = SAR_TrainDataSet(img_file, data, a_mat, desc, kp, desc_sift, kp_sift, patch_size, stride)
    print('num train data:', len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    return trainloader


def setup_testloader(data_files, args):
    data_file, img_file = data_files
    data, desc, kp, desc_sift, kp_sift, patch_size, stride  = pickle.load(open("%s.pkl"%data_file,'rb'))
    testset = SAR_DataSet(img_file, data, desc, kp, desc_sift, kp_sift, patch_size, stride)
    print('num test data:', len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    return testloader


if __name__ == '__main__':
    from configs import *
    import numpy as np

    traindata = dataconfig['Haywrd']['traindata']
    data_file, img_file = traindata
    data, desc, kp, desc_sift, kp_sift, patch_size, stride = pickle.load(open("%s.pkl"%data_file,'rb'))
    a_mat = sparse.load_npz("%s.npz"%data_file)
    trainset = SAR_TrainDataSet(img_file, data, a_mat, desc, kp, desc_sift, kp_sift, patch_size, stride)
    img, img2, label = trainset[300]
    # img = img.numpy()*255
    # img = img.astype(np.uint8).transpose(1,2,0)
    # # img = Image.fromarray(img)
    # aa = 1

    # # homography test
    # rho          = 32
    # patch_size   = 224
    # top_point    = (32,32)
    # left_point   = (patch_size+32, 32)
    # bottom_point = (patch_size+32, patch_size+32)
    # right_point  = (32, patch_size+32)
    # four_points = [top_point, left_point, bottom_point, right_point]
    # test_image = img.copy()

    # perturbed_four_points = []
    # for point in four_points:
    #     perturbed_four_points.append((point[0] + random.randint(-rho,rho), point[1]+random.randint(-rho,rho)))
    # H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )
    # H_inverse = np.linalg.inv(H)
    # warped_image = cv2.warpPerspective(img,H_inverse, (224,224))
    # annotated_warp_image = warped_image.copy()
    # Ip1 = test_image[top_point[1]:bottom_point[1],top_point[0]:bottom_point[0]]
    # Ip2 = warped_image[top_point[1]:bottom_point[1],top_point[0]:bottom_point[0]]
    # Image.fromarray(Ip1).show()
    # Image.fromarray(Ip2).show()
