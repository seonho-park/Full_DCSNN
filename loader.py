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
import glob

class SAR_DataSet_POLSAR1(torchdata.Dataset):    
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = np.load(os.path.join(self.img_path, "%04d.npy"%(idx)))
        patch = Image.fromarray(patch)
        patch = transforms.functional.to_tensor(patch)
        return patch, idx


class SAR_DataSet(torchdata.Dataset):    
    def __init__(self, img_path, patch_size=224):
        self.img_path = img_path
        self.patch_size = patch_size
        self.data = glob.glob(os.path.join(img_path, '*.png'))
 
    def get_data(self,idx):
        data = pickle.load(open(os.path.join(self.img_path, "%04d.pkl"%(idx)),'rb'))
        return data

    def get_image(self,idx):
        return np.load(os.path.join(self.img_path, "%04d.npy"%(idx)))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = Image.open(os.path.join(self.img_path, "%04d.png"%(idx)))
        patch = transforms.functional.to_tensor(patch)
        return patch, idx

# reference: https://github.com/mazenmel/Deep-homography-estimation-Pytorch/blob/master/DataGenerationAndProcessing.py
rho          = 32
patch_size   = 160
top_point    = (32,32)
left_point   = (patch_size+32, 32)
bottom_point = (patch_size+32, patch_size+32)
right_point  = (32, patch_size+32)
four_points = [top_point, left_point, bottom_point, right_point]



class SAR_TrainDataSet_POLSAR1(SAR_DataSet_POLSAR1):
    def __init__(self, img_path, data, a_mat, desc, kp, desc_sift, kp_sift, patch_size=800, stride=200, homography = True):
        super(SAR_TrainDataSet_POLSAR1,self).__init__(img_path, data, desc, kp, desc_sift, kp_sift, patch_size, stride)
        self.sim_mat = a_mat
        self.homography = homography

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_path, "%04d.npy"%(idx)))
        
        hflip = True if random.random() < 0.5 else False
        vflip = True if random.random() < 0.5 else False
        
        img1 = Image.fromarray(img)
        img1 = transforms.functional.to_tensor(img1)

        # img1, H1 = self.warp_image(img, hflip, vflip) # homography
        # img1 = Image.fromarray(img1)
        # img1 = transforms.functional.to_tensor(img1)

        # # warp 2
        if self.homography:
            img2, H2 = self.warp_image(img, hflip, vflip)
            img2 = Image.fromarray(img2)
            img2 = transforms.functional.to_tensor(img2)
            return img2, img1, idx
        else:
            return img1, img1, idx


    def get_sim_mat(self, idxs, device):
        sim_mat = np.zeros((idxs.shape[0], self.sim_mat.get_shape()[0]))
        for (i,), idx in np.ndenumerate(idxs):
            sim_mat[i,:] = self.sim_mat.getrow(idx).todense().squeeze()
        
        return torch.FloatTensor(sim_mat).to(device)

    def warp_image(self, img, hflip, vflip):
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-rho,rho), point[1]+random.randint(-rho,rho)))
        if hflip:
            perturbed_four_points[0], perturbed_four_points[1] = perturbed_four_points[1], perturbed_four_points[0]
            perturbed_four_points[2], perturbed_four_points[3] = perturbed_four_points[3], perturbed_four_points[2]
        if vflip:
            perturbed_four_points[0], perturbed_four_points[3] = perturbed_four_points[3], perturbed_four_points[0]
            perturbed_four_points[1], perturbed_four_points[2] = perturbed_four_points[2], perturbed_four_points[1]

        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inv = np.linalg.inv(H)
        warped_image = cv2.warpPerspective(img, H_inv, (224,224))
        return warped_image, H_inv
        


class SAR_TrainDataSet(SAR_DataSet):
    def __init__(self, img_path, a_mat, patch_size=224):
        super(SAR_TrainDataSet,self).__init__(img_path, patch_size)
        self.sim_mat = a_mat

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, "%04d.png"%(idx)))
        data = pickle.load(open(os.path.join(self.img_path, "%04d.pkl"%(idx)),'rb'))

        hflip = True if random.random() < 0.5 else False
        vflip = True if random.random() < 0.5 else False

        img = np.array(img)

        kp = data['keypoint']
        kp_img = np.zeros((self.patch_size, self.patch_size),dtype=np.uint8)
        nkp = kp.shape[0]
        for i in range(nkp):
            kp_img[int(kp[i,1]), int(kp[i,0])] = 255
        kp_img = Image.fromarray(kp_img, mode='L').resize((224,224))

        # warp
        img_warp, H1 = self.warp_image(img, hflip, vflip) # homography
        # img = Image.fromarray(img)
        img_warp = Image.fromarray(img_warp)
        # img = transforms.functional.to_tensor(img)
        img_warp = transforms.functional.to_tensor(img_warp)

        # # warp 2
        img2, H2 = self.warp_image(img, hflip, vflip)
        img2 = Image.fromarray(img2)
        img2 = transforms.functional.to_tensor(img2)

        # return img, img_warp, idx
        return img_warp, img2, idx

    def warp_image(self, img, hflip, vflip):
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-rho,rho), point[1]+random.randint(-rho,rho)))
        if hflip:
            perturbed_four_points[0], perturbed_four_points[1] = perturbed_four_points[1], perturbed_four_points[0]
            perturbed_four_points[2], perturbed_four_points[3] = perturbed_four_points[3], perturbed_four_points[2]
        if vflip:
            perturbed_four_points[0], perturbed_four_points[3] = perturbed_four_points[3], perturbed_four_points[0]
            perturbed_four_points[1], perturbed_four_points[2] = perturbed_four_points[2], perturbed_four_points[1]

        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inv = np.linalg.inv(H)
        warped_image = cv2.warpPerspective(img, H_inv, (224,224))
        return warped_image, H_inv

    def get_sim_mat(self, idxs, device):
        sim_mat = np.zeros((idxs.shape[0], self.sim_mat.get_shape()[0]))
        for (i,), idx in np.ndenumerate(idxs):
            sim_mat[i,:] = self.sim_mat.getrow(idx).todense().squeeze()
        
        return torch.FloatTensor(sim_mat).to(device)
        


def setup_trainloader(data_files, args):
    data_file, img_file = data_files
    data, desc, kp, desc_sift, kp_sift, patch_size, stride = pickle.load(open("%s.pkl"%data_file,'rb'))
    a_mat = sparse.load_npz("%s_A_mat.npz"%data_file)
    trainset = SAR_TrainDataSet_POLSAR1(img_file, data, a_mat, desc, kp, desc_sift, kp_sift, patch_size, stride)

    # trainset = SAR_TrainDataSet(img_file, a_mat)
    
    print('num train data:', len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    return trainloader


def setup_testloader(data_files, args):
    data_file, img_file = data_files
    data, desc, kp, desc_sift, kp_sift, patch_size, stride  = pickle.load(open("%s.pkl"%data_file,'rb'))
    testset = SAR_DataSet_POLSAR1(img_file, data, desc, kp, desc_sift, kp_sift, patch_size, stride)
    # testset = SAR_DataSet(img_file)
    print('num test data:', len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    return testloader


if __name__ == '__main__':
    from configs import *
    import numpy as np

    traindata = dataconfig['ykdelB']['traindata']
    data_file, img_file = traindata
    # data, desc, kp, desc_sift, kp_sift, patch_size, stride = pickle.load(open("%s.pkl"%data_file,'rb'))
    a_mat = sparse.load_npz("%s_A_mat.npz"%data_file)
    trainset = SAR_TrainDataSet(img_file, a_mat)
    img, img2, label = trainset[800]
    img, img2, label = trainset[301]
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
