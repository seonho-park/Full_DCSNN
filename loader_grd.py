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

class SAR_DataSet(torchdata.Dataset):    
    def __init__(self, img_path, patch_size=224):
        self.img_path = img_path
        self.patch_size = patch_size
        self.data = glob.glob(os.path.join(img_path, '*.png'))
 
    def get_data(self,idx):
        data = pickle.load(open(os.path.join(self.img_path, "%04d.pkl"%(idx)),'rb'))
        return data

    def get_coordinates(self, idx):
        data = self.get_data(idx)
        return data["coordinates"]

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


class SAR_TrainDataSet(SAR_DataSet):
    def __init__(self, img_path, a_mat, patch_size=224, homography = True):
        super(SAR_TrainDataSet,self).__init__(img_path, patch_size)
        self.sim_mat = a_mat
        self.homography = homography

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, "%04d.png"%(idx)))
        img = np.array(img)

        data = pickle.load(open(os.path.join(self.img_path, "%04d.pkl"%(idx)),'rb'))
        kp = data['keypoint']
        kp_img = np.zeros((self.patch_size, self.patch_size),dtype=np.uint8)
        nkp = kp.shape[0]
        for i in range(nkp):
            kp_img[int(kp[i,1]), int(kp[i,0])] = 255
        kp_img = Image.fromarray(kp_img, mode='L').resize((224,224))

        if self.homography:
            # warp 1
            hflip = True if random.random() < 0.5 else False
            vflip = True if random.random() < 0.5 else False
            img_warp1, H1 = self.warp_image(img, hflip, vflip) # homography
            img_warp1 = Image.fromarray(img_warp1)
            img_warp1 = transforms.functional.to_tensor(img_warp1)

            # warp 2
            hflip = True if random.random() < 0.5 else False
            vflip = True if random.random() < 0.5 else False
            img_warp2, H1 = self.warp_image(img, hflip, vflip) # homography
            img_warp2 = Image.fromarray(img_warp2)
            img_warp2 = transforms.functional.to_tensor(img_warp2)
            return img_warp1, img_warp2, idx
        else:
            img = Image.fromarray(img)
            img = transforms.functional.to_tensor(img)
            return img, img, idx


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
    if args.sartype.lower() in ['grd']:
        a_mat = sparse.load_npz("%s_A_mat.npz"%data_file)
        trainset = SAR_TrainDataSet(img_file, a_mat, homography = not args.no_homography)
    elif args.sartype.lower() in ['mlc']:
        data, desc, kp, desc_sift, kp_sift, patch_size, stride = pickle.load(open("%s.pkl"%data_file,'rb'))
        a_mat = sparse.load_npz("%s_A_mat.npz"%data_file)
        import loader
        trainset = loader.SAR_TrainDataSet_POLSAR1(img_file, data, a_mat, desc, kp, desc_sift, kp_sift, patch_size, stride, homography = not args.no_homography)

    print('num train data:', len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bstrain, shuffle=True, num_workers = args.nworkers)
    return trainloader


def setup_testloader(data_files, args):
    data_file, img_file = data_files
    if args.sartype.lower() in ['grd']:
        testset = SAR_DataSet(img_file)
    elif args.sartype.lower() in ['mlc']:
        import loader
        data, desc, kp, desc_sift, kp_sift, patch_size, stride  = pickle.load(open("%s.pkl"%data_file,'rb'))
        testset = loader.SAR_DataSet_POLSAR1(img_file, data, desc, kp, desc_sift, kp_sift, patch_size, stride)
    print('num test data:', len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bstest, shuffle=False, num_workers = args.nworkers)
    return testloader


def load_gtmat(sartype, gtmat_fn):
    if sartype.lower() in ['grd']:
        return sparse.load_npz(gtmat_fn) # load groundtruth matrix
    elif sartype.lower() in ['mlc']:
        return np.load(gtmat_fn) 


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
