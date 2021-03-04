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
        img = Image.open(os.path.join(self.img_path, "%04d.png"%(idx)))
        img = np.array(img)
        # normalize img1
        # img = (img-img.min())/(img.max()-img.min())
        # img = (255.*img).astype(np.uint8)

        img = Image.fromarray(img)
        img = transforms.functional.to_tensor(img)

        # keypoint
        data = pickle.load(open(os.path.join(self.img_path, "%04d.pkl"%(idx)),'rb'))
        kp = data['keypoint']
        kp_img = np.zeros((self.patch_size, self.patch_size),dtype=np.uint8)
        nkp = kp.shape[0]
        for i in range(nkp):
            kp_img[int(kp[i,1]), int(kp[i,0])] = 255
        kp_img = Image.fromarray(kp_img, mode='L')

        # filtered ground truth keypoint
        kp_img = kp_img.filter(ImageFilter.GaussianBlur(radius=1))
        kp_img = np.array(kp_img)
        kp_img = (kp_img - kp_img.min()) / (kp_img.max() - kp_img.min())
        # kp_img = (255.*kp_img).astype(np.uint8)
        # kp_img = Image.fromarray(kp_img, mode = 'L')
        kp_img = transforms.functional.to_tensor(kp_img)
        return img, kp_img, idx

# reference: https://github.com/mazenmel/Deep-homography-estimation-Pytorch/blob/master/DataGenerationAndProcessing.py
rho          = 32
patch_size   = 160
top_point    = (32,32)
left_point   = (patch_size+32, 32)
bottom_point = (patch_size+32, patch_size+32)
right_point  = (32, patch_size+32)
four_points = [top_point, left_point, bottom_point, right_point]


class SAR_TrainDataSet(SAR_DataSet):
    def __init__(self, img_path, a_mat, patch_size=224):
        super(SAR_TrainDataSet,self).__init__(img_path, patch_size)
        self.sim_mat = a_mat

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.img_path, "%04d.png"%(idx)))
        img1 = np.array(img1)
        # normalize img1
        # img1 = (img1-img1.min())/(img1.max()-img1.min())
        # img1 = (255.*img1).astype(np.uint8)

        img = Image.fromarray(img1)
        img = transforms.functional.to_tensor(img)

        # warp - homography
        hflip = True if random.random() < 0.5 else False
        vflip = True if random.random() < 0.5 else False
        # hflip = True
        # vflip = False
        img_warp, H_inv, H, H4 = self.warp_image(img1, hflip, vflip) # homography
        # a = np.matmul(H, np.array([50.,160.,1.]))
        # a /= a[2]

        # b = np.matmul(H, np.array([0.,120, 1.]))
        # b /= b[2]

        # print(H4.min(), H4.max())
        img_warp = Image.fromarray(img_warp)
        img_warp = transforms.functional.to_tensor(img_warp)

        # filtered ground truth keypoint
        data = pickle.load(open(os.path.join(self.img_path, "%04d.pkl"%(idx)),'rb'))
        kp = data['keypoint']
        kp_img = np.zeros((self.patch_size, self.patch_size),dtype=np.uint8)
        nkp = kp.shape[0]
        for i in range(nkp):
            kp_img[int(kp[i,1]), int(kp[i,0])] = 255 # kp is in x,y whereas the image is h(y), w(x)
         
        kp_img_warp = cv2.warpPerspective(kp_img, H_inv, (self.patch_size,self.patch_size))

        kp_img = Image.fromarray(kp_img, mode='L')
        kp_img = kp_img.filter(ImageFilter.GaussianBlur(radius=1))
        kp_img = np.array(kp_img)
        kp_img = (kp_img - kp_img.min()) / (kp_img.max() - kp_img.min())
        kp_img = (255.*kp_img).astype(np.uint8)
        
        kp_img_warp = Image.fromarray(kp_img_warp, mode='L')
        kp_img_warp = kp_img_warp.filter(ImageFilter.GaussianBlur(radius=1))
        kp_img_warp = np.array(kp_img_warp)
        if kp_img_warp.max() != kp_img_warp.min():
            kp_img_warp = (kp_img_warp - kp_img_warp.min()) / (kp_img_warp.max() - kp_img_warp.min())
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        #         kp_img_warp2 = (kp_img_warp2 - kp_img_warp2.min()) / (kp_img_warp2.max() - kp_img_warp2.min())
        #     except RuntimeWarning:
        #         print(kp_img_warp2.max() - kp_img_warp2.min())
        
        kp_img_warp = (255.*kp_img_warp).astype(np.uint8)

        kp_img = transforms.functional.to_tensor(kp_img)
        kp_img_warp = transforms.functional.to_tensor(kp_img_warp)

        return img, img_warp, kp_img, kp_img_warp, idx, H, H4
        

    def warp_image(self, img, hflip, vflip):
        h,w,_ = img.shape
        # hflip = False
        # vflip = False
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
        H /= H.sum()
        H_inv = np.linalg.inv(H)
        # H_inv2 = np.linalg.inv(H/H.sum())
        warped_image = cv2.warpPerspective(img, H_inv, (h,w))
        # warped_image2 = cv2.warpPerspective(img, H_inv/H_inv.sum(), (h,w))
        # warped_image3 = cv2.warpPerspective(img, H_inv2, (h,w))
        # x = np.array([0.,0.,1.]) # [h, w, 0]
        # x_ = np.matmul(H,x)
        # x_ /= x_[2]

        H4 = [((perturbed_pt[0]-pt[0])/h, (perturbed_pt[1]-pt[1])/w)for perturbed_pt, pt in zip(perturbed_four_points, four_points) ]
        
        return warped_image, H_inv, H, np.array(H4)

    def get_sim_mat(self, idxs, device):
        sim_mat = np.zeros((idxs.shape[0], self.sim_mat.get_shape()[0]))
        for (i,), idx in np.ndenumerate(idxs):
            sim_mat[i,:] = self.sim_mat.getrow(idx).todense().squeeze()
        
        return torch.FloatTensor(sim_mat).to(device)
        


def setup_trainloader(data_files, args):
    data_file, img_file = data_files
    if args.sartype.lower() in ['grd']:
        a_mat = sparse.load_npz("%s_A_mat.npz"%data_file)
        trainset = SAR_TrainDataSet(img_file, a_mat)
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

    traindata = dataconfig['grd']['Haywrd']['traindata']
    data_file, img_file = traindata
    
    # data, desc, kp, desc_sift, kp_sift, patch_size, stride = pickle.load(open("%s.pkl"%data_file,'rb'))
    a_mat = sparse.load_npz("%s_A_mat.npz"%data_file)
    trainset = SAR_TrainDataSet(img_file, a_mat)
    # img, img2, label = trainset[800]
    aa = trainset[6539]
    # img, img2, label = trainset[301]
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
