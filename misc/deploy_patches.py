import skimage
from skimage.util.shape import view_as_windows
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import sar_sift2
import _pickle as pickle
from scipy.sparse import dok_matrix, csr_matrix, save_npz

fns = { 
    'Haywrd_15302_18074': 
    np.asarray([[38.043598880, 38.135679773, 36.685931725, 36.776397359],
                [-122.433585116,-122.207729954,-121.563436042,-121.340585919]]),
    'Haywrd_15302_19031':
    np.asarray([[38.041462430, 38.133545644, 36.685560021, 36.776030089],
                [-122.432230663,-122.206364098,-121.563261348,-121.340396038]]),
    'ykdelB_26906_18052': 
    np.asarray([[61.032030532, 61.231101554, 61.042536665, 61.241673808],
                [-164.235928604,-164.247481682,-161.695712313,-161.691201160]]),
    'ykdelB_26906_19066':
    np.asarray([[61.032046883, 61.231145866, 61.042495873, 61.241660626],
                [-164.236208860,-164.247763839,-161.685132393,-161.680552162]])
}

patch_size = 600
patch_resize = 224
stride = 100
# savepath = '/home/sean/data/POLSAR2'
savepath = '/media/sean/Storage/data/POLSAR3'

def process(fn, location):
    basename = fn
    print('process...:', basename)
    if not os.path.isdir(os.path.join(savepath,basename)):
        os.mkdir(os.path.join(savepath,basename))
    
    # read 3 channels, VVVV, HVHV, HHHH corresponding to RGB respectively
    try:
        img_R = Image.open('../data/%s_VVVV.jpg'%(fn))
        img_G = Image.open('../data/%s_HVHV.jpg'%(fn))
        img_B = Image.open('../data/%s_HHHH.jpg'%(fn))
    except:
        img_R = Image.open('./data/%s_VVVV.jpg'%(fn))
        img_G = Image.open('./data/%s_HVHV.jpg'%(fn))
        img_B = Image.open('./data/%s_HHHH.jpg'%(fn))
    
    img_R = np.array(img_R)[...,np.newaxis]
    img_G = np.array(img_G)[...,np.newaxis]
    img_B = np.array(img_B)[...,np.newaxis]
    nh, nw = img_R.shape[:2] # height and width

    X = np.array([[0,0], [0,nw-1], [nh-1, 0], [nh-1, nw-1]])
    y_latitude = location[0]
    y_longitude = location[1]
    reg_latitude = LinearRegression().fit(X,y_latitude)
    reg_longitude = LinearRegression().fit(X,y_longitude)

    img = np.concatenate((img_R, img_G, img_B),axis=2)
    patches = view_as_windows(img, (patch_size,patch_size,3)).squeeze()
    patch_dim = patches.shape[0:2]
    

    patch_id = 0
    patch_list = []
    for j in range(50,patch_dim[0],stride):
        for i in range(10,patch_dim[1],stride):
            print("PATCH: %d"%patch_id, j, i)
            fullfn_png = os.path.join(savepath,basename,"%04d.png"%(patch_id))
            fullfn_pkl = os.path.join(savepath,basename,"%04d.pkl"%(patch_id))
            patch = patches[j,i]
            patch = Image.fromarray(patch)
            patch = patch.resize((patch_resize, patch_resize))
            sar_sift_out = sar_sift2.detect_and_extract(np.array(patch), Mmax = 4, threshold = 0.9, duplicate = False, verbose=False)
            if sar_sift_out is False:
                continue
            else:
                desc, kp = sar_sift_out
                if kp.shape[0]<=25: # not enough keypoints
                    continue
                
                X = np.asarray([[j,i], [j,i+patch_size], [j+patch_size,i], [j+patch_size,i+patch_size]])
                latitudes = reg_latitude.predict(X)
                longitudes = reg_longitude.predict(X)
                patch_georeference_dict = {
                    'upperleft': np.array([latitudes[0],longitudes[0]]),
                    'upperright': np.array([latitudes[1],longitudes[1]]),
                    'lowerleft': np.array([latitudes[2],longitudes[2]]),
                    'lowerright': np.array([latitudes[3],longitudes[3]])
                }
                patch_info = {
                    'id': patch_id,
                    'georeference': patch_georeference_dict,
                    # 'data': patch_data,
                    'keypoint': kp.astype(np.uint8),
                    'descriptor': desc.astype(np.float32),
                    'coordinate': np.asarray([j,i]),
                    'swath': j
                }
                patch.save(fullfn_png) # save img
                pickle.dump(patch_info, open(fullfn_pkl,'wb'), protocol=3)
                
                patch_list.append(patch_info)
                patch_id += 1

            # # visualize data
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # ax.imshow(np.array(patch))
            # ax.scatter(kp[:,0], kp[:,1], c='r', s=2.5)
            # plt.show()
    
    npatch = len(patch_list)
    print('num patches', patch_id, len(patch_list))
    print("compose A mat") 
    A_mat = dok_matrix((patch_id, patch_id), dtype=np.float32) # sparse matrix
    norm_max = patch_size/stride
    for i in range(npatch):
        for j in range(i,npatch):
            id_i = patch_list[i]['id']
            id_j = patch_list[j]['id']
            coord_i = patch_list[i]['coordinate']
            coord_j = patch_list[j]['coordinate']
            diff = (coord_i - coord_j)/stride
            diff_norm = np.linalg.norm(diff)
            if diff_norm < norm_max:
                A_mat[id_i, id_j] = (norm_max - diff_norm)/norm_max
                A_mat[id_j, id_i] = (norm_max - diff_norm)/norm_max

    print("... end")
    save_npz("%s/%s_A_mat.npz"%(savepath,basename), A_mat.tocsr())

def main():
    for fn in fns:
        process(fn, fns[fn])

if __name__ == '__main__':
    main()
