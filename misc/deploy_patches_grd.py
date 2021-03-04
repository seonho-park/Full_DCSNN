import numpy as np
import os
from PIL import Image
from skimage.util.shape import view_as_windows
import _pickle as pickle
from scipy.sparse import dok_matrix, csr_matrix, save_npz
from shapely.geometry import Polygon

# temp
import geopandas as gpd
import matplotlib.pyplot as plt


import sar_sift2

# ul: Center Latitude & Longitude of Upper Left Pixel of GRD Image 
# spacing: GRD Latitude (row) & Longitude (column) Pixel Spacing
fns = { 
    # 'Haywrd_15302_18074': {"ul": (38.14032876, -122.43490512000001), 'spacing': (-5.556e-05, 5.556e-05)}, 
    # 'Haywrd_15302_19031': {"ul": (38.13821748, -122.43357168000001), 'spacing': (-5.556e-05, 5.556e-05)}
    'ykdelB_26906_18052': {"ul": (61.245510360000004, -164.24735751), 'spacing': (-5.556e-05, 0.00011111)},
    'ykdelB_26906_19066': {"ul": (61.245454800000005, -164.24780195), 'spacing': (-5.556e-05, 0.00011111)}
        }


patch_size = 600
patch_resize = 224
stride = 100
min_nkp = 80

savepath = '/media/sean/Storage/data/POLSAR_GRD'
inputpath = '/media/sean/Storage/data/POLSAR_GRD/RAW'
Image.MAX_IMAGE_PIXELS = None


def sarsift(*patches):
    # sarsift_R = sar_sift2.detect_and_extract(np.array(patch_R), Mmax = 4, threshold = 0.9, duplicate = False, verbose=False)
    kps = []
    for patch in patches:
        sarsift_out = sar_sift2.detect(np.array(patch), Mmax = 4, threshold = 1.5, duplicate = False, verbose=False)
        if isinstance(sarsift_out, bool) and sarsift_out is False:
            return False
        else:
            kps.append(sarsift_out)

    kps = np.concatenate(kps, axis=0)
    idxs = np.unique(kps, axis=0, return_index = True)[1].tolist()
    kps_new = np.array([kps[i] for i in idxs])

    return kps_new
            
def cal_lat_long(latlong_dict,i,j):
    # i: column index, j: row index
    lat = latlong_dict["ul"][0] + latlong_dict["spacing"][0]*j
    long = latlong_dict["ul"][1] + latlong_dict["spacing"][1]*i
    return lat, long



def process(file_symbol, latlong_dict):
    if not os.path.isdir(os.path.join(savepath,file_symbol)):
        os.mkdir(os.path.join(savepath,file_symbol))
    
    img_R = np.array(Image.open(os.path.join(inputpath, file_symbol, '%s_HHHH.jpg'%(file_symbol)))) # shape: height X width
    img_G = np.array(Image.open(os.path.join(inputpath, file_symbol, '%s_HVHV.jpg'%(file_symbol))))
    img_B = np.array(Image.open(os.path.join(inputpath, file_symbol, '%s_VVVV.jpg'%(file_symbol))))

    patches_R = view_as_windows(img_R, (patch_size,patch_size)).squeeze()
    patches_G = view_as_windows(img_G, (patch_size,patch_size)).squeeze()
    patches_B = view_as_windows(img_B, (patch_size,patch_size)).squeeze()
    assert patches_R.shape == patches_G.shape and patches_G.shape == patches_B.shape
    patch_dim = patches_R.shape[0:2]
    
    patch_id = 0
    patch_list = []
    for j in range(0,patch_dim[0],stride): # height - row direction
        for i in range(0,patch_dim[1],stride): # width - column direction
            
            patch_R = Image.fromarray(patches_R[j,i]).resize((patch_resize, patch_resize))
            patch_G = Image.fromarray(patches_G[j,i]).resize((patch_resize, patch_resize))
            patch_B = Image.fromarray(patches_B[j,i]).resize((patch_resize, patch_resize))

            kps = sarsift(patch_R, patch_G, patch_B)
            if kps is False or kps.shape[0] <= min_nkp:
                continue
            print("PATCH:%d | height:%d | width:%d | nkp:%d"%(patch_id, j, i, kps.shape[0]))
            patch_geographic_coordinates = {
                'upperleft' : cal_lat_long(latlong_dict,i,j),
                'upperright': cal_lat_long(latlong_dict,i+patch_size,j),
                'lowerleft' : cal_lat_long(latlong_dict,i,j+patch_size),
                'lowerright': cal_lat_long(latlong_dict,i+patch_size,j+patch_size)
            }
            patch_info = {
                'id': patch_id,
                'coordinates': patch_geographic_coordinates,
                'keypoint': kps.astype(np.uint8),
                'index': np.asarray([j,i])
            }
            patch = np.concatenate((np.array(patch_R)[...,np.newaxis], np.array(patch_G)[...,np.newaxis], np.array(patch_B)[...,np.newaxis]), axis=2)
            patch = Image.fromarray(patch)
            # kp_img = np.zeros((patch_resize, patch_resize),dtype=np.uint8)
            # nkp = kps.shape[0]
            # for i in range(nkp):
            #     kp_img[int(kps[i,1]), int(kps[i,0])] = 255
            # kp_img = Image.fromarray(kp_img, mode='L')
            patch.save(os.path.join(savepath,file_symbol,"%04d.png"%(patch_id)))
            pickle.dump(patch_info, open(os.path.join(savepath,file_symbol,"%04d.pkl"%(patch_id)),'wb'), protocol=3)
            patch_list.append(patch_info)
            patch_id += 1


    print("COMPOSE ADJACENCY MATRIX") 
    npatch = len(patch_list)
    print('NUM PATCHES', patch_id, len(patch_list))
    A_mat = dok_matrix((patch_id, patch_id), dtype=np.float32) # sparse matrix
    for i in range(npatch):
        id_i = patch_list[i]['id']
        coord_i = patch_list[i]['coordinates']
        poly_i = Polygon([(coord_i['lowerleft'][0], coord_i['lowerleft'][1]),\
                          (coord_i['lowerright'][0], coord_i['lowerright'][1]),\
                          (coord_i['upperright'][0], coord_i['upperright'][1]),\
                          (coord_i['upperleft'][0], coord_i['upperleft'][1])])
        area_i = poly_i.area
        for j in range(i,npatch):
            id_j = patch_list[j]['id']
            coord_j = patch_list[j]['coordinates']
            poly_j = Polygon([(coord_j['lowerleft'][0], coord_j['lowerleft'][1]),\
                              (coord_j['lowerright'][0], coord_j['lowerright'][1]),\
                              (coord_j['upperright'][0], coord_j['upperright'][1]),\
                              (coord_j['upperleft'][0], coord_j['upperleft'][1])])
            area_j = poly_j.area
            if poly_i.intersects(poly_j):
                area_ij = poly_i.intersection(poly_j).area
                print(id_i, id_j, patch_list[i]['index'], patch_list[j]['index'], area_ij, 2*area_ij/(area_i + area_j))
                # poly_i_ = gpd.GeoSeries(poly_i)
                # poly_j_ = gpd.GeoSeries(poly_j)
                # ax = poly_i_.plot()
                # poly_j_.plot(ax=ax, color='red')
                A_mat[id_i, id_j] = A_mat[id_j, id_i] = 2*area_ij/(area_i + area_j)
    
    print("... end")
    save_npz("%s/%s_A_mat.npz"%(savepath,file_symbol), A_mat.tocsr())


def main():
    for fn in fns:
        process(fn, fns[fn])


if __name__ == '__main__':
    main()