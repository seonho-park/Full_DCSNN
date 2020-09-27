import numpy as np
# import cv2
import scipy.ndimage
import math
import time
from skimage.color import rgb2gray

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.float64(np.exp(-((x**2 + y**2)/(2.0*sigma**2))))
    return g/g.sum()
    
def build_scale(image, sigma=2., Mmax=8, ratio=2**(1/3.), d=0.04):
    M,N = image.shape
    sar_harris_function = np.zeros((M,N,Mmax))
    gradient = np.zeros((M, N, Mmax))
    angle = np.zeros((M, N, Mmax))
    
    for i in range(Mmax):
        scale = float(sigma*ratio**(i))
        radius = int(round(2*scale))
        j = list(range(-radius,radius+1,1))
        k = list(range(-radius,radius+1,1))
        xarry,yarry = np.meshgrid(j,k)
        W = np.exp(-(np.abs(xarry)+np.abs(yarry))/scale)
        W34 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        W12 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        W14 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        W23 = np.zeros((2*radius+1,2*radius+1),dtype=float)
        
        W34[radius+1:2*radius+1,:] = W[radius+1:2*radius+1,:]
        W12[0:radius,:] = W[0:radius,:]
        W14[:,radius+1:2*radius+1] = W[:,radius+1:2*radius+1]
        W23[:,0:radius] = W[:,0:radius]

        M34 = scipy.ndimage.correlate(image, W34, mode='nearest')
        M12 = scipy.ndimage.correlate(image, W12, mode='nearest')
        M14 = scipy.ndimage.correlate(image, W14, mode='nearest')
        M23 = scipy.ndimage.correlate(image, W23, mode='nearest')
        
        Gx = np.log(M14/M23)
        Gy = np.log(M34/M12)
        
        Gx[np.where(np.imag(Gx))] = np.abs(Gx[np.where(np.imag(Gx))])
        Gy[np.where(np.imag(Gy))] = np.abs(Gy[np.where(np.imag(Gy))])
        Gx[np.where(np.isfinite(Gx)==0)] = 0
        Gy[np.where(np.isfinite(Gy)==0)] = 0
           
        gradient[:,:,i] = np.sqrt(np.square(Gx)+np.square(Gy))
        temp_angle = np.arctan2(Gy, Gx)
        
        temp_angle = temp_angle/math.pi*180
        temp_angle[np.where(temp_angle<0)] = temp_angle[np.where(temp_angle<0)]+360
        angle[:,:,i] = temp_angle
        
        Csh_11 = scale**2 * np.square(Gx)
        Csh_12 = scale**2 * Gx*Gy
        Csh_22 = scale**2 * np.square(Gy)
        
        gaussian_sigma = math.sqrt(2)*scale
        width = round(3*gaussian_sigma)
        width_windows = int(2*width+1)
        W_gaussian = fspecial_gauss(width_windows,gaussian_sigma)
        
        l = list(range(0,width_windows,1))
        m = list(range(0,width_windows,1))
        a,b = np.meshgrid(l,m)
        index0,index1 = np.where((np.square(a-width)-1)+np.square(b-width- 1)>width**2)
        W_gaussian[index0,index1] = 0
        
        Csh_11 = scipy.ndimage.correlate(Csh_11, W_gaussian, mode='nearest')
        Csh_12 = scipy.ndimage.correlate(Csh_12, W_gaussian, mode='nearest')
        Csh_21 = Csh_12
        Csh_22 = scipy.ndimage.correlate(Csh_22, W_gaussian, mode='nearest')
        
        sar_harris_function[:,:,i] = Csh_11*Csh_22-Csh_21*Csh_12-d*(Csh_11+Csh_22)**2
        
    return sar_harris_function,gradient,angle

def calc_descriptors(gradient,angle,kps):
    circle_bin = 8
    LOG_DESC_HIST_BINS = 8
    
    # M = key_point_array.shape[0]
    M = len(kps)
    d = circle_bin
    n = LOG_DESC_HIST_BINS
    descriptors = np.zeros((M,(2*d+1)*n),dtype=np.float32)

    for i, kp in enumerate(kps):
        x,y,scale,layer,main_angle,_ = kp.squeeze().tolist()
        current_grad = gradient[...,int(layer)]
        current_angle = angle[...,int(layer)]
        descriptors[i,:] = calc_log_polar_descriptor(current_grad,current_angle,int(x),int(y),scale,main_angle,d,n)
    return descriptors

    # locs = key_point_array
    
    # for i in range(M):
    #     x = int(key_point_array[i,0])
    #     y = int(key_point_array[i,1])
    #     scale = key_point_array[i,2]
    #     layer = int(key_point_array[i,3])
    #     main_angle = key_point_array[i,4]
    #     current_gradient = gradient[:,:,layer]
    #     current_angle = angle[:,:,layer]
    #     descriptors[i,:] = calc_log_polar_descriptor(current_gradient,current_angle,x,y,scale,main_angle,d,n)
            
    # return descriptors

def calc_log_polar_descriptor(gradient,angle,x,y,scale,main_angle,d,n):
    
    cos_t = math.cos(-main_angle/180.*math.pi)
    sin_t = math.sin(-main_angle/180.*math.pi)
    
    M,N = gradient.shape
    radius = int(round(min(12*scale,min(M/2,N/2))))
    
    radius_x_left = x-radius
    radius_x_right = x+radius+1
    radius_y_up = y-radius
    radius_y_down = y+radius+1
    
    if radius_x_left <0:
        radius_x_left = 0
    if radius_x_right>=N:
        radius_x_right = N
    if radius_y_up <0:
        radius_y_up = 0
    if radius_y_down >= M:
        radius_y_down = M
    
    center_x = x-radius_x_left 
    center_y = y-radius_y_up 
    
    sub_gradient = gradient[radius_y_up:radius_y_down,radius_x_left:radius_x_right]
    sub_angle = angle[radius_y_up:radius_y_down,radius_x_left:radius_x_right]
    sub_angle = np.round((sub_angle-main_angle)*n/360)
    sub_angle[sub_angle<0] = sub_angle[sub_angle<0] + n
    sub_angle[sub_angle==0] = n

    X = list(range(-(x-radius_x_left),radius_x_right-x,1)) 
    Y = list(range(-(y-radius_y_up),radius_y_down-y,1))
    [XX,YY] = np.meshgrid(X,Y)
    c_rot = XX*cos_t - YY*sin_t
    r_rot = XX*sin_t + YY*cos_t
    
    log_angle = np.arctan2(r_rot,c_rot)
    log_angle = log_angle/math.pi*180.
    log_angle[log_angle<0] = log_angle[log_angle<0] +360
    np.seterr(divide='ignore')
    log_amplitude = np.log2(np.sqrt(np.square(c_rot)+np.square(r_rot)))
    
    log_angle = np.round(log_angle*d/360.)
    log_angle[log_angle<=0] = log_angle[log_angle<=0] + d
    log_angle[log_angle>d] = log_angle[log_angle>d] - d

    r1 = math.log(radius*0.25,2)
    r2 = math.log(radius*0.73,2)
    log_amplitude[log_amplitude<=r1] = 1   
    log_amplitude[(log_amplitude>r1) * (log_amplitude<=r2)] =2
    log_amplitude[log_amplitude>r2] = 3
    
    temp_hist = np.zeros(((2*d+1)*n,1))
    row,col = log_angle.shape
    
    for i in range(row):
        for j in range(col):
            if (i-center_y)**2+(j-center_x)**2 <=radius**2:
                angle_bin = log_angle[i,j]
                amplitude_bin = log_amplitude[i,j]
                bin_vertical = sub_angle[i,j]
                Mag = sub_gradient[i,j]
                
                if amplitude_bin==1:
                    temp_hist[int(bin_vertical)-1] =  temp_hist[int(bin_vertical)-1] + Mag
                else:
                    temp_hist[int(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)-1] = temp_hist[int(((amplitude_bin-2)*d+angle_bin-1)*n+bin_vertical+n)-1] + Mag
                
    temp_hist = temp_hist/np.sqrt(np.dot(temp_hist.T,temp_hist))
    temp_hist[temp_hist>0.2] = 0.2
    temp_hist = temp_hist/np.sqrt(np.dot(temp_hist.T,temp_hist))
    descriptor = temp_hist.reshape(-1)
    
    return descriptor

def find_scale_extreme(sar_harris_function, gradient, angle, threshold=0.8, sigma=2., ratio=2**(1/3.)):
    M,N,num = sar_harris_function.shape
    BORDER_WIDTH = 1
    # HIST_BIN = 36
    HIST_BIN = 12
    SIFT_ORI__PEAK_RATIO = 0.8
    # key_number = 0
    # key_point_array = np.zeros((M*N,num-2))
    # key_point_array = np.zeros((M*N,6))
    kps = []
    for i in range(num):
        temp_current = sar_harris_function[:,:,i]
        gradient_current = gradient[:,:,i]
        angle_current = angle[:,:,i]
        for j in range(BORDER_WIDTH,M-BORDER_WIDTH,1):
            for k in range(BORDER_WIDTH,N-BORDER_WIDTH,1):
                temp = temp_current[j,k]
                if temp>=threshold \
                   and temp>=temp_current[j-1,k-1] and temp>=temp_current[j-1,k] and temp>=temp_current[j-1,k+1] \
                   and temp>=temp_current[j,k-1] and temp>=temp_current[j,k+1] \
                   and temp>=temp_current[j+1,k-1] and temp>=temp_current[j+1,k] and temp>=temp_current[j+1,k+1] :

                    scale = sigma*ratio**(i+1)
                    hist, max_value = calculate_orientation_hist(k,j,scale,gradient_current,angle_current,HIST_BIN)
                    
                    mag_thr = max_value*SIFT_ORI__PEAK_RATIO
                    cand_idxs = np.argwhere((hist>=mag_thr) & (hist>np.roll(hist,1)) & (hist>np.roll(hist,-1))).squeeze()
                    for idx in np.nditer(cand_idxs): 
                        kp = np.asarray([k, j, sigma*ratio**(i), i, 360/HIST_BIN*idx, hist[idx]])
                        kps.append(kp.reshape(1,-1))

    return kps

def calculate_orientation_hist(x,y,scale,gradient,angle,n):
    M,N = gradient.shape
    radius = int(round(min(6*scale,min(M/2,N/2))))
    sigma = 2*scale
    radius_x_left = x-radius
    radius_x_right = x+radius+1
    radius_y_up = y-radius
    radius_y_down = y+radius+1
    
    if radius_x_left <0:
        radius_x_left = 0
    if radius_x_right>=N:
        radius_x_right = N
    if radius_y_up <0:
        radius_y_up = 0
    if radius_y_down >= M:
        radius_y_down = M
        
    center_x = x-radius_x_left 
    center_y = y-radius_y_up 
    
    sub_gradient = gradient[radius_y_up:radius_y_down,radius_x_left:radius_x_right]
    sub_angle = angle[radius_y_up:radius_y_down,radius_x_left:radius_x_right]
    
    #X = list(range(-(x-radius_x_left),radius_x_right-x,1)) 
    #Y = list(range(-(y-radius_y_up),radius_y_down-y,1))
    #[XX,YY] = np.meshgrid(X,Y)
    W = sub_gradient
    bins = np.round(sub_angle*n/360.) #TODO: 360 is working properly?
    
    bins[bins>=n] = bins[bins>=n]-n
    bins[bins<0] = bins[bins<0]+n

    tem_hist = np.zeros(n)
    row,col = bins.shape

    for i in range(row):
        for j in range(col):
            if (i-center_y)**2+(j-center_x)**2 <=radius**2:
                tem_hist[int(bins[i,j])] = tem_hist[int(bins[i,j])] + W[i,j]

    

#smooth histogram
#TODO: looks incorrect
    hist = np.zeros(n)
    hist[0] = (tem_hist[n-2] + tem_hist[2])/16. + 4*(tem_hist[n-1]+tem_hist[1])/16. + tem_hist[0]*6/16. 
    hist[1] = (tem_hist[n-1] + tem_hist[3])/16. + 4*(tem_hist[0]+tem_hist[2])/16. + tem_hist[1]*6/16.
    hist[2:n-2] = (tem_hist[0:n-4] + tem_hist[4:n])/16. + 4*(tem_hist[1:n-3]+tem_hist[3:n-1])/16. + tem_hist[2:n-2]*6/16.
    # hist[n-2] = (tem_hist[n-4] + tem_hist[0])/16. + 4*(tem_hist[n-3]+tem_hist[n-1])/16. + tem_hist[1]*6/16. #TODO: looks incorrect
    # hist[n-1] = (tem_hist[n-3] + tem_hist[1])/16. + 4*(tem_hist[n-1]+tem_hist[0])/16. + tem_hist[n-1]*6/16.
    hist[n-2] = (tem_hist[n-4] + tem_hist[0])/16. + 4*(tem_hist[n-3]+tem_hist[n-1])/16. + tem_hist[n-2]*6/16.
    hist[n-1] = (tem_hist[n-3] + tem_hist[1])/16. + 4*(tem_hist[n-2]+tem_hist[0])/16. + tem_hist[n-1]*6/16.

    max_value = np.amax(hist)
    return hist, max_value

def delete_duplicate(kps): # new
    kps_coord = np.concatenate(kps)[:,:2]
    _, idxs = np.unique(kps_coord, axis=0, return_index = True)
    kps_new = list( kps[i] for i in idxs.tolist() )
    return kps_new


def delete_duplications(kp1,kp2,des1,des2): # old
    temp_index = []
    for i in range(kp1.shape[0]):
        for j in range(i+1,kp1.shape[0],1):
            if i!=j and (kp1[i]==kp1[j]).all():
               temp_index.append(j)    
    temp = list(set(temp_index))  
    kp1_ = np.delete(kp1,temp,0)
    des1_ = np.delete(des1,temp,0)
    
    temp_index = []
    for k in range(kp2.shape[0]):
        for l in range(k+1,kp2.shape[0],1):
            if k!=l and (kp2[k]==kp2[l]).all():
               temp_index.append(l)               
    temp = list(set(temp_index))  
    kp2_ = np.delete(kp2,temp,0)
    des2_ = np.delete(des2,temp,0)
    return kp1_,kp2_,des1_,des2_

def deep_match(kp1_location,kp2_location,deep_des1,deep_des2,ratio):
    deep_kp1 = []
    deep_kp2 = []
    for i in range(deep_des1.shape[0]):
        des = np.tile(deep_des1[i],(deep_des2.shape[0],1))
        error = des - deep_des2
        RMSE = np.sqrt(np.sum(np.square(error),axis=1)/float(error.shape[1]))
        small_index = np.argsort(RMSE, axis=0)
        if RMSE[small_index[0]]< RMSE[small_index[1]]*ratio:
            deep_kp1.append(np.asarray((kp1_location[i][0],kp1_location[i][1])))
            deep_kp2.append(np.asarray((kp2_location[small_index[0]][0],kp2_location[small_index[0]][1])))
            #deep_des2 = np.delete(deep_des2, small_index[0], 0)
    return np.concatenate(deep_kp1),np.concatenate(deep_kp2)

def detect_and_extract(img, Mmax = 2, threshold = 0.9, sigma = 0.63, duplicate = True, verbose = False):
    t0 = time.time()
    img = rgb2gray(img)
    sar_harris_function, gradient, angle = build_scale(img, sigma = sigma, Mmax = Mmax)
    t1 = time.time()
    if verbose:
        print('TIME build scale: %.4f'%(t1-t0))
    kps = find_scale_extreme(sar_harris_function, gradient, angle, threshold=threshold, sigma = sigma)
    t2 = time.time()
    if verbose:
        print('TIME find scale extreme: %.4f'%(t2-t1), 'nkp:', len(kps))
    
    if len(kps) == 0:
        return False

    if not duplicate:
        t21 = time.time()
        kps = delete_duplicate(kps)
        t22 = time.time()
        if verbose:
            print('TIME delete_duplicate: %.4f'%(t22-t21), 'nkp:', len(kps))

    descriptor = calc_descriptors(gradient, angle, kps)
    t3 = time.time()
    if verbose:
        print('TIME calc descriptor: %.4f'%(t3-t2))
    kps_output = np.concatenate(kps)[:,:2]
    if verbose:
        print("descriptor shape, keypoint shape", descriptor.shape, kps_output.shape)
    return descriptor, kps_output

def detect(img, Mmax = 2, threshold = 0.9, sigma = 0.63, duplicate = True, verbose = False):
    t0 = time.time()
    img = rgb2gray(img)
    sar_harris_function, gradient, angle = build_scale(img, sigma = sigma, Mmax = Mmax)
    t1 = time.time()
    if verbose:
        print('TIME build scale: %.4f'%(t1-t0))
    kps = find_scale_extreme(sar_harris_function, gradient, angle, threshold=threshold, sigma = sigma)
    t2 = time.time()
    if verbose:
        print('TIME find scale extreme: %.4f'%(t2-t1), 'nkp:', len(kps))
    
    if len(kps) == 0:
        return False

    if not duplicate:
        t21 = time.time()
        kps = delete_duplicate(kps)
        t22 = time.time()
        if verbose:
            print('TIME delete_duplicate: %.4f'%(t22-t21), 'nkp:', len(kps))
    kps_output = np.concatenate(kps)[:,:2]
    return kps_output

    
def test():
    from sift import SIFT
    from skimage.io import imread, imsave
    import loader
    import matplotlib.pyplot as plt

    # img = 5*np.ones((50,40))
    # img[:25,:25] = 20.
    
    # img = imread('./examples/jinco2.jpg') # load jinco
    # data = './data/ykdelB_26906_19066'
    # data = './data/Haywrd_15302_19031'
    # testloader = loader.setup_testloader(data, args)
    # dataset = testloader.dataset
    # # print('num train data:', len(dataset))
    # ndataset = len(dataset)
    # img = dataset.get_image(20)
    # imsave('./examples/Haywrd_15302_19031_example.png', img)

    img = imread('./examples/Haywrd_15302_19031_example.png') # load jinco
    
    # print('image shape', img.shape)
    desc, kp = detect_and_extract(img, Mmax = 4, threshold = 5., duplicate = False)

    sift_detector = SIFT(img, num_octave=8)
    kp_sift, desc_sift = sift_detector.compute()

    
    f, ax = plt.subplots(1,2,figsize=(15,15))
    ax[0].imshow(img)
    ax[0].scatter(kp[:,0], kp[:,1], c='r', s=2.5)
    ax[0].axis('off')
    ax[1].imshow(img)
    ax[1].scatter(kp_sift[:,0], kp_sift[:,1], c='r', s=2.5)
    ax[1].axis('off')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bstest', type=int, default=32, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=8, help='the number of workers used in DataLoader')
    args = parser.parse_args()

    test()