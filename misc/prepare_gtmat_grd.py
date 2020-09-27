import argparse
import numpy as np
import os
import sys
from shapely.geometry import Polygon

from scipy.sparse import dok_matrix, save_npz

# import matplotlib.pyplot as plt
# from PIL import Image
# import geopandas as gpd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import loader_grd
import configs

def main():
    assert args.dataname in ['Haywrd','ykdelB']
    traindata = configs.dataconfig[args.dataname]['traindata']
    testdata = configs.dataconfig[args.dataname]['testdata']

    trainloader = loader_grd.setup_testloader(traindata, args)
    testloader = loader_grd.setup_testloader(testdata, args)
    ntrain = len(trainloader.dataset)
    ntest = len(testloader.dataset)
    traindata = trainloader.dataset
    testdata = testloader.dataset

    gtmat = np.zeros((ntest, ntrain), dtype=np.int_)
    gtmat = dok_matrix((ntest, ntrain), dtype=np.int_) # sparse matrix

    poly_test = []
    poly_train = []
    for itest in range(ntest):
        geo_itest = testdata.get_coordinates(itest)
        poly_test.append(Polygon([(geo_itest['lowerleft'][0], geo_itest['lowerleft'][1]),\
                              (geo_itest['lowerright'][0], geo_itest['lowerright'][1]),\
                              (geo_itest['upperright'][0], geo_itest['upperright'][1]),\
                              (geo_itest['upperleft'][0], geo_itest['upperleft'][1])])
        )
    for itrain in range(ntrain):
        geo_itrain = traindata.get_coordinates(itrain)
        poly_train.append(Polygon([(geo_itrain['lowerleft'][0], geo_itrain['lowerleft'][1]),\
                                (geo_itrain['lowerright'][0], geo_itrain['lowerright'][1]),\
                                (geo_itrain['upperright'][0], geo_itrain['upperright'][1]),\
                                (geo_itrain['upperleft'][0], geo_itrain['upperleft'][1])])
        )
    for itest in range(ntest):
        for itrain in range(ntrain):
            if poly_test[itest].intersects(poly_train[itrain]):
                gtmat[itest, itrain] = 1
                print(itest, itrain)


    save_npz("%s_gtmat.npz"%(args.dataname), gtmat.tocsr())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')
    parser.add_argument('--bstest', type=int, default=128, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=16, help='the number of workers used in DataLoader')
    args = parser.parse_args()
    main()