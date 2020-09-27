import argparse
import numpy as np
import os
import sys
from shapely.geometry import Polygon
import geopandas as gpd

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import loader
import utils
import configs

def main():
    assert args.dataname in ['Haywrd','ykdelB']
    traindata = configs.dataconfig[args.dataname]['traindata']
    testdata = configs.dataconfig[args.dataname]['testdata']

    trainloader = loader.setup_testloader(traindata, args)
    testloader = loader.setup_testloader(testdata, args)
    ntrain = len(trainloader.dataset)
    ntest = len(testloader.dataset)
    traindata = trainloader.dataset
    testdata = testloader.dataset

    gtmat = np.zeros((ntest, ntrain), dtype=np.int_)

    for itest in range(200,ntest):
        # geo_itest = testloader.dataset.get_data(itest)['georeference']
        geo_itest = testdata.get_georeference(itest)
        # Polygon([])
        poly_itest = Polygon([(geo_itest['lowerleft'][0], geo_itest['lowerleft'][1]),\
                              (geo_itest['lowerright'][0], geo_itest['lowerright'][1]),\
                              (geo_itest['upperright'][0], geo_itest['upperright'][1]),\
                              (geo_itest['upperleft'][0], geo_itest['upperleft'][1])])
        poly_itest_ = gpd.GeoSeries(poly_itest)
 
        for itrain in range(200,ntrain):
            # geo_itrain = trainloader.dataset.get_data(itrain)['georeference']
            geo_itrain = traindata.get_georeference(itrain)
            poly_itrain = Polygon([(geo_itrain['lowerleft'][0], geo_itrain['lowerleft'][1]),\
                                   (geo_itrain['lowerright'][0], geo_itrain['lowerright'][1]),\
                                   (geo_itrain['upperright'][0], geo_itrain['upperright'][1]),\
                                   (geo_itrain['upperleft'][0], geo_itrain['upperleft'][1])])
            poly_itrain_ = gpd.GeoSeries(poly_itrain)
            if poly_itrain.intersects(poly_itest):
                # check another case
                print(poly_itest.intersects(poly_itrain),itest, itrain)
                print(utils.is_correct(geo_itest, geo_itrain))
                if not utils.is_correct(geo_itest, geo_itrain):
                    ax = poly_itest_.plot()
                    poly_itrain_.plot(ax=ax, color='red')
                    
                    
                    plt.show()


                    img_itest = np.load(os.path.join(testdata.img_path, "%04d.npy"%(itest)))
                    img_itest = Image.fromarray(img_itest)
                    # img_itest.show()

                    img_itrain = np.load(os.path.join(traindata.img_path, "%04d.npy"%(itrain)))
                    img_itrain = Image.fromarray(img_itrain)
                    
                    aa = 1
                    a = utils.is_correct(geo_itest, geo_itrain)
                # img_itrain.show()

                gtmat[itest,itrain] = 1


            # if utils.is_correct(geo_itest, geo_itrain):
            #     gtmat[itest,itrain] = 1

    # np.save(open("%s_gtmat.npy"%(args.dataname),'wb'), gtmat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Haywrd', help='data name Haywrd|ykdelB')
    parser.add_argument('--bstest', type=int, default=128, help='batch size for testing')
    parser.add_argument('--nworkers', type=int, default=16, help='the number of workers used in DataLoader')
    args = parser.parse_args()
    main()