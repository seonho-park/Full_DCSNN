# POLSAR1 600x600, stride:100
dataconfig_mlc = {
    'Haywrd': {'traindata': ['/media/sean/Storage/data/POLSAR/Haywrd_15302_18074', '/home/sean/data/POLSAR/Haywrd_15302_18074'],
               'testdata':  ['/media/sean/Storage/data/POLSAR/Haywrd_15302_19031', '/home/sean/data/POLSAR/Haywrd_15302_19031'],
               'gtmat': '/media/sean/Storage/data/POLSAR/Haywrd_gtmat.npy'
               },
    'ykdelB': {'traindata': ['/media/sean/Storage/data/POLSAR/ykdelB_26906_18052','/home/sean/data/POLSAR/ykdelB_26906_18052'],
               'testdata':  ['/media/sean/Storage/data/POLSAR/ykdelB_26906_19066','/home/sean/data/POLSAR/ykdelB_26906_19066'],
               'gtmat': '/media/sean/Storage/data/POLSAR/ykdelB_gtmat.npy'
               }
}

# POLSAR2 224x224, stride:50
# dataconfig = {
#     'Haywrd': {'traindata': ['/home/sean/data/POLSAR2/Haywrd_15302_18074', '/home/sean/data/POLSAR2/Haywrd_15302_18074'],
#                'testdata':  ['/home/sean/data/POLSAR2/Haywrd_15302_19031', '/home/sean/data/POLSAR2/Haywrd_15302_19031'],
#                'gtmat': '/media/sean/Storage/data/POLSAR2/Haywrd_gtmat.npy'
#                },
#     'ykdelB': {'traindata': ['/home/sean/data/POLSAR2/ykdelB_26906_18052','/home/sean/data/POLSAR2/ykdelB_26906_18052'],
#                'testdata':  ['/home/sean/data/POLSAR2/ykdelB_26906_19066','/home/sean/data/POLSAR2/ykdelB_26906_19066'],
#                'gtmat': '/media/sean/Storage/data/POLSAR2/ykdelB_gtmat.npy'
#                }
# }

# POLSAR_GRD: grd file (not mlc)
dataconfig_grd = {
    'Haywrd': {'traindata': ['/home/sean/data/POLSAR_GRD/Haywrd_15302_18074', '/home/sean/data/POLSAR_GRD/Haywrd_15302_18074'],
               'testdata':  ['/home/sean/data/POLSAR_GRD/Haywrd_15302_19031', '/home/sean/data/POLSAR_GRD/Haywrd_15302_19031'],
               'gtmat': '/media/sean/Storage/data/POLSAR_GRD/Haywrd_gtmat.npz'
               },
    'ykdelB': {'traindata': ['/home/sean/data/POLSAR_GRD/ykdelB_26906_18052','/home/sean/data/POLSAR_GRD/ykdelB_26906_18052'],
               'testdata':  ['/home/sean/data/POLSAR_GRD/ykdelB_26906_19066','/home/sean/data/POLSAR_GRD/ykdelB_26906_19066'],
               'gtmat': '/media/sean/Storage/data/POLSAR_GRD/ykdelB_gtmat.npz'
               }
}

dataconfig = {"mlc": dataconfig_mlc, "grd": dataconfig_grd}