import torch
import os
import glob
import shutil
import json
import argparse
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from pprint import pprint

class Logger():
    def __init__(self, result_dir, filename, args):
        self.result_dir = None
        if result_dir is not None:
            self.result_dir = result_dir._str
            self.logger_fn = self.result_dir+'/log.txt'
            self.logger = open(self.logger_fn,'w')
            self.initial_write(filename, args)

    def initial_write(self, filename, args):
        self.logger.write("%s \n"%(filename))
        self.logger.write("Export directory: %s\n"%self.result_dir)
        self.logger.write("Arguments:\n")
        self.logger.write(json.dumps(vars(args)))
        self.logger.write("\n")

    def write(self, string, end='\n'):
        print(string, end=end)
        if self.result_dir is not None:
            self.logger.write("%s\n"%(string))
    
    def write_only(self, string):
        if self.result_dir is not None:
            self.logger.write("%s\n"%(string))
        

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    return device 


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def config_backup_get_log(args, sub_dir, filename):
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    # set result dir
    current_time = str(datetime.now())
    dir_name = '%s/%s_%s'%(sub_dir, current_time, args.suffix)
    result_dir = Path(os.path.join("./results", dir_name)) if not args.suffix in ['test'] else None
    
    # if result_dir is not None:
    #     if not os.path.isdir(result_dir):
    #         os.mkdir(result_dir)
    #         os.mkdir(result_dir+'/codes')
    if result_dir is not None:
        result_dir.mkdir(exist_ok=True, parents=True)
        Path(os.path.join(result_dir, "codes")).mkdir()

        # deploy codes
        files = glob.iglob('*.py')
        for file in files:
            shutil.copy2(file, os.path.join(result_dir,"codes"))
        # model_files = glob.iglob('./model/*.py')
        # for model_file in model_files:
        #     shutil.copy2(model_file, result_dir+'/codes/model')

    # printout information
    print("Export directory:", result_dir)
    print("Arguments:")
    pprint(vars(args))

    logger = Logger(result_dir, filename, args)

    return logger, result_dir, dir_name

def is_correct(query_georef, georef): # should be changed!
    ul_query = query_georef['upperleft']
    ur_query = query_georef['upperright']
    ll_query = query_georef['lowerleft']
    lr_query = query_georef['lowerright']
    queries = [query_georef['upperleft'], query_georef['upperright'], query_georef['lowerleft'], query_georef['lowerright']]
    # center_query = 0.25*(ul_query+ur_query+ll_query+lr_query)
    
    ul_ref = georef['upperleft']
    lr_ref = georef['upperright']
    ll_ref = georef['lowerleft']
    lr_ref = georef['lowerright']
    latitudes = np.asarray([ul_ref[0], lr_ref[0], ll_ref[0], lr_ref[0]])
    latitude_min = latitudes.min()
    latitude_max = latitudes.max()

    longitudes = np.asarray([ul_ref[1], lr_ref[1], ll_ref[1], lr_ref[1]])
    longitude_min = longitudes.min()
    longitude_max = longitudes.max()

    for query in queries:
        if query[0]>=latitude_min and query[0]<=latitude_max and query[1]>=longitude_min and query[1]<=longitude_max:
            return True
    return False