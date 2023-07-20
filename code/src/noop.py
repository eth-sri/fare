from ctypes import c_uint
import sys 
import argparse
import numpy as np
import os 
from box import Box

from src.common.metrics import *
from src.common.dataset import get_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    ### Set up, load data
    rundir = f'result/{args.dataset}/noop/0'
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    
    data, meta = get_dataset(Box({'name': args.dataset, 'val_size': 0}))
    
    embeddings = {
        'z_train': data['train'][0],
        'z_test': data['test'][0],
        'c_train': data['train'][1].reshape(-1,1),
        'c_test': data['test'][1].reshape(-1,1),
        'y_train': data['train'][2].reshape(-1,1),
        'y_test': data['test'][2].reshape(-1,1)
    }
    
    np.save(f'{rundir}/embeddings.npy', embeddings, allow_pickle=True)


# python3 -m src.noop --dataset ACSIncome-CA-2014
