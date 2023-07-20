# From https://github.com/cjshui/fair-path

from ctypes import c_uint
import sys 
import argparse
import numpy as np
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from box import Box
from src.common.metrics import *
from src.common.dataset import get_dataset
from .model import Fea, Clf
from .utils import  train_implicit, evaluate_pp_implicit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--kappa', type=float, default=None)
    parser.add_argument('--resultdir', type=str, default=None) # result/
    parser.add_argument('--n_epoch', type=int, default=1000)
    args = parser.parse_args()
    return args

args = get_args()

### Set up, load data
rundir = f'{args.resultdir}/{args.dataset}/fair-path/kappa={args.kappa}'
if not os.path.exists(rundir):
    os.makedirs(rundir)

data, meta = get_dataset(Box({'name': args.dataset, 'val_size': 0}))

X_train = data['train'][0]
X_test = data['test'][0]
A_train = data['train'][1].reshape(-1,1)
A_test = data['test'][1].reshape(-1,1)
y_train = data['train'][2].reshape(-1,1)
y_test = data['test'][2].reshape(-1,1)

A_train, A_test = A_train.reshape(-1), A_test.reshape(-1)
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

for iter in range(1):
    # get train/test data
    # X_train, X_test, A_train, A_test, y_train, y_test = load_toxic_bert()

    ### convert Y to binary with threshold=0
    # y_train = np.array(y_train > 0, dtype=np.int)
    # y_test = np.array(y_test > 0, dtype=np.int)

    # initialize model
    fea = Fea(input_size=len(X_train[0])).cuda()
    clf_0 = Clf().cuda()
    clf_1 = Clf().cuda()


    optim_fea = optim.Adam(fea.parameters(), lr=1e-3, eps=1e-3)
    optim_clf_0 = optim.Adam(clf_0.parameters(), lr=1e-3, eps=1e-3)
    optim_clf_1 = optim.Adam(clf_1.parameters(), lr=1e-3, eps=1e-3)

    criterion = nn.BCELoss()
        
    train_implicit(fea, clf_0, clf_1, criterion, optim_fea, optim_clf_0, optim_clf_1, X_train, A_train, y_train, kappa=args.kappa,n_epoch=args.n_epoch,max_inner=15,out_step=5)

    with torch.no_grad():
        z_train = fea(torch.tensor(X_train).cuda().float())
        z_test = fea(torch.tensor(X_test).cuda().float())
        c_train = A_train.reshape(-1, 1)
        c_test = A_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
    embeddings = {
        'z_train': z_train.cpu().numpy(),
        'z_test': z_test.cpu().numpy(),
        'c_train': c_train,
        'c_test': c_test,
        'y_train': y_train,
        'y_test': y_test,
    }
    
    ap_test, gap_test = evaluate_pp_implicit(fea, clf_0, clf_1, X_test, y_test, A_test)

    print("The accuracy is:", ap_test)
    print("The prediction gap is:", gap_test)

    np.save(f'{rundir}/embeddings.npy', embeddings, allow_pickle=True)
