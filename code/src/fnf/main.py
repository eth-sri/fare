# https://github.com/eth-sri/fnf
from ctypes import c_uint
import sys 
import argparse
import numpy as np
import os 
import torch
import random

from box import Box
from src.common.metrics import *
from src.common.dataset import get_dataset
from .train_gen_categorical import train_gen_cat
from .train_enc_categorical import train_enc_cat


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--resultdir', type=str, default=None) # result/
    parser.add_argument('--n_epoch', type=int, default=60)
    parser.add_argument('--n_bins', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')

    # enc
    parser.add_argument('--encode', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save-encoding', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--with_test', action='store_true')

    args = parser.parse_args()
    return args

args = get_args()
device = args.device

### Set up, load data
rundir = f'{args.resultdir}/{args.dataset}/fnf/gamma={args.gamma}'
if not os.path.exists(rundir):
    os.makedirs(rundir)

data, meta = get_dataset(Box({'name': args.dataset, 'val_size': 0}))


X_train = data['train'][0]
X_test = data['test'][0]
A_train = data['train'][1].reshape(-1)
A_test = data['test'][1].reshape(-1)
y_train = data['train'][2].reshape(-1)
y_test = data['test'][2].reshape(-1)

ds = 1

new_cols_train, new_cols_valid, new_cols_test = [], [], []
new_ft_pos = []

curr_col = 0
print(meta['ft_pos'])

ft_pos = list(meta['ft_pos'].items())
ft_pos.sort(key=lambda x: x[1][0] if isinstance(x[1], tuple) else x[1])

for ft_name, cols in ft_pos:
    if ft_name == 'POBP':
        continue
    if isinstance(cols, tuple) and cols[1] - cols[0] + 1 > 100:
        N = 50
        c1, c2 = cols
        k = (c2-c1)//N + 1
        
        train_ids = np.floor_divide(np.argmax(X_train[:,c1:c2], axis=1), N)
        test_ids = np.floor_divide(np.argmax(X_test[:,c1:c2], axis=1), N)

        tmp_cols_train = np.zeros((X_train.shape[0], int(k)))
        tmp_cols_train[np.arange(X_train.shape[0]), train_ids] = 1.0
        new_cols_train += [tmp_cols_train]

        tmp_cols_test = np.zeros((X_test.shape[0], int(k)))
        tmp_cols_test[np.arange(X_test.shape[0]), test_ids] = 1.0
        new_cols_test += [tmp_cols_test]

        # avg = X_train[:,c1:c2].mean(axis=0)
        # k = 10
        # topk = np.argsort(avg)[-(k-1):]

        # for j in range(k-1):
        #     new_cols_train += [np.expand_dims(X_train[:,c1:c2][:,topk[j]], axis=1)]
        # new_cols_train += [1-np.sum(X_train[:,c1+topk], axis=1, keepdims=True)]

        # for j in range(k-1):
        #     new_cols_test += [np.expand_dims(X_test[:,c1:c2][:,topk[j]], axis=1)]
        # new_cols_test += [1-np.sum(X_test[:,c1+topk], axis=1, keepdims=True)]

        new_ft_pos += [(ft_name, (curr_col, curr_col+k))]
        curr_col += k
    else:
        if isinstance(cols, tuple):
            new_cols_train += [X_train[:, cols[0]:cols[1]]]
            new_cols_test += [X_test[:, cols[0]:cols[1]]]
            new_ft_pos += [(ft_name, (curr_col, curr_col+cols[1]-cols[0]))]
            curr_col += cols[1]-cols[0]
        else:
            new_cols_train += [X_train[:, cols:cols+1]]
            new_cols_test += [X_test[:, cols:cols+1]]
            new_ft_pos += [(ft_name, curr_col)]
            curr_col += 1

X_train = np.concatenate(new_cols_train, axis=1)
X_test = np.concatenate(new_cols_test, axis=1)

# for ft_name, cols in new_ft_pos.items():
#     if isinstance(cols, tuple):
#         c1, c2 = cols
#         print(ft_name, cols, np.sum(X_train[:, c1:c2], axis=1))
#     else:
#         print(ft_name, cols, X_train[:, cols])
# exit(0)

p_valid = 0.2
ids = np.arange(X_train.shape[0])
np.random.shuffle(ids)
valid_ids = ids[:int(p_valid*X_train.shape[0])]
train_ids = ids[int(p_valid*X_train.shape[0]):]

X_train, X_valid = X_train[train_ids], X_train[valid_ids]
A_train, A_valid = A_train[train_ids], A_train[valid_ids]
y_train, y_valid = y_train[train_ids], y_train[valid_ids]

cols_train, cols_valid, cols_test = [], [], []

print(new_ft_pos)

dims = []

for ft_name, cols in new_ft_pos:
    if isinstance(cols, tuple):
        c1, c2 = cols
        cols_train += [X_train[:, c1:c2]]
        cols_valid += [X_valid[:, c1:c2]]
        cols_test += [X_test[:, c1:c2]]
        dims += [c2-c1]
    else:
        i = cols
        q = np.linspace(0, 1, args.n_bins+1)
        bins = np.quantile(X_train[:, i], q)
        bins[0], bins[-1] = bins[0] - 1e-2, bins[-1] + 1e-2

        def encode_column(feats):
            disc = np.digitize(feats, bins)
            disc = np.clip(disc, 1, args.n_bins) # test data outside of training bins
            one_hot_vals = torch.zeros(feats.shape[0], args.n_bins)
            one_hot_vals[np.arange(feats.shape[0]), disc - 1] = 1.0
            return one_hot_vals

        cols_train += [encode_column(X_train[:, i])]
        cols_valid += [encode_column(X_valid[:, i])]
        cols_test += [encode_column(X_test[:, i])]
        dims += [args.n_bins]


X_train = np.concatenate(cols_train, axis=1)
X_valid = np.concatenate(cols_valid, axis=1)
X_test = np.concatenate(cols_test, axis=1)

X_train = torch.tensor(X_train).float().to(device)
X_valid = torch.tensor(X_valid).float().to(device)
X_test = torch.tensor(X_test).float().to(device)

if not os.path.exists(f'src/fnf/{args.dataset}'):
    os.makedirs(f'src/fnf/{args.dataset}')

# load cached src/fnf/%s/made{1/2}.pt
train_gen_cat(args, X_train, A_train, X_valid, A_valid)

embeddings = train_enc_cat(args, dims, new_ft_pos, X_train, A_train, y_train, X_valid, A_valid, y_valid, X_test, A_test, y_test)

np.save(f'{rundir}/embeddings.npy', embeddings, allow_pickle=True)
