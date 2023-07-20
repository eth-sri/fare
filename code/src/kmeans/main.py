import argparse
import numpy as np
import os 
from box import Box
from scipy.stats import mode
from sklearn.cluster import KMeans

from src.common.metrics import *
from src.common.dataset import get_dataset
import matplotlib.pyplot as plt
from src.tree.alphabeta_adversary import AlphaBetaAdversary

# TODO consolidate this and tree as "restricted encoder"

def learn(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(data['train'][0])
    clusters = kmeans.cluster_centers_ 
    z_train = clusters[kmeans.predict(data['train'][0])]
    z_val = clusters[kmeans.predict(data['val'][0])]
    z_test = clusters[kmeans.predict(data['test'][0])]
    return z_train, z_val, z_test # TODO this breaks with huge K 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--resultdir', type=str, default='result')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)

    ### Set up, load data
    if args.resultdir is not None:
        rundir = f'{args.resultdir}/{args.dataset}/kmeans/k={args.k}/'
        if not os.path.exists(rundir):
            os.makedirs(rundir)
    
    data, meta = get_dataset(Box({'name': args.dataset, 'val_size': 0}))
    data['train'] = list(data['train'])
    data['test'] = list(data['test'])

    # (!) We need internally a validation set 
    n_trainval = data['train'][0].shape[0] 
    n_val = int(args.val_split * n_trainval) # 60 : 20 : 20

    perm = np.random.permutation(n_trainval)

    val_idxs = perm[:n_val]
    train_idxs = perm[n_val:]

    data['val'] = []
    for i in range(3):
        data['val'].append(data['train'][i][val_idxs])
    for i in range(3):
        data['train'][i] = data['train'][i][train_idxs]

    for k in ['train', 'val', 'test']:
        s = ''
        for i in range(3):
            s += ' ' + f'{data[k][i].shape}'
        print(s)

    # Prepare embeddings to save later, we will not change c/y
    embeddings = {
        'c_train': data['train'][1].reshape(-1,1),
        'c_val': data['val'][1].reshape(-1,1),
        'c_test': data['test'][1].reshape(-1,1),

        'y_train': data['train'][2].reshape(-1,1),
        'y_val': data['val'][2].reshape(-1,1),
        'y_test': data['test'][2].reshape(-1,1)
    }

    ### Run algorithm, it will internally use validation 
    z_train, z_val, z_test = learn(data, args.k)
    embeddings['z_train'] = z_train 
    embeddings['z_val'] = z_val 
    embeddings['z_test'] = z_test 

    for v in embeddings.values():
        assert type(v) == np.ndarray

    # proof on the embeddings object
    err_budget = 0.05 # find best UB s.t. we are 95% confident

    def _print_and_parse_adv(ret, name):
        print("\033[0;32m", flush=True)
        ub = -1
        if len(ret)>1: 
            print(f'{name} Proof (train): {ret[0]:.4f} with error {err_budget}')
            print(f'{name} Proof:  {ret[1]:.4f} with error {err_budget}')
            ub = ret[1].item()
        else:
            print(f'{name} Proof:  {ret[0]:.4f} with error {err_budget}')
            ub = ret[0].item()
        print("\033[0m", flush=True)
        return ub 

    def eval_adversary(adv, embeddings, name, metric='dp'):
        print(f'Evaluating adversary with metric: {metric}')
        if metric == 'dp':
            return _print_and_parse_adv(adv.ub_demographic_parity(embeddings), name) 
        elif metric == 'eopp':
            return _print_and_parse_adv(adv.ub_equal_opportunity(embeddings), name) 
        elif metric == 'eo':
            return _print_and_parse_adv(adv.ub_equalized_odds(embeddings), name) 
        else:
            raise RuntimeError(f'Unknown metric: {metric}')

    verbose = False

    adv = AlphaBetaAdversary(args.k, err_budget, eps_glob=0.005, eps_ab=0.005, method='cp', verbose=verbose)
    dpub = eval_adversary(adv, embeddings, 'AlphaBeta', metric='dp')
    #eoppub = eval_adversary(adv, embeddings, 'AlphaBeta', metric='eopp')
    #eoub = eval_adversary(adv, embeddings, 'AlphaBeta', metric='eo')

    # Save
    if args.resultdir is not None:
        embeddings['dp_ub'] = dpub
        #embeddings['eopp_ub'] = eoppub
        #embeddings['eo_ub'] = eoub
        np.save(f'{rundir}/embeddings.npy', embeddings, allow_pickle=True)

    print('KMEANS DONE.')
