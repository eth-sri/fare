from ctypes import c_uint
import sys 
import argparse
import numpy as np
import numpy.matlib
import os 
from box import Box
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import time

from .alphabeta_adversary import AlphaBetaAdversary

from src.common.metrics import demographic_parity_difference, equalized_odds_difference
from src.common.dataset import get_dataset
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree 

def learn(data, cat_pos, max_leaf_nodes, min_samples_leaf, alpha, gini_metric, max_pc=None, pc=None):

    x_train, s_train, y_train = data['train']
    x_val, _, _ = data['val'] # unused now
    x_test, s_test, y_test = data['test']
    s_train, s_test = s_train.reshape(-1, 1), s_test.reshape(-1, 1)
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print('getting 0th label!')
        y_train, y_test = y_train[:, 0],  y_test[:, 0]
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    # create and fit the tree
    criterion = f'fair_gini_{gini_metric}' # e.g., fair_gini_dp
    T = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_leaf_nodes, random_state=43, min_samples_leaf=min_samples_leaf)
    if pc is None:
        T = T.fit(x_train, y_train, s_train, cat_pos=cat_pos, alpha=alpha)
    else:
        a = np.sum( s_train == 1 )
        idx_a = np.where( s_train == 1 )[0]
        b = np.sum( s_train == 0 )
        idx_b = np.where( s_train == 0 )[0]
        c = np.maximum( b / max_pc, a / max_pc )
        if a < b:
            idx_a_r = np.random.choice( idx_a, int(c * (1-pc)), replace=False)
            idx_b_r = np.random.choice( idx_b, int(c * pc), replace=False)
            idx = np.sort(np.concatenate((idx_a_r,idx_b_r)))
        else:
            idx_b_r = np.random.choice( idx_b, int(c * (1-pc)), replace=False)
            idx_a_r = np.random.choice( idx_a, int(c * pc), replace=False)
            idx = np.sort(np.concatenate((idx_a_r,idx_b_r)))
        x_train_temp = x_train[idx]
        s_train_temp = s_train[idx]
        y_train_temp = y_train[idx]
        print( x_train_temp.shape[0], np.sum(s_train_temp == 0)/x_train_temp.shape[0]  )
        T = T.fit(x_train_temp, y_train_temp, s_train_temp, cat_pos=cat_pos, alpha=alpha)

    # plot tree
    if not os.path.exists('src/tree/out/'):
        os.makedirs('src/tree/out/')
    plot_tree(T, node_ids=True)
    plt.savefig(f'src/tree/out/tree_{alpha}.pdf')
    plt.clf()

    print('tree built and saved')
    return T

def eval(T, data, cat_pos):
    x_train, s_train, y_train = data['train']
    x_val, _, _ = data['val'] # unused now
    x_test, s_test, y_test = data['test']
    s_train, s_test = s_train.reshape(-1, 1), s_test.reshape(-1, 1)
    if  len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print('getting 0th label!')
        y_train, y_test = y_train[:, 0],  y_test[:, 0]
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    print('############ Eval #################')
    
    base_train = y_train.sum() / y_train.shape[0]
    base_train = max(1-base_train, base_train) * 100
    base_test = y_test.sum() / y_test.shape[0]
    base_test = max(1-base_test, base_test) * 100
    print(f'Base Rates: train={base_train:.3f} test={base_test:.3f}')

    # score acc and DP
    acc_train = T.score(x_train, y_train)*100
    acc_test = T.score(x_test, y_test)*100
    proba_train = T.predict_proba(x_train)
    proba_test = T.predict_proba(x_test)

    dp_train, _ = demographic_parity_difference(y_train, s_train, proba_train)
    dp_test, _ = demographic_parity_difference(y_test, s_test, proba_test)
    print(f"\033[0;32m (DP) TRAIN = ({acc_train:.3f}, {dp_train:.3f}) [&&] TEST = ({acc_test:.3f}, {dp_test:.3f})", flush=True)
    print("\033[0m", flush=True)

    #############################################

    # get leaf/cell IDs reached from train set
    nb_cells = (T.tree_.children_left == -1).sum()
    cells_train = T.apply(x_train)
    cell_ids = sorted(list(set(cells_train)))
    
    # ensure all cells are present
    assert len(cell_ids) == nb_cells
    #assert len(sorted(list(set(T.apply(x_val))))) == nb_cells
    cells_test = T.apply(x_test)
    assert len(sorted(list(set(cells_test)))) == nb_cells

    #####################################################3

    # get medians for each cell
    medians = {}
    for cid in cell_ids:
        # get all train set xs that go to this cell
        xs = x_train[np.where(cells_train == cid)]
        
        # get median
        median = np.zeros(xs.shape[1])
        for i in range(xs.shape[1]): 
            if i in cat_pos:
                # categorical takes mode
                median[i] = mode(xs[:, i].astype(int))[0] # check
            else:
                # continuous takes median
                median[i] = np.median(xs[:, i]) # needs numpy 1.9.0
                #median[i] = np.mean(xs[:, i])
        medians[cid] = median 

    # encode a set of xs with the tree
    def encode(xs, T):
        cells = T.apply(xs)
        zs = []
        for cell in cells:
            zs.append(medians[cell])
        return np.vstack(zs)
    
    # Return embeddings
    z_train = encode(x_train, T)
    z_val = encode(x_val, T)
    z_test = encode(x_test, T) 
    return nb_cells, z_train, z_val, z_test 

def prep_data(data, meta):
    
    use_also_s = False 
    if use_also_s:
        data['train'] = (
            np.hstack([data['train'][0], data['train'][1].reshape(-1, 1)]),
            data['train'][1],
            data['train'][2]
        )
        
        data['test'] = (
            np.hstack([data['test'][0], data['test'][1].reshape(-1, 1)]),
            data['test'][1],
            data['test'][2]
        )

        meta['input_shape'] = data['train'][0].shape[1]
        meta['ft_pos']['Sensitive'] = 7

    # Revert 1-hot encoding -> tree gets cats as cats 
    x_train, x_test = [], []
    cat_pos = []
    for new_idx, idx in enumerate(meta['ft_pos'].values()):
        if type(idx) == tuple:
            # cat 
            slc = data['train'][0][:, idx[0]:idx[1]]
            assert slc.max(axis=1).min().item() == 1
            x_train.append(slc.argmax(axis=1)+1)

            slc = data['test'][0][:, idx[0]:idx[1]]
            assert slc.max(axis=1).min().item() == 1
            x_test.append(slc.argmax(axis=1)+1)
            cat_pos.append(new_idx)
        else:
            # cont 
            x_train.append(data['train'][0][:, idx])
            x_test.append(data['test'][0][:, idx])
    data['train'] = [np.vstack(x_train).T, data['train'][1], data['train'][2]]
    data['test'] = [np.vstack(x_test).T, data['test'][1], data['test'][2]]
    cat_pos = np.asarray(cat_pos, dtype=np.int32)

    #####################################################################
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
    # so it's: 
    # {z/c/y}_{train/test}

    # TODO make nicer
    if 'all_labels' in meta:
        embeddings = {
            'c_train': data['train'][1].reshape(-1,1),
            'c_val': data['val'][1].reshape(-1,1),
            'c_test': data['test'][1].reshape(-1,1),

            'y_train': data['train'][2],
            'y_val': data['val'][2],
            'y_test': data['test'][2]
        }
    else:
        embeddings = {
            'c_train': data['train'][1].reshape(-1,1),
            'c_val': data['val'][1].reshape(-1,1),
            'c_test': data['test'][1].reshape(-1,1),

            'y_train': data['train'][2].reshape(-1,1),
            'y_val': data['val'][2].reshape(-1,1),
            'y_test': data['test'][2].reshape(-1,1)
        }
    
    return data, cat_pos, embeddings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-k', type=int, default=None)
    parser.add_argument('--min-ni', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--val-split', type=float, default=0.5)

    parser.add_argument('--gini-metric', type=str, default='dp', choices=['dp', 'eopp', 'eo'])
    parser.add_argument('--eval-metric', type=str, default='all', choices=['dp', 'eopp', 'eo', 'all'])

    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--dataset-embed', type=str, default=None, help='If not None, use this dataset for embeddings')
    parser.add_argument('--resultdir', type=str, default='src/tree/out') # result/
    parser.add_argument('--max_pc', type=float, default=0.9) # Used for sens attribute imbalance
    parser.add_argument('--pc', type=float, default=None) # Used for sens attribute imbalance
    parser.add_argument('--imputed', type=float, default=0.0) # Used for imputation experiment
    args = parser.parse_args()
    return args

def get_imputed(data, cat_pos, percent):
    cat_val = mode(data['train'][0][:,cat_pos], axis=0)[0]
    not_cat_pos = set(range(data['train'][0].shape[1])) - set(cat_pos)
    not_cat_pos = list(not_cat_pos)
    cont_val = np.mean(data['train'][0][:,not_cat_pos], axis=0)

    def remove_vals( orig_data, percent ):    
        data = orig_data.reshape(-1)
        impute_chance = np.random.uniform(0,1,data.shape) < percent
        data[ impute_chance ] = None
        data = data.reshape( orig_data.shape )
        return data

    def impute_vals( data, cat_val, cont_val, cat_pos ):
        n_cols = data.shape[1]
        cat_idx = 0
        cont_idx = 0
        for i in range(n_cols):
            idx = np.isnan(data[:,i])
            if i in cat_pos:
                data[idx,i] = cat_val[0,cat_idx]
                cat_idx += 1
            else:
                data[idx,i] = cont_val[cont_idx]
                cont_idx += 1
        return data

    train = remove_vals( data['train'][0], percent )
    test = remove_vals( data['test'][0], percent )
    val = remove_vals( data['val'][0], percent )
    
    train = impute_vals( train, cat_val, cont_val, cat_pos )
    test = impute_vals( test, cat_val, cont_val, cat_pos )
    val = impute_vals( val, cat_val, cont_val, cat_pos )

    data['train'][0] = train
    data['test'][0] = test
    data['val'][0] = val

    print('\n\n\n\nImputation -- set seed to 0\n\n\n\n') 
    # 0.5^6 probability event with this seed (all examples in valset)
    return data
 

if __name__ == "__main__":
    args = get_args()
    np.random.seed(42) 

    ### Set up, load data
    if args.resultdir is not None:
        if args.gini_metric == 'dp':
            methodname = 'tree'
        else:
            methodname = f'tree-{args.gini_metric}'
        rundir = f'{args.resultdir}/{args.dataset}/{methodname}/k={args.max_k},ni={args.min_ni},a={args.alpha},s={args.val_split}'
        if args.pc is not None:
            rundir += ',mp={args.max_pc},pc={args.pc}'
        if args.imputed > 0:
            rundir += ',i={args.imputed}'
        rundir += '/'
            
        if not os.path.exists(rundir):
            os.makedirs(rundir)
    
    # Get data
    data, meta = get_dataset(Box({'name': args.dataset, 'val_size': 0}))
    data, cat_pos, embeddings = prep_data(data, meta)

    # Get evaluation data
    if args.dataset_embed is not None:
        data_embed, meta_embed = get_dataset(Box({'name': args.dataset_embed, 'val_size': 0}))
        data_embed, cat_pos_embed, embeddings = prep_data(data_embed, meta_embed)
        if args.imputed > 0.0:
            data_embed = get_imputed(data_embed, cat_pos_embed, args.imputed)
    else:
        data_embed, cat_pos_embed, embeddings = data, cat_pos, embeddings # use the same dataset

    ### Run the algorithm, it will internally use validation 
    T = learn(data, cat_pos, args.max_k, args.min_ni, args.alpha, args.gini_metric, max_pc=args.max_pc, pc=args.pc)
    k, z_train, z_val, z_test = eval(T, data_embed, cat_pos_embed)

    embeddings['z_train'] = z_train 
    embeddings['z_val'] = z_val 
    embeddings['z_test'] = z_test 

    #############################################################################

    # revert (train, val) split for final embeddings
    final_embeddings = {
        'c_train': np.vstack([embeddings['c_train'], embeddings['c_val']]),
        'c_test': embeddings['c_test'],

        'y_train': np.vstack([embeddings['y_train'], embeddings['y_val']]),
        'y_test': embeddings['y_test'],

        'z_train': np.vstack([embeddings['z_train'], embeddings['z_val']]),
        'z_test': embeddings['z_test'],
    }
    for v in final_embeddings.values():
        assert type(v) == np.ndarray

    # Proof on the embeddings object
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

    def eval_adversary(adv, emb, name, metric='dp'):
        # hack to make transfer work, TODO refactor 
        embeddings = {}
        for k, v in emb.items():
            embeddings[k] = v 
            if 'y_' in k and  len(v.shape) > 1 and v.shape[1] > 1:
                embeddings[k] = v[:,0]

        print(f'Evaluating adversary with metric: {metric}')
        if metric == 'dp':
            return _print_and_parse_adv(adv.ub_demographic_parity(embeddings), name) 
        elif metric == 'eopp':
            return _print_and_parse_adv(adv.ub_equal_opportunity(embeddings), name) 
        elif metric == 'eo':
            return _print_and_parse_adv(adv.ub_equalized_odds(embeddings), name) 
        else:
            raise RuntimeError(f'Unknown metric: {metric}')

    if meta['c_type'] == 'binary':
        # Usual flow for binary sensitive attributes
        adv = AlphaBetaAdversary(k, err_budget, eps_glob=0.005, eps_ab=0.005, method='cp', verbose=True)
        
        eval_metrics = [args.eval_metric]
        if args.eval_metric == 'all':
            eval_metrics = ['dp', 'eopp', 'eo']
        
        for eval_metric in eval_metrics:
            final_embeddings[f'{eval_metric}_ub'] = eval_adversary(adv, embeddings, 'AlphaBeta', metric=eval_metric)

        # Save
        if args.resultdir is not None:
            np.save(f'{rundir}/embeddings.npy', final_embeddings, allow_pickle=True)
        print('TREE DONE.')
    else:
        # Multi: for more S we need to do all pairs
        unique_s = np.unique(np.vstack([embeddings['c_train'], embeddings['c_val'], embeddings['c_test']]))
        nb_s = unique_s.shape[0]

        err_budget /= (nb_s * (nb_s-1) / 2)
        total_dpub = 0

        for i in range(nb_s):
            for j in range(i+1, nb_s):
                curr_embeddings = {}
                for split in ['train', 'val', 'test']:
                    maski = (embeddings[f'c_{split}'] == i).ravel()
                    maskj = (embeddings[f'c_{split}'] == j).ravel()
                    
                    curr = embeddings[f'c_{split}'][maski | maskj]
                    curr[curr == i] = -1
                    curr[curr == j] = 1 
                    curr[curr == -1] = 0 
                    curr_embeddings[f'c_{split}'] = curr

                    curr_embeddings[f'z_{split}'] = embeddings[f'z_{split}'][maski | maskj]
                adv = AlphaBetaAdversary(k, err_budget, eps_glob=0.005, eps_ab=0.005, method='cp', verbose=False)
                dpub = eval_adversary(adv, curr_embeddings, 'AlphaBeta', metric='dp')
                total_dpub = max(total_dpub, dpub)

        print(f'There are {nb_s} sensitive values')
        print(f'Total DP upper bound is: {total_dpub}')