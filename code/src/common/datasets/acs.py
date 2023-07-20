import importlib

import numpy as np
from folktables import (ACSDataSource, ACSEmployment, ACSIncome, ACSMobility,
                        ACSPublicCoverage, ACSTravelTime, BasicProblem, adult_filter)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def threshold_for_multi(p):
    ret = p.copy() 
    ret[:] = 0
    ret[p > 20000] = 1 
    ret[p > 50000] = 2 
    ret[p > 100000] = 3
    return ret

ACSIncomeMulti = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform = lambda x: threshold_for_multi(x),
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# Holds:
#
# feature_names, cont_feats, disc_feats, tot_feats, sens_feats
#
# feat_data list of big tuples
#
# original data: X/y_train_orig, X/y_val_orig, X/y_test_orig
# one-hot data: same but _oh
# loaders (with one-hot) for train, val, test
def load_acs(dataname):
    toks = dataname.split('-')
    assert len(toks) == 3 
    dataset, state, year = toks[0], toks[1], int(toks[2]) 

    all_cont_feats = {
        'ACSIncome': [0, 2, 7],
        'ACSIncomeMulti': [0, 2, 7],

        'ACSEmployment': [0, 1],
        'ACSPublicCoverage': [0, 1, 14],
        'ACSMobility': [0, 1, 18, 19, 20],
        'ACSTravelTime': [0, 1, 15]
    }

    # Get data source
    data_source = ACSDataSource(survey_year=str(year), horizon='1-Year', survey='person')
    if state == 'ALL':
        states = None
    else:
        states = [state]
    acs_data = data_source.get_data(states=states, download=True)
    m = importlib.import_module('folktables')

    if dataset == 'ACSIncomeMulti':
        ACSClass = ACSIncomeMulti
    else:
        ACSClass = getattr(m, dataset)

    # Get all features and labels and distinguish types
    features, labels, _ = ACSClass.df_to_numpy(acs_data)
    feature_names = ACSClass.features
    cont_feats = all_cont_feats[dataset] 
    disc_feats = [i for i in range(len(feature_names)) if i not in cont_feats]

    if dataset == 'ACSIncomeMulti':
        # We use 'RAC1P' as sensitive
        sens_idx = ACSClass.features.index('RAC1P')
        mask1 = (features[:, sens_idx]==1)
        mask6 = (features[:, sens_idx]==6)

        features[(~mask1) & (~mask6), sens_idx] = 0
        features[mask6] = 2 
        # 0 = other, 1 = white (used to be 1), 2 = asian (used to be 6)
    else:
        # We use 'SEX' as sensitive (1/2), just set 2 to 0 
        sens_idx = ACSClass.features.index('SEX')
        mask2 = (features[:, sens_idx]==2)
        features[mask2, sens_idx] = 0 

    # Split and then pull out sens
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    s_train = X_train[:, sens_idx]
    X_train = np.delete(X_train, sens_idx, axis=1)
    s_test = X_test[:, sens_idx]
    X_test = np.delete(X_test, sens_idx, axis=1)

    # Adjust cont/disc
    cont_feats = [(c-1 if c>sens_idx else c) for c in cont_feats]
    disc_feats = [(c-1 if c>sens_idx else c) for c in disc_feats if c != sens_idx]
    feature_names = [f for f in feature_names if f != 'SEX']
    #tot_feats = len(disc_feats) + len(cont_feats)

    # Permute such that continuous go first (not necessary but easier to handle in Tree)
    perm = cont_feats + disc_feats 
    invperm = {}
    for i, idx in enumerate(perm):
        invperm[idx] = i
    X_train = X_train[:, perm]
    X_test = X_test[:, perm]
    feature_names = [feature_names[i] for i in perm]
    cont_feats = [invperm[x] for x in cont_feats]
    disc_feats = [invperm[x] for x in disc_feats]

    # NOTE: oh sets all to zero, check
    # NOTE: Preserve order when building _oh
    # Scale and transform
    scaler = MinMaxScaler()
    scaler.fit(X_train[:, cont_feats])
    X_train[:, cont_feats] = scaler.transform(X_train[:, cont_feats])
    X_test[:, cont_feats] = scaler.transform(X_test[:, cont_feats])
    
    # Categories are made from train set so train set has /all/ categories present at least once 
    # (Last col might group all infrequent and unknown (<5) cats)
    # (We assert that if there is an unknown cat in test set there is an infreq column to put it in, 
    # so no example has [0,0,0,0] -> important for the tree)
    # Test set might not have some categories represented, but that's irrelevant
    oh_enc = OneHotEncoder(sparse=False, handle_unknown='infrequent_if_exist', min_frequency=5) # if under 5 examples
    oh_enc.fit(X_train[:, disc_feats])
    X_train_oh = np.concatenate([X_train[:, cont_feats], oh_enc.transform(X_train[:, disc_feats])], axis=1)
    X_test_oh = np.concatenate([X_test[:, cont_feats], oh_enc.transform(X_test[:, disc_feats])], axis=1)
    categories = oh_enc.categories_

    # make ft pos
    ft_pos = {} 
    beg = 0
    enc_idx = 0
    for i in range(len(cont_feats) + len(disc_feats)):
        name = feature_names[i]
        if i in disc_feats:
            #tot_vals = len(np.unique(X_train[:, i])) # we want categories
            # take into account infrequent categories
            tot_vals = oh_enc.categories_[enc_idx].shape[0]
            inf = oh_enc.infrequent_categories_[enc_idx]
            if inf is not None:
                tot_vals -= inf.shape[0]
                tot_vals += 1

            ft_pos[name] = (beg, beg+tot_vals)
            beg += tot_vals 
            enc_idx += 1
        elif i in cont_feats:
            ft_pos[name] = beg
            beg += 1
        else:
            assert False
    
    # checks
    assert enc_idx == len(oh_enc.categories_)
    assert X_train_oh.shape[1] == beg
    
    cnt_maptozero = 0 
    cnt_emptycat = 0
    for it, X in enumerate([X_train_oh, X_test_oh]):
        for v in ft_pos.values():
            if type(v) != tuple:
                continue
            slic = X[:, v[0]:v[1]]
            maptozero = np.where(slic.sum(1)==0)[0].shape[0]
            emptycat = np.where(slic.sum(0)==0)[0].shape[0]
            cnt_maptozero += maptozero
            cnt_emptycat += emptycat
        if it == 0:
            assert cnt_emptycat == 0
    assert cnt_maptozero == 0 

    # return 
    data = {
        'train': (X_train_oh, s_train.astype(np.int), y_train.astype(np.int)),
        'test': (X_test_oh, s_test.astype(np.int), y_test.astype(np.int)),
        'ft_pos': ft_pos # important for tree 
    }
    return data


        
