"""
Preprocessing based on FNF which is based on https://github.com/truongkhanhduy95/Heritage-Health-Prize
"""
import zipfile
from os import path
from urllib import request

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.common.datasets.abstract_dataset import AbstractDataset

class HealthDataset(AbstractDataset):

    column_names = ['MemberID', 'ProviderID', 'Sex', 'AgeAtFirstClaim']
    claims_cat_names = ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc']

    def __init__(self, split, args, normalize=True, p_test=0.2, p_val=0.2, preprocess=True):
        super().__init__('health', split, p_test, p_val)

        self.data_dir = './data'
        health_file = path.join(self.data_dir, 'health_full.csv')

        if not (args.load and path.exists(health_file)):
            data_file = path.join(self.data_dir, 'HHP_release3.zip')
            
            if not path.exists(data_file):
                request.urlretrieve('https://foreverdata.org/1015/content/HHP_release3.zip', data_file)

            zf = zipfile.ZipFile(data_file)
            for fn1 in ['Claims.csv', 'DrugCount.csv', 'LabCount.csv', 'Members.csv']:
                fn2 = path.join(self.data_dir, fn1)
                if path.exists(fn2):
                    continue
                with zf.open(fn1) as f1:
                    with open(fn2, 'wb') as f2:
                        for line in f1:
                            f2.write(line)

            df_claims = self.preprocess_claims(pd.read_csv(open(path.join(self.data_dir,'Claims.csv'),'r'), sep=','))
            df_drugs = self.preprocess_drugs(pd.read_csv(open(path.join(self.data_dir,'DrugCount.csv'),'r'), sep=','))
            df_labs = self.preprocess_labs(pd.read_csv(open(path.join(self.data_dir,'LabCount.csv'),'r'), sep=','))
            df_members = self.preprocess_members(pd.read_csv(open(path.join(self.data_dir,'Members.csv'),'r'), sep=','))

            df_labs_drugs = pd.merge(df_labs, df_drugs, on=['MemberID', 'Year'], how='outer')
            df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims, on=['MemberID', 'Year'], how='outer')
            df_health = pd.merge(df_labs_drugs_claims, df_members, on=['MemberID'], how='outer')

            df_health.drop(['Year', 'MemberID'], axis=1, inplace=True)
            df_health.fillna(0, inplace=True)

            df_health.to_csv(health_file, index=False)

        df_health = pd.read_csv(open(path.join(health_file),'r'), sep=',')

        labels = df_health[args.label]
        if 'max_CharlsonIndex' in args.label:
            labels.loc[:,'max_CharlsonIndex'] =  1 - df_health['max_CharlsonIndex']

        if args.transfer:
            print('\n\n\n\n\n\nTRANSFER\n\n\n\n\n')
            drop_cols = [col for col in df_health.columns if col.startswith('PrimaryConditionGroup=')]
            df_health.drop(drop_cols, axis=1, inplace=True)

        features = df_health.drop('max_CharlsonIndex', axis=1)

        continuous_vars = [col for col in features.columns if '=' not in col]
        self.continuous_columns = [features.columns.get_loc(var) for var in continuous_vars]

        self.protected_unique = 2
        protected = np.logical_or(
            features['AgeAtFirstClaim=60-69'], np.logical_or(
                features['AgeAtFirstClaim=70-79'], features['AgeAtFirstClaim=80+']
            )
        )

        drop_cols = [col for col in df_health.columns if col.startswith('AgeAtFirstClaim=')]
        features.drop( drop_cols, axis=1, inplace=True) 

        self.one_hot_columns = {}
        for column_name in HealthDataset.column_names:
            ids = [i for i, col in enumerate(features.columns) if col.startswith('{}='.format(column_name))]
            if len(ids) > 0:
                assert len(ids) == ids[-1] - ids[0] + 1
            self.one_hot_columns[column_name] = ids
        print('categorical features: ', self.one_hot_columns.keys())

        self.column_ids = {col: idx for idx, col in enumerate(features.columns)}
        one_hot_keys = [k for k in self.one_hot_columns if len(self.one_hot_columns[k])>0]
        self.ft_pos = self.column_ids
        for one_k in one_hot_keys:
            p = self.one_hot_columns[one_k][0], self.one_hot_columns[one_k][-1]+1
            self.ft_pos = { k:self.ft_pos[k] for k in self.ft_pos if one_k not in k}
            self.ft_pos[one_k] = p

        if args.fnfpruned:
            cols = features.columns.tolist()
            if args.transfer: 
                cols_left = [cols[i] for i in [0, 1, 2, 3, 16, 31, 44]]
            else:
                cols_left = [cols[i] for i in [2, 3, 4, 5, 8, 19, 33, 61, 71, 77, 92]]

            new_ft_pos = {}
            for i,k in enumerate(cols_left):
                new_ft_pos[k] = i
            self.ft_pos = new_ft_pos

        features = torch.tensor(features.values.astype(np.float32))
        labels = torch.tensor(labels.values.astype(np.int64)).bool().long()
        if labels.shape[1] == 1:
            labels = labels[:,0]
        protected = torch.tensor(protected.values.astype(np.bool))
        if p_test > 1e-6:
            X_train, self.X_test, y_train, self.y_test, protected_train, self.protected_test = train_test_split(
                features, labels, protected, test_size=p_test, random_state=0
            )
        else:
            X_train, self.X_test, y_train, self.y_test, protected_train, self.protected_test = features, None, labels, None, protected, None

        if p_val > 1e-6:
            self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = train_test_split(
                X_train, y_train, protected_train, test_size=p_val, random_state=0
            )
        else:
            self.X_train, self.X_val, self.y_train, self.y_val, self.protected_train, self.protected_val = X_train, None, y_train, None, protected_train, None

        if normalize:
            # self.continuous_columns = np.arange(self.X_train.shape[1]-1)
            self._normalize(self.continuous_columns)
        
        if args.fnfpruned and args.transfer:
            self.X_train = self.X_train[:,[0, 1, 2, 3, 16, 31, 44]]
        if args.fnfpruned and not args.transfer:
            self.X_train = self.X_train[:,[2, 3, 4, 5, 8, 19, 33, 61, 71, 77, 92]]
        self.X_train = self.X_train.to(args.device)
        self.y_train = self.y_train.to(args.device)
        self.protected_train = self.protected_train.to(args.device)
        
        if p_val > 1e-6:
            if args.fnfpruned and args.transfer:
                self.X_val = self.X_val[:,[0, 1, 2, 3, 16, 31, 44]]
            if args.fnfpruned and not args.transfer:
                self.X_val = self.X_val[:,[2, 3, 4, 5, 8, 19, 33, 61, 71, 77, 92]]
            self.X_val = self.X_val.to(args.device)
            self.y_val = self.y_val.to(args.device)
            self.protected_val = self.protected_val.to(args.device)

        if p_test > 1e-6:
            if args.fnfpruned and args.transfer:
                self.X_test = self.X_test[:,[0, 1, 2, 3, 16, 31, 44]]
            if args.fnfpruned and not args.transfer:
                self.X_test = self.X_test[:,[2, 3, 4, 5, 8, 19, 33, 61, 71, 77, 92]]
            self.X_test = self.X_test.to(args.device)
            self.y_test = self.y_test.to(args.device)
            self.protected_test = self.protected_test.to(args.device)
        

        self._assign_split()
        #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",X_train.shape,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    @staticmethod
    def preprocess_claims(df_claims):
        df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
        df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

        df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
        df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
        df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
        df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
        df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
        df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
        df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
        df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
        df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
        df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
        df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
        df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12

        df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
        df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
        df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 2
        df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 3

        df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
        df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
        df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
        df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
        df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
        df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
        df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 11
        df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
        df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
        df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 180
        df_claims['LengthOfStay'].fillna(0, inplace=True)
        df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)

        for cat_name in HealthDataset.claims_cat_names:
            df_claims[cat_name].fillna(f'{cat_name}_?', inplace=True)
        df_claims = pd.get_dummies(df_claims, columns=HealthDataset.claims_cat_names, prefix_sep='=')

        oh = [col for col in df_claims if '=' in col]

        agg = {
            'ProviderID': ['count', 'nunique'],
            'Vendor': 'nunique',
            'PCP': 'nunique',
            'CharlsonIndex': 'max',
            # 'PlaceSvc': 'nunique',
            # 'Specialty': 'nunique',
            # 'PrimaryConditionGroup': 'nunique',
            # 'ProcedureGroup': 'nunique',
            'PayDelay': ['sum', 'max', 'min']
        }
        for col in oh:
            agg[col] = 'sum'

        df_group = df_claims.groupby(['Year', 'MemberID'])
        df_claims = df_group.agg(agg).reset_index()
        df_claims.columns = [
                                'Year', 'MemberID', 'no_Claims', 'no_Providers', 'no_Vendors', 'no_PCPs',
                                'max_CharlsonIndex', 'PayDelay_total', 'PayDelay_max', 'PayDelay_min'
                            ] + oh

        return df_claims

    @staticmethod
    def preprocess_drugs(df_drugs):
        df_drugs.drop(columns=['DSFS'], inplace=True)
        # df_drugs['DSFS'] = df_drugs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
        df_drugs = df_drugs.groupby(['Year', 'MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
        df_drugs.columns = ['Year', 'MemberID', 'DrugCount_total', 'DrugCount_months']
        print('df_drugs.shape = ', df_drugs.shape)
        return df_drugs

    @staticmethod
    def preprocess_labs(df_labs):
        df_labs.drop(columns=['DSFS'], inplace=True)
        # df_labs['DSFS'] = df_labs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
        df_labs = df_labs.groupby(['Year', 'MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
        df_labs.columns = ['Year', 'MemberID', 'LabCount_total', 'LabCount_months']
        print('df_labs.shape = ', df_labs.shape)
        return df_labs

    @staticmethod
    def preprocess_members(df_members):
        df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
        df_members['Sex'].fillna('?', inplace=True)
        df_members = pd.get_dummies(
            df_members, columns=['AgeAtFirstClaim', 'Sex'], prefix_sep='='
        )
        print('df_members.shape = ', df_members.shape)
        return df_members

def load_health(transfer=False, label=None,  fnfpruned=False, p_test=0.2):
    import argparse
    args = argparse.Namespace()
    args.load = True
    args.device = 'cpu'
    args.transfer = transfer
    args.label = label
    args.fnfpruned = fnfpruned

    train_dataset = HealthDataset('train', args, p_test=p_test, p_val=0)
    c_train = train_dataset.protected_train.numpy()
    X_train = train_dataset.X_train.numpy()
    y_train = train_dataset.y_train.numpy()

    c_test = train_dataset.protected_test.numpy()
    X_test = train_dataset.X_test.numpy()
    y_test = train_dataset.y_test.numpy()

    data = {}
    data['train'] = [ X_train, c_train, y_train ]
    data['test'] = [ X_test, c_test, y_test ]
    data['ft_pos'] = train_dataset.ft_pos

    return data
