import logging

from box import Box
import numpy as np 

from src.common.datasets.acs import load_acs
from src.common.datasets.adult import load_adult
from src.common.datasets.health_preprocessed import load_health as load_health_preprocessed

logger = logging.getLogger()


def get_dataset(data_config):
    """ Take config and return dataloader"""
    dataname = data_config.name
    val_size = data_config.val_size
    all_labels = None

    # Defaults, if different then overwrite
    y_size = 2 
    y_type = "binary"
    c_size = 2
    c_type = "binary"

    if dataname == "adult": #old
        data = load_adult()
    elif dataname == 'health_notransfer': # with FNF preprocessing, all features
        data = load_health_preprocessed(label=['max_CharlsonIndex'],p_test=0.2)
    elif dataname == 'health_transfer': # with FNF preprocessing for transfer, depreecate
        data = load_health_preprocessed(label=['max_CharlsonIndex'],transfer=True,p_test=0.2)
    elif dataname == 'health_transfer_all':
        prefix = 'PrimaryConditionGroup='
        all_labels = ['max_CharlsonIndex', 'MSC2a3', 'METAB3', 'ARTHSPIN', 'NEUMENT', 'RESPR4', 'MISCHRT', 'SKNAUT', 'GIBLEED', 'INFEC4', 'TRAUMA']
        
        all_labels_prefixed = [all_labels[0]]
        for i in range(1, 11):
            all_labels_prefixed.append(f'{prefix}{all_labels[i]}')
        data = load_health_preprocessed(label=all_labels_prefixed,transfer=True,p_test=0.2)
    elif dataname == 'health_transfer_fnfpruned':
        data = load_health_preprocessed(label=['max_CharlsonIndex'],transfer=True, fnfpruned=True, p_test=0.2)
    elif dataname == 'health_transfer_fnfpruned_all':
        prefix = 'PrimaryConditionGroup='
        all_labels = ['max_CharlsonIndex', 'MSC2a3', 'METAB3', 'ARTHSPIN', 'NEUMENT', 'RESPR4', 'MISCHRT', 'SKNAUT', 'GIBLEED', 'INFEC4', 'TRAUMA']
        
        all_labels_prefixed = [all_labels[0]]
        for i in range(1, 11):
            all_labels_prefixed.append(f'{prefix}{all_labels[i]}')
        data = load_health_preprocessed(label=all_labels_prefixed,transfer=True, fnfpruned=True, p_test=0.2)
    elif dataname.startswith('ACS'):
        half = None
        if dataname.endswith('-L'):
            dataname = dataname[:-2]
            half = True
        if dataname.endswith('-R'):
            dataname = dataname[:-2]
            half = False
        split = dataname.rsplit("-M=", 1)
        M = 1
        if len(split) == 2:
            dataname, M = split
            M = int(M)
        if ',' in dataname:
            names = dataname.split(',')
            name, years = names[0], names[1:]
            datas = []
            for year in years:
                data = load_acs(name+year)
                datas.append( data )
            keys = datas[0].keys()
            data_total = {}
            data_total['ft_pos'] = datas[0]['ft_pos']
            for k in ['train','test']:
                combos = []
                for ks in range(3):
                    combo = [ datas[i][k][ks] for i in range(len(datas)) ]
                    combo = np.concatenate(combo, axis=0)
                    combos.append( combo ) 
                data_total[k] = tuple(combos)
            data = data_total
        else:
            data = load_acs(dataname)
        if M > 1:
            data['train'] = ( np.repeat( data['train'][0], M, axis=0 ), np.repeat( data['train'][1], M, axis=0 ), np.repeat( data['train'][2], M, axis=0 ) )
            data['test'] = ( np.repeat( data['test'][0], M, axis=0 ), np.repeat( data['test'][1], M, axis=0 ), np.repeat( data['test'][2], M, axis=0 ) )
        if half is True:
            nrows = data['train'][0].shape[0]
            nrows_half = nrows // 2
            data['train'] = data['train'][0][:nrows_half], data['train'][1][:nrows_half], data['train'][2][:nrows_half]
            nrows = data['test'][0].shape[0]
            nrows_half = nrows // 2
            data['test'] = data['test'][0][:nrows_half], data['test'][1][:nrows_half], data['test'][2][:nrows_half]
        if half is False:
            nrows = data['train'][0].shape[0]
            nrows_half = nrows // 2
            data['train'] = data['train'][0][nrows_half:], data['train'][1][nrows_half:], data['train'][2][nrows_half:]
            nrows = data['test'][0].shape[0]
            nrows_half = nrows // 2
            data['test'] = data['test'][0][nrows_half:], data['test'][1][nrows_half:], data['test'][2][nrows_half:]
        c_size = 2
        c_type = "binary"
        if dataname.startswith('ACSIncomeMulti'): # multiclass + multisens
            y_size = 4
            y_type = "multi"
            c_size = 3
            c_type = "multi"

    else:
        logger.error(f"Invalid data name {dataname} specified")
        raise Exception(f"Invalid data name {dataname} specified")

    if val_size > 0: 
        print('each method should split validation internally!')
        import code; code.interact(local=dict(globals(), **locals()))

    train, test = data["train"], data["test"]

    # (X, s, y)
    meta = {
        "input_shape": train[0].shape[1:],
        "c_size": c_size,
        "c_type": c_type,
        "y_size": y_size,
        "y_type": y_type,
    }
    if all_labels is not None:
        meta['all_labels'] = all_labels 
        
    if 'ft_pos' in data:
        meta['ft_pos'] = data['ft_pos']
    
    ret = (
        Box({"train": (
            train[0].astype(np.float32),
            train[1].astype(np.long),
            train[2].astype(np.long)
        ), "test": (
            test[0].astype(np.float32),
            test[1].astype(np.long),
            test[2].astype(np.long)
        )}), meta)
    
    return ret
