from src.common.dataset import get_dataset
from box import Box 
import sys
import os 

if __name__ == '__main__':
    sig = sys.argv[2]
    dataACS, meta = get_dataset(Box({'name': sys.argv[1], 'val_size': 0}))
    #os.system('rm src/laftr/data/acs/*')
    import numpy as np 
    np.save(f'src/laftr/data/acs/{sys.argv[1]}_{sig}.npy', dataACS)
    print(f'SAVED WITH SIG: {sig}')