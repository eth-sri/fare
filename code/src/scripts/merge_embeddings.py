# when evaluating with % call this to merge 
# todo add command-line etc etc

#"""Take a folder... go to all folders and look for config.json"""
import argparse
#import importlib
#import logging
import os
#import datetime
import numpy as np
from box import Box 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='none')
    parser.add_argument("--name", default='none')
    parser.add_argument("--mod", type=int, default=11)

    args = parser.parse_args()
    final = {
        'checksums': {}
    }

    for i in range(args.mod):
        path = f'result/_eval/{args.dataset}/{args.name}_{i}%{args.mod}.npy'
        curr = np.load(path, allow_pickle = True).item().to_dict()
        for k, v in curr.items():
            if k != 'checksums':
                final[k] = v 

    final = Box(final)
    path = f'result/_eval/{args.dataset}/{args.name}.npy'
    np.save(path, final)
    print(f'Merged to {path}')


