import argparse
import numpy as np
import os
from copy import deepcopy

from .config import yaml_config_hook
from .runner import _runner
from .misc import save_result

# https://github.com/kwkimonline/sIPM-LFR

if __name__ == '__main__':
    seed = 100
    ''' parsers '''
    parser = argparse.ArgumentParser(description='Learning Fair Representation via sIPM')
    config = yaml_config_hook('src/sipm/config.yaml')
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    parser.add_argument('--resultdir', type=str, default=None) # result/
    args = parser.parse_args()
    print(args)

    rundir = f'{args.resultdir}/{args.dataset}/sipm/lmda={args.lmda},lmdaR={args.lmdaR},lmdaF={args.lmdaF}/'
    if not os.path.exists(rundir):
        os.makedirs(rundir)
        
    ''' running experiments '''
    Runner = _runner(args.dataset, args.scaling,
                     args.batch_size,
                     args.epochs, args.finetune_epochs, args.opt, args.model_lr, args.aud_lr, 
                     args.aud_steps, args.acti, args.num_layer, args.head_net, args.aud_dim,
                     args.eval_freq, rundir=rundir
                    )

    embeddings = Runner.learning(0, seed, args.lmda, args.lmdaF, args.lmdaR)
    np.save(f'{rundir}/embeddings.npy', embeddings, allow_pickle=True)

