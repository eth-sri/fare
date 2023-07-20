import argparse
import csv
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--with_test', action='store_true')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gamma', type=float,required=True)
parser.add_argument('--resultdir', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--metric', type=str, choices=['stat_parity', 'eq_odds', 'eq_opp'])
args = parser.parse_args()

metric = args.metric 

seeds = [100]
batch_size = 256


if args.dataset == 'health_notransfer':
    transfer = False
    label = None
elif args.dataset == 'health_transfer':
    transfer = True
    label = None
elif args.dataset == 'health_transfer_mcs':
    transfer = True
    label = 'PrimaryConditionGroup=MSC2a3'
elif args.dataset == 'health_transfer_meta':
    transfer = True
    label = 'PrimaryConditionGroup=METAB3'
elif args.dataset == 'health_transfer_art':
    transfer = True
    label = 'PrimaryConditionGroup=ARTHSPIN'
elif args.dataset == 'health_transfer_neu':
    transfer = True
    label = 'PrimaryConditionGroup=NEUMENT'
elif args.dataset == 'health_transfer_resp':
    transfer = True
    label = 'PrimaryConditionGroup=RESPR4'
elif args.dataset.startswith('health_transfer_PCG_'):
    label = 'PrimaryConditionGroup=' + args.dataset[len('health_transfer_PCG_'):]
    transfer = True
else:
    assert False
    
if not transfer:
    label = 'max_CharlsonIndex'
    prior_epochs = 80
    adv_epochs = 80
    n_epochs = 80
    kl_start = 0
    kl_end = 10
    log_epochs = 10

    lr = 1e-3
    weight_decay = 0.0
    n_blocks = 6
elif label is None:
    label = 'max_CharlsonIndex'
    prior_epochs = 60
    adv_epochs = 60
    n_epochs = 60
    kl_start = 0
    kl_end = 5
    log_epochs = 10

    lr = 1e-3
    weight_decay = 0.0
    n_blocks = 6
else:
    prior_epochs = 1
    adv_epochs = 1
    n_epochs = 1
    kl_start = 0
    kl_end = 5
    log_epochs = 1

    lr = 1e-3
    weight_decay = 0.0
    n_blocks = 6

p_val = 0.2
p_test = 0.2

for seed in seeds:
    out_file = f'{args.resultdir}logs/health_{seed}.csv'
    
    with open(f'{out_file}', 'w') as csvfile:
        field_names = ['gamma', 'stat_dist', 'valid_unbal_acc', 'valid_bal_acc', 'test_unbal_acc', 'test_bal_acc', 'adv_valid_acc', 'adv_test_acc', 'test_dem_par', 'test_eq_0', 'test_eq_1']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        print(f'Running gamma={args.gamma}')
        cmd = f'python src/fnf/health_flow_multi.py --fair_criterion {metric} --load --prior flow --prior_epochs {prior_epochs} --batch_size {batch_size} --n_epochs {n_epochs} --adv_epochs {adv_epochs} --gamma {args.gamma} --seed {seed} --kl_start {kl_start} --kl_end {kl_end} --log_epochs {log_epochs} --lr {lr} --weight_decay {weight_decay} --n_blocks {n_blocks} --p_val {p_val} --p_test {p_test} --out_file {out_file} --device {args.device} --resultdir {args.resultdir} --dataset {args.dataset}'
        if transfer:
            cmd += ' --transfer'
        if not label is None:
            cmd += f' --label {label}'
        if args.with_test:
            cmd += ' --with_test'
        print(cmd)
        os.system(cmd)

