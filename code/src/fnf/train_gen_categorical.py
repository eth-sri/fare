import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .made import MADE


def train_gen_cat(args, X_train, A_train, X_valid, A_valid):
    device = args.device

    train_all = X_train
    train_prot = A_train
    train1 = train_all[train_prot == 1]
    train2 = train_all[train_prot == 0]

    valid_all = X_valid
    valid_prot = A_valid
    valid1 = valid_all[valid_prot == 1]
    valid2 = valid_all[valid_prot == 0]

    train1_loader = torch.utils.data.DataLoader(train1, batch_size=args.batch_size, shuffle=True)
    train2_loader = torch.utils.data.DataLoader(train2, batch_size=args.batch_size, shuffle=True)

    valid1_loader = torch.utils.data.DataLoader(valid1, batch_size=args.batch_size, shuffle=True)
    valid2_loader = torch.utils.data.DataLoader(valid2, batch_size=args.batch_size, shuffle=True)

    made1 = MADE(X_train.shape[1], [args.hidden, args.hidden]).to(device)
    made2 = MADE(X_train.shape[1], [args.hidden, args.hidden]).to(device)

    opt_made1 = optim.Adam(list(made1.parameters()), lr=1e-2, weight_decay=1e-4)
    opt_made2 = optim.Adam(list(made2.parameters()), lr=1e-2, weight_decay=1e-4)

    lr_scheduler1 = optim.lr_scheduler.StepLR(opt_made1, step_size=args.n_epochs//2, gamma=0.1)
    lr_scheduler2 = optim.lr_scheduler.StepLR(opt_made2, step_size=args.n_epochs//2, gamma=0.1)

    best_valid_loss1, best_valid_loss2 = None, None

    if not os.path.exists(f'{args.dataset}'):
        os.makedirs(f'{args.dataset}')

    for epoch in range(args.n_epochs):
        print(f'[gen cat] epoch {epoch}/{args.n_epochs}', flush=True)
        tot_loss1, n_batches1 = 0, 0
        for inputs1 in train1_loader:
            opt_made1.zero_grad()
            n_batches1 += 1
            inputs1 = inputs1.to(device)
            outs1 = made1(inputs1)
            loss = F.binary_cross_entropy_with_logits(outs1, inputs1, reduction='none')
            loss = loss.sum(-1).mean()
            loss.backward()
            opt_made1.step()
            tot_loss1 += loss.item()

        tot_loss2, n_batches2 = 0, 0
        for inputs2 in train2_loader:
            opt_made2.zero_grad()
            n_batches2 += 1
            inputs2 = inputs2.to(device)
            outs2 = made2(inputs2)
            loss = F.binary_cross_entropy_with_logits(outs2, inputs2, reduction='none')
            loss = loss.sum(-1).mean()
            loss.backward()
            opt_made2.step()
            tot_loss2 += loss.item()

        if (epoch+1) % 20 == 0:
            print('epoch: %d, loss1: %.4f, loss2: %.4f' % (epoch+1, tot_loss1/n_batches1, tot_loss2/n_batches2))

        with torch.no_grad():
            tot_loss1, tot_loss2, n_batches = 0, 0, 0

            inputs1 = valid1.to(device)
            outs1 = made1(inputs1)
            loss = F.binary_cross_entropy_with_logits(outs1, inputs1, reduction='none')
            valid_loss1 = loss.sum(-1).mean()

            inputs2 = valid2.to(device)
            outs2 = made2(inputs2)
            loss = F.binary_cross_entropy_with_logits(outs2, inputs2, reduction='none')
            valid_loss2 = loss.sum(-1).mean().item()

            if (epoch + 1) % 20 == 0:
                print('[valid] epoch: %d, loss1: %.4f, loss2: %.4f' % (epoch+1, valid_loss1, valid_loss2))
            if best_valid_loss1 is None or valid_loss1 < best_valid_loss1:
                if args.verbose:
                    print('best valid_loss_1, saving network...')
                best_valid_loss1 = valid_loss1
                torch.save(made1, 'src/fnf/%s/made1.pt' % (args.dataset))
            if best_valid_loss2 is None or valid_loss2 < best_valid_loss2:
                if args.verbose:
                    print('best valid_loss_2, saving network...')
                best_valid_loss2 = valid_loss2
                torch.save(made2, 'src/fnf/%s/made2.pt' % (args.dataset))

        lr_scheduler1.step()
        lr_scheduler2.step()

