import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.std import tqdm


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', type=str, default='./data/mit_ecg_processed/')
    parser.add_argument('--save', dest='save', type=str, default='./cache/')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--norm', dest='normalize', action='store_true')
    parser.add_argument('--score', dest='score_mode', choices=['simple', 'dirichlet'], required=True)

    parser.add_argument('--length', dest='length', type=int, default=320)
    parser.add_argument('--channel', dest='num_channel', type=int, default=2)
    parser.add_argument('--trans', dest='num_trans', type=int, default=32)
    parser.add_argument('--batch', dest='batch_size', type=int, default=128)
    parser.add_argument('--epoch', dest='epoch', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)

    parser.add_argument('--gpu', dest='num_gpu', type=int, default=1)
    parser.add_argument('--seed', dest='seed', type=int, default=2020)

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def train(model, args, writer, transformation, train_loader, val_loader, criterion, optimizer):
    ############################################################
    #                         Training                         #
    ############################################################
    model.train()
    for i in range(args.epoch):
        train_losses = []
        with tqdm(train_loader, desc='EPOCH [%d/%d]' % (i + 1, args.epoch)) as loader:
            for x in loader:
                x = transformation.apply_transformation_all(x)  # (batch, feature)
                y = torch.cat([torch.full((args.batch_size,), i, dtype=torch.long) for i in range(args.num_trans)])
                x, y = x.cuda(), y.cuda()

                y_hat = model(x)

                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                loader.set_postfix({'loss': np.mean(train_losses)})

        performance = evaluate(model, args, writer, transformation, train_loader, val_loader, test_mode='val',
                               score_mode=args.score_mode)
        model.train()
        writer.add_scalar('Loss/train', np.mean(train_losses), i)
        writer.add_scalar('Metrics/f1', performance['f1'], i)
        writer.add_scalar('Metrics/precision', performance['precision'], i)
        writer.add_scalar('Metrics/recall', performance['recall'], i)

        # scheduler.step()

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    with open(os.path.join(args.save, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    args = parse_args()

    # Set the random seed
    if args.seed is not None:
        print('Setting manual seed %d...' % args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
