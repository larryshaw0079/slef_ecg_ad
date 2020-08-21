import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm.std import tqdm

from src.data import ECGDataset
from src.utils import RandomTransformation


def parse_args():
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

    return parser.parse_args()


def train(model, args, writer, transformation, train_loader, val_loader, criterion, optimizer):
    ############################################################
    #                         Training                         #
    ############################################################
    model.train()
    for i in range(args.epoch):
        train_losses = []
        with tqdm(train_loader, desc='EPOCH [%d/%d]' % (i + 1, args.epoch)) as loader:
            for x in loader:
                x = transformation.apply_transformation_all(x)
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
    print(args)

    # GPU setting
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(-1 * np.array(gpu_memory))
    os.system('rm tmp')
    assert (args.num_gpu <= len(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids[:args.num_gpu]))
    print('Current GPU [%s], free memory: [%s] MB' % (
        os.environ['CUDA_VISIBLE_DEVICES'], ','.join(map(str, np.array(gpu_memory)[gpu_ids[:args.num_gpu]]))))

    # Set the random seed
    if args.seed is not None:
        print('Setting manual seed %d...' % args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    writer = SummaryWriter()

    train_samples, val_samples, val_labels, test_samples, test_labels = prepare_data(args.data)

    train_dataset = ECGDataset(train_samples, normalize=True, num_channel=args.num_channel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    val_dataset = ECGDataset(val_samples, val_labels, normalize=True, num_channel=args.num_channel)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    test_dataset = ECGDataset(test_samples, test_labels, normalize=True, num_channel=args.num_channel)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)

    model = Classifier(input_length=args.length, input_channel=args.num_channel, num_class=args.num_trans).cuda()
    summary(model, input_size=(args.num_channel, args.length), batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    # trans_mats = np.random.randn(args.num_trans, train_samples.shape[-1], train_samples.shape[-1])
    transformation = RandomTransformation(train_samples.shape[-1], args.num_trans, normalize=args.normalize)
    if args.resume:
        print('Loading checkpoint...')
        try:
            with open(os.path.join(args.save, 'model.pth'), 'rb') as f:
                model.load_state_dict(torch.load(f))
        except FileNotFoundError:
            print('Checkpoint not found, start training...')
            train(model, args, writer, transformation, train_loader, val_loader, criterion, optimizer)
    else:
        print('Start training...')
        train(model, args, writer, transformation, train_loader, val_loader, criterion, optimizer)

    print('Start evaluating...')
    performance, roc_fig, pr_fig = evaluate(model, args, writer, transformation, train_loader, test_loader,
                                            test_mode='test', score_mode=args.score_mode)

    roc_fig.savefig(os.path.join(args.save, 'roc_curve.svg'))
    pr_fig.savefig(os.path.join(args.save, 'pr_curve.svg'))
    print(performance)

    writer.close()
