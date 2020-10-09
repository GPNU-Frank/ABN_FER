'''
Training script for CK+/6
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import models.cifar as models
from models import ResNetAndGCN, Model, ModelAG, ModelAGATT
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, pickle_2_img_single, pickle_2_img_and_landmark, pickle_2_img_and_landmark_ag
import logging
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser(description='PyTorch ckp Training')
# Datasets
parser.add_argument('-d', '--dataset', default='ckp', type=str)
parser.add_argument('--dataset-path', default='data/ck+_img_and_55_landmark_3_frame_A+G_separate_224.pickle')  # windows style
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
parser.add_argument('-f', '--folds', default=10, type=int, help='k-folds cross validation.')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=4, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 30, 70],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.95, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints/ckp_stgcn_ag_att', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='stgcn_ag_att')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
# parser.add_argument('--use-gpu', action='store_true',
#                     help='use gpu')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'ckp', 'Dataset can only be ckp.'

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # load data
    print('==> Preparing dataset %s' % args.dataset)
    features, landmarks_g, landmarks_h, labels = pickle_2_img_and_landmark_ag(args.dataset_path)
    num_classes = 6

    # Model
    print("==> creating model '{}'".format(args.arch))
    # model = ResNetAndGCN(20, num_classes=num_classes)
    model = ModelAGATT(2, 36, 6, {}, False, dropout=0.3)
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # print('    resnet params: %.2fM' % (sum(p.numel() for p in model.resnet.parameters())/1000000.0))
    # print('    stgcn params: %.2fM' % (sum(p.numel() for p in model.st_gcn.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    # 分层优化
    # resnet_para = [model.conv1.parameters(), model.layer1.parameters(), model.layer2.parameters(), model.layer3.parameters(), model.layer4.parameters()]
    # optimizer = optim.SGD([
    #     {'params': model.gcn11.parameters()}, 
    #     {'params': model.gcn12.parameters()}, 
    #     {'params': model.gcn21.parameters()}, 
    #     {'params': model.gcn22.parameters()}, 
    #     {'params': model.gcn31.parameters()}, 
    #     {'params': model.gcn32.parameters()}, 
    #     {'params': model.fc.parameters()}, 
    #     {'params': model.conv1.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
    #     {'params': model.bn1.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
    #     {'params': model.layer1.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
    #     {'params': model.layer2.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
    #     {'params': model.layer3.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
    #     {'params': model.layer4.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
    #     ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Resume
    title = 'ckp-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log_stat.log'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stat.log'), title=title)
        logger.set_names(['fold_num', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # logging
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.checkpoint, 'log_info.log'),
                        filemode='a+',
                        format="%(asctime)-15s %(levelname)-8s  %(message)s")
    # log configuration
    logging.info('-' * 10 + 'configuration' + '*' * 10)
    for arg in vars(args):
        logging.info((arg, str(getattr(args, arg))))

    acc_fold = []
    reset_lr = state['lr']
    for f_num in range(args.folds):
        state['lr'] = reset_lr
        model.reset_all_weights()
        # optimizer = optim.SGD([
        # {'params': model.gcn11.parameters()}, 
        # {'params': model.gcn12.parameters()}, 
        # {'params': model.gcn21.parameters()}, 
        # {'params': model.gcn22.parameters()}, 
        # {'params': model.gcn31.parameters()}, 
        # {'params': model.gcn32.parameters()}, 
        # {'params': model.fc.parameters()}, 
        # {'params': model.conv1.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
        # {'params': model.bn1.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
        # {'params': model.layer1.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
        # {'params': model.layer2.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
        # {'params': model.layer3.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
        # {'params': model.layer4.parameters(), 'lr': 0.005, 'weight_decay': 5e-3},
        # ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(args.lr)
        # save each fold's acc and reset configuration
        average_acc = 0
        best_acc = 0
    

        # 10-fold cross validation
        train_x, train_lm_g, train_lm_h, train_y = [], [], [], []
        test_x, test_lm_g, test_lm_h, test_y = [], [], [], []
        for id_fold in range(args.folds):
            if id_fold == f_num:
                test_x = features[id_fold]
                test_lm_g = landmarks_g[id_fold]
                test_lm_h = landmarks_h[id_fold]
                test_y = labels[id_fold]
            else:
                train_x = train_x + features[id_fold]
                train_lm_g = train_lm_g + landmarks_g[id_fold]
                train_lm_h = train_lm_h + landmarks_h[id_fold]
                train_y = train_y + labels[id_fold]
        # convert array to tensor
        train_x = torch.tensor(train_x, dtype=torch.float) / 255.0  #(b_s, 128, 128)
        train_x = train_x.unsqueeze(1)  #(b_s, 1, 128, 128)

        train_lm_g = np.stack(train_lm_g)
        train_lm_h = np.stack(train_lm_h)
        # 只要坐标信息， 不需要归一化
        # train_lm_g = (train_lm_g - np.mean(train_lm_g, axis=0)) / np.std(train_lm_g, axis=0)
        # train_lm_g = (train_lm_g - np.amin(train_lm_g)) / np.amax(train_lm_g) - np.amin(train_lm_g)
        # train_lm_h = (train_lm_h - np.mean(train_lm_h, axis=0)) / np.std(train_lm_h, axis=0)
        train_lm_g = torch.tensor(train_lm_g, dtype=torch.float)
        train_lm_h = torch.tensor(train_lm_h, dtype=torch.float)
        # train_lm = train_lm.unsqueeze(2)

        test_x = torch.tensor(test_x, dtype=torch.float) / 255.0
        test_x = test_x.unsqueeze(1)
        # 只要坐标信息， 不需要归一化
        # test_lm_g = (test_lm_g - np.mean(test_lm_g, axis=0)) / np.std(test_lm_g, axis=0)
        # test_lm_g = (test_lm_g - np.amin(test_lm_g)) / np.amax(test_lm_g) - np.amin(test_lm_g)
        # test_lm_h = (test_lm_h - np.mean(test_lm_h, axis=0)) / np.std(test_lm_h, axis=0)
        test_lm_g = torch.tensor(test_lm_g, dtype=torch.float)
        test_lm_h = torch.tensor(test_lm_h, dtype=torch.float)
        # test_lm = test_lm.unsqueeze(2)
        train_y, test_y = torch.tensor(train_y), torch.tensor(test_y)

        train_dataset = torch.utils.data.TensorDataset(train_x, train_lm_g, train_lm_h, train_y)
        train_iter = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch,
            shuffle=True
        )

        test_dataset = torch.utils.data.TensorDataset(test_x, test_lm_g, test_lm_h, test_y)
        test_iter = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch,
            shuffle=False
        )

        # test for fold order
        print(len(test_dataset))

        if args.evaluate:
            print('\nEvaluation only')
            test_loss, test_acc = test(train_x + test_x, train_y + test_y, model, criterion, start_epoch, use_cuda)
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
            continue

        # show plt
        # plt.show(block=False)

        # Train and val
        for epoch in range(start_epoch, args.epochs):
            
            # 在特定的epoch 调整学习率
            adjust_learning_rate(optimizer, epoch)
            # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

            train_loss, train_acc = train(train_iter, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(test_iter, model, criterion, epoch, use_cuda)

            # append logger file
            logger.append([f_num, state['lr'], train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, f_num, checkpoint=args.checkpoint)

        # compute average acc
        acc_fold.append(best_acc)
        average_acc = sum(acc_fold) / len(acc_fold)

        logging.info('fold: %d, best_acc: %.2f, average_acc: %.2f' % (f_num, best_acc, average_acc))
    logger.close()
    # logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    logging.info('acc_fold' + str(acc_fold))
    print('average acc:')
    print(average_acc)


def train(train_iter, model, criterion, optimizer, epoch, use_cuda):

    # set train_loader，batch size 太大跑不动
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_iter))
    for batch_idx, (inputs, landmarks_g, landmarks_h, targets) in enumerate(train_iter):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, landmarks_g, landmarks_h, targets = inputs.cuda(), landmarks_g.cuda(), landmarks_h.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        per_outputs = model(landmarks_g, landmarks_h)

        # # 采用 L1 正则化
        # regularization_loss = 0
        # for param in model.parameters():
        #     regularization_loss += torch.sum(torch.abs(param))
        per_loss = criterion(per_outputs, targets)

        loss = per_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(per_outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(inputs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(test_iter, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(test_iter))
    for batch_idx, (inputs, landmarks_g, landmarks_h, targets) in enumerate(test_iter):
    # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, landmarks_g, landmarks_h, targets = inputs.cuda(), landmarks_g.cuda(), landmarks_h.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(landmarks_g, landmarks_h)
        loss = criterion(outputs, targets)

        """
        np_inputs = inputs.numpy()
        np_att = attention.numpy()
        for item_in, item_att in zip(np_inputs, np_att):
            print(item_in.shape, item_att.shape)
        """

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(inputs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, f_num, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'fold_' + str(f_num) + '_' + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'fold_' + str(f_num) + '_model_best.pth.tar'))


# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


if __name__ == '__main__':
    main()
