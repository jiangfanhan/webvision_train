
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import tensorflow as tf
import pandas as pd
import time
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models,datasets
from inceptionresnetv2 import inceptionresnetv2
import torch.backends.cudnn as cudnn
import csv
from imagenet_val_dataset import imagenet_val, reduce_imagenet_val
import tensorboard
from logger import Logger
from my_dataloader import my_dataloader
from my_network import load_pretrained_model, load_scratch_model

import argparse

parser = argparse.ArgumentParser(description='train imagenet or webvision')
parser.add_argument('--dataset', default='webvision', type=str, help='which dataset to use')
parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='mini batch size')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--epochs', default=100, type=int, help='epochs to train')
parser.add_argument('--resume', action='store_true', help='if need use ckpt' )
parser.add_argument('--ckpt_path', default='pytorch_train/checkpoint/ckpt_350000_epoch37', type=str, help='path to loaded checkpoint')
parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay factor')
parser.add_argument('--decay_epochs', default=25, type=int, help='frequency to decay the lr')
parser.add_argument('--validation',action='store_true', help='turn the model to validation mode')
parser.add_argument('--network', default= 'alexnet', type=str, help='network used to train' )
parser.add_argument('--warm_start', action='store_true', help='using warm start method')
args = parser.parse_args()

loader = my_dataloader(args.dataset, args.batch_size)
use_cuda = torch.cuda.is_available()
file_time = time.strftime('%Y-%m-%d-%H:%M:%S')

def val_in_train(model, criterion, iter):
    print 'validate on the validation set'
    model.eval()
    val_batch_size = args.batch_size / 2
    loader = my_dataloader(args.dataset, val_batch_size)
    val_loader = loader.val_loader()
    train_loss = 0.0
    total_loss = 0.0
    correct = 0
    corr_top5 =0
    total_corr1 = 0
    total_corr5 = 0
    total = 0
    subtotal = 0
    to_num = 0
    since = time.time()
    for num, (image_batched,label_batched)in enumerate(val_loader):
        if use_cuda:
            image_batched, label_batched = image_batched.cuda(), label_batched.cuda()
        image_batched, label_batched = Variable(image_batched), Variable(label_batched)
        outputs = model(image_batched)
        loss = criterion(outputs, label_batched)
        train_loss += loss.data[0]
        total_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        _, predicted_5 = torch.topk(outputs.data, 5)
        for i in range(5):
            corr_top5 += predicted_5[:, i].eq(label_batched.data).cpu().sum()
            total_corr5 += predicted_5[:, i].eq(label_batched.data).cpu().sum()
        total += label_batched.size(0)
        correct += predicted.eq(label_batched.data).cpu().sum()
        total_corr1 += predicted.eq(label_batched.data).cpu().sum()
        subtotal += label_batched.size(0)
        to_num += 1
        if to_num % 100 == 0:
            time1 = time.time()
            duration = time1 - since
            acc_top1 = 100. * correct / subtotal
            acc_top5 = 100. * corr_top5 / subtotal
            print ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
                to_num, val_batch_size * (to_num), train_loss / 100, acc_top1, acc_top5, corr_top5, subtotal))
            print ('sample/second = {}'.format(val_batch_size * 100 / duration))
            correct = 0
            corr_top5 = 0
            subtotal = 0
            train_loss = 0.0
            since = time1
    print 'Average on the validation set'
    to_acc1 = 100. * total_corr1 / total
    to_acc5 = 100. * total_corr5 / total
    info = {
        'val_loss': total_loss / to_num,
        'val_top1': to_acc1,
        'val_top5': to_acc5
    }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, iter)
    print 'Num of pictures: {}\nLoss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
        total, total_loss/to_num, to_acc1 , to_acc5, total_corr5, total)
    return to_acc5

def train_model(model, criterion, optimizer ,scheduler ,start_iter, best_acc ,start_epoch, num_epochs):
    model.train()
    train_loader = loader.train_loader()
    train_loss = 0.0
    correct = 0
    total = 0
    subtotal = 0
    corr_top5 =0
    to_num = start_iter
    since = time.time()
    remain = start_epoch % scheduler.step_size
    train_epoch = num_epochs - start_epoch
    for i in range(remain):
        scheduler.step()
    for iter_epoch in range(train_epoch):
        epoch = iter_epoch + start_epoch
        model.train()
        scheduler.step()
        for num , (image_batched,label_batched)in enumerate(train_loader):
            if use_cuda:
                image_batched, label_batched = image_batched.cuda(), label_batched.cuda()
            image_batched, label_batched = Variable(image_batched), Variable(label_batched)
            optimizer.zero_grad()
            outputs =model(image_batched)
            loss = criterion(outputs, label_batched)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_5 = torch.topk(outputs.data,5)
            for i in range(5):
                corr_top5 += predicted_5[:,i].eq(label_batched.data).cpu().sum()
            total += label_batched.size(0)
            correct += predicted.eq(label_batched.data).cpu().sum()
            subtotal +=label_batched.size(0)
            to_num +=1 
            if to_num % 100 == 0:
                time1 = time.time()
                duration = time1 - since
                acc_top1 = 100.*correct/subtotal
                acc_top5 = 100.*corr_top5/subtotal
                print ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
                    to_num, args.batch_size*(to_num), train_loss/100, acc_top1, acc_top5, corr_top5, subtotal))
                print ('sample/second = {}'.format(args.batch_size* 100 / duration))
                print >> f, ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
                    to_num, args.batch_size*(to_num), train_loss/100, acc_top1, acc_top5, corr_top5, subtotal))
                print >> f, ('sample/second = {}'.format(args.batch_size * 100 / duration))
                info = {
                    'loss': train_loss/100,
                    'top1': acc_top1,
                    'top5': acc_top5,
                    'learning rate': optimizer.param_groups[0]['lr']
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, to_num)
                correct = 0
                corr_top5 = 0
                subtotal = 0
                train_loss = 0.0
                since = time1
        curr_acc = val_in_train(net, criterion, to_num)
        if best_acc <= curr_acc:
            best_acc = curr_acc
            print 'saving..'
            state = {
                'state_dict': net.state_dict(),
                'acc': curr_acc,
                'num_iter': to_num,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epochs': epoch
            }
            if not os.path.isdir('checkpoint/{}-{}-{}'.format(args.dataset, args.network, file_time)):
                os.mkdir('checkpoint/{}-{}-{}'.format(args.dataset, args.network, file_time))
            torch.save(state, './checkpoint/{}-{}-{}/ckpt-{}-epoch{}'.format(
                args.dataset, args.network, file_time, to_num, epoch + 1))

            ## keep the number of files in the folder
            file_list = os.listdir('checkpoint/{}-{}-{}'.format(args.dataset, args.network, file_time))
            file_num = []
            for ele in file_list:
                ele_list = ele.split('-')
                num = int(ele_list[1])
                file_num.append((num, ele))
            file_num.sort(key=lambda file_nums: file_nums[0])
            if len(file_num) > 10:
                os.remove(os.path.join('checkpoint/{}-{}-{}'.format(args.dataset, args.network, file_time), file_num[0][1]))

def validate_model(model, criterion):
    print 'validate on the validation set'
    model.eval()
    val_loader = loader.val_loader()
    train_loss = 0.0
    total_loss = 0.0
    correct = 0
    corr_top5 =0
    total_corr1 = 0
    total_corr5 = 0
    total = 0
    subtotal = 0
    to_num = 0
    since = time.time()
    for num, (image_batched,label_batched)in enumerate(val_loader):
        if use_cuda:
            image_batched, label_batched = image_batched.cuda(), label_batched.cuda()
        image_batched, label_batched = Variable(image_batched), Variable(label_batched)
        outputs = model(image_batched)
        loss = criterion(outputs, label_batched)
        train_loss += loss.data[0]
        total_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        _, predicted_5 = torch.topk(outputs.data, 5)
        for i in range(5):
            corr_top5 += predicted_5[:, i].eq(label_batched.data).cpu().sum()
            total_corr5 += predicted_5[:, i].eq(label_batched.data).cpu().sum()
        total += label_batched.size(0)
        correct += predicted.eq(label_batched.data).cpu().sum()
        total_corr1 += predicted.eq(label_batched.data).cpu().sum()
        subtotal += label_batched.size(0)
        to_num += 1
        if to_num % 100 == 0:
            time1 = time.time()
            duration = time1 - since
            acc_top1 = 100. * correct / subtotal
            acc_top5 = 100. * corr_top5 / subtotal
            print ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
                to_num, args.batch_size * (to_num), train_loss / 100, acc_top1, acc_top5, corr_top5, subtotal))
            print ('sample/second = {}'.format(args.batch_size * 100 / duration))
            print >> f, ('[{} {}] Loss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
                to_num, args.batch_size * (to_num), train_loss / 100, acc_top1, acc_top5, corr_top5, subtotal))
            print >> f, ('sample/second = {}'.format(args.batch_size * 100 / duration))
            correct = 0
            corr_top5 = 0
            subtotal = 0
            train_loss = 0.0
            since = time1
    print 'Average on the validation set'
    to_acc1 = 100. * total_corr1 / total
    to_acc5 = 100. * total_corr5 / total
    print 'Num of pictures: {}\nLoss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
        total, total_loss/to_num, to_acc1 , to_acc5, total_corr5, total)
    print >> f, 'Num of pictures: {}\nLoss: {:.3f} | Acc_top1: {:.3f} | Acc_top5: {:.3f} ({}/{})'.format(
        total, total_loss/to_num, to_acc1 , to_acc5, total_corr5, total)



if args.pretrained:
    print '=>load pretrained model'
    net = load_pretrained_model(args.network)
else:
    print '=>creating new model or load from checkpoint'
    net = load_scratch_model(args.network)

    if args.dataset == 'renotate_webvision_reduce' or args.dataset == 'reduce_imagenet':
        net.classifier._modules['6'] = nn.Linear(4096, 957)
best_acc = 0.0
start_iter = 0
lr = args.lr
start_epoch = 0
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.resume:
    print '=>loading from checkpoint from {}'.format(args.ckpt_path)
    assert os.path.isdir('checkpoint'), 'Error no directory found!'
    checkpoint = torch.load(os.path.join('/home/jfhan',args.ckpt_path))
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    best_acc = checkpoint['acc']
    start_iter = checkpoint['num_iter']
    lr = checkpoint['learning_rate']
    start_epoch = checkpoint['epochs']
    if args.warm_start:
        print '=> using warm start method'
        lr = args.lr
        best_acc = 0.0
        start_iter = 0
        start_epoch = 0
else:
    print '=>Just create new model'




criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs, gamma=args.lr_decay)
if not args.validation:
    if not os.path.isdir('logs/{}-{}-{}'.format(args.dataset, args.network,file_time)):
        os.mkdir('logs/{}-{}-{}'.format(args.dataset, args.network, file_time))
    logger = Logger('./logs/{}-{}-{}'.format(args.dataset, args.network, file_time))

f = open('./log-{}.txt'.format(file_time), 'a+')
if args.resume:
    print >> f, ('resume a checkpoint:{}'.format(args.resume))
    print >> f, ('checkpoint{}'.format(args.ckpt_path))
else:
    print >> f, ('resume a checkpoint:{}'.format(args.resume))
print >> f, ('basic settings')
print >> f, ('dataset:{}'.format(args.dataset))
print >> f, ('learning_rate:{}'.format(lr))
print >> f, ('batch size:{}'.format(args.batch_size))
print >> f, ('momentum:{}'.format(args.momentum))
print >> f, ('netoork:{}').format(args.network)
print >> f, ('epochs/decay:{}'.format(args.decay_epochs))
print >> f, ('total_epoch:{}').format(args.epochs)
print >> f, ('lr_decay_factor:{}'.format(args.lr_decay))
print >> f, ('validation:{}'.format(args.validation))
if not args.validation:
    train_model(net,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=exp_lr_scheduler,
                start_iter=start_iter,
                best_acc=best_acc,
                start_epoch=start_epoch,
                num_epochs=args.epochs)
else:
    validate_model(net, criterion=criterion)
