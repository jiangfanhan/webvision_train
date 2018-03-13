import csv
import torch
from torch.autograd import grad ,Variable
from torchvision import models, transforms, datasets
import torch.backends.cudnn as cudnn
import torch.nn as nn
from folder import ImageFolder
from torch.utils.data import Dataset, DataLoader
from imagenet_val_dataset import imagenet_val, reduce_imagenet_val
from pre_processing import preprocessing
import os
import numpy as np
from  PIL import Image
# index of classes in webvision less than 200 after annotate
ignore_list = [80, 134, 152, 156, 158, 167, 186, 188, 193, 206, 209,
               214, 215, 220, 221, 238, 240, 255, 282, 332, 389, 392, 446, 465, 482, 522, 530,
               542, 543, 585, 590, 639, 729, 744, 790, 804, 810, 818, 826, 833, 848, 885, 901]
ignore_list.reverse()

# origin is the mapping from before to after
origin = [x for x in range(1000)]
for i in ignore_list:
    origin.pop(i)


def get_path(scv_file_path, file_num):
    path_acc = []
    label_acc = []
    for i in range(file_num):
        with open('{}/{}.csv'.format(scv_file_path,i), 'rb') as f:
            reader = csv.reader(f)
            for path, label in reader:
                path_acc.append(path)
                label = int(label)
                label_acc.append(label)
    return path_acc,label_acc

def reduce_get_path(scv_file_path, file_num):
    path_acc = []
    label_acc = []
    for i in range(file_num):
        with open('{}/{}.csv'.format(scv_file_path, i), 'rb') as f:
            reader = csv.reader(f)
            for path, label in reader:
                if int(label) not in ignore_list:
                    path_acc.append(path)
                    label_acc.append(origin.index(int(label)))
    return path_acc, label_acc

def get_google_path(scv_file_path, file_num):
    path_acc = []
    label_acc = []
    for i in range(file_num):
        with open('{}/{}.csv'.format(scv_file_path,i), 'rb') as f:
            reader = csv.reader(f)
            for path, label in reader:
                # path_sep = path.split('/')
                # if path_sep[7] == 'google':
                path_acc.append(path)
                label = int(label)
                label_acc.append(label)
    return path_acc,label_acc

class Webvision(Dataset):
    def __init__(self,path, label, transform=None):

        self.path = path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.path)
    def __getitem__(self, index):
        image_name = self.path[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(self.label[index])
        sample = (image,label)
        return sample

def get_cifar_path (scv_file):
    path_acc = []
    label_acc = []
    with open('{}.csv'.format(scv_file), 'rb') as f:
        reader = csv.reader(f)
        for path, label in reader:
            path_acc.append(path)
            label = int(label)
            label_acc.append(label)
        return path_acc, label_acc

class cifar_train(Dataset):
    def __init__(self, path ,label, transform=None):
        self.path = path
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.path)
    def __getitem__(self, index):
        image_name = self.path[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(self.label[index])
        sample = (image,label)
        return sample



class my_dataloader(object):
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.trans = preprocessing(self.name)

    def train_loader(self):

        if self.name == 'webvision':
            train_path, train_label = get_path(scv_file_path='path/train', file_num=120)

            webvision_train = Webvision(train_path, train_label, transform=self.trans.trans_train())

            return DataLoader(webvision_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'renotate_webvision':

            renotate_label = []
            renotate_path = []
            with open('/home/jfhan/data/merged_sensetime_annotate.txt', 'rb') as f_truth:
                for line in f_truth.readlines():
                    line_list = line.strip('\n').split('/')
                    sp_line = line_list[3].split('"')
                    if sp_line[1] == '\xe6\xad\xa3\xe6\xa0\xb7\xe6\x9c\xac':
                        renotate_label.append(line_list[0])
                        renotate_path.append(
                            os.path.join('/home/jfhan/data/webvision/resized_images', line_list[1], line_list[2],
                                         sp_line[0]))

            renotate_train = Webvision(renotate_path, renotate_label, transform=self.trans.trans_train())

            return DataLoader(renotate_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'renotate_webvision_reduce':
            renotate_label = []
            renotate_path = []
            with open('/home/jfhan/data/merged_sensetime_annotate.txt', 'rb') as f_truth:
                for line in f_truth.readlines():
                    line_list = line.strip('\n').split('/')
                    sp_line = line_list[3].split('"')
                    if sp_line[1] == '\xe6\xad\xa3\xe6\xa0\xb7\xe6\x9c\xac':
                        if int(line_list[0]) not in ignore_list:
                            renotate_label.append(origin.index(int(line_list[0])))
                            renotate_path.append(
                                os.path.join('/home/jfhan/data/webvision/resized_images', line_list[1], line_list[2],
                                             sp_line[0]))
            renotate_train = Webvision(renotate_path, renotate_label, transform=self.trans.trans_train())

            return DataLoader(renotate_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'webvision_google_half':

            train_path, train_label = get_google_path(scv_file_path='path/train', file_num=120)

            webvision_train = Webvision(train_path, train_label, transform=self.trans.trans_train())

            return DataLoader(webvision_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'imagenet':
            train_data_dir = '/home/jfhan/data/imagenet_train_pic'

            imagenet_train = datasets.ImageFolder(train_data_dir, transform=self.trans.trans_train())

            return DataLoader(imagenet_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'scan_top5':
            path_acc5 = []
            label_acc5 = []
            with open('final_top5.csv', 'rb') as f:
                reader = csv.reader(f)
                for path, label in reader:
                    path_acc5.append(path)
                    label_acc5.append(int(label))
            webvision_train = Webvision(path_acc5, label_acc5, transform=self.trans.trans_train())

            return DataLoader(webvision_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'mini_webvision':
            path_train = []
            label_train = []
            with open('small_version_webvision.csv', 'rb') as f:
                reader = csv.reader(f)
                for path, label in reader:
                    path_train.append(path)
                    label_train.append(int(label))
            webvision_train = Webvision(path_train,label_train, transform=self.trans.trans_train())
            return DataLoader(webvision_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'mini_scan_top5':
            path_acc5 = []
            label_acc5 = []
            with open('mini_top5.csv', 'rb') as f:
                reader = csv.reader(f)
                for path, label in reader:
                    path_acc5.append(path)
                    label_acc5.append(int(label))
            webvision_train = Webvision(path_acc5, label_acc5, transform=self.trans.trans_train())

            return DataLoader(webvision_train, batch_size=self.batch_size, shuffle=True, num_workers=10)
        elif self.name == 'cifar_10':
            trans = transforms.Compose([transforms.Pad(4),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_path , train_label = get_cifar_path(scv_file='cifar_10')
            cifar_tr = cifar_train(train_path, train_label, transform=trans)
            return DataLoader(cifar_tr, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'cifar_10_n02':
            trans = transforms.Compose([transforms.Pad(4),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_path , train_label = get_cifar_path(scv_file='cifar_10_n02')
            cifar_tr = cifar_train(train_path, train_label, transform=trans)
            return DataLoader(cifar_tr, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'cifar_10_n06':
            trans = transforms.Compose([transforms.Pad(4),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_path , train_label = get_cifar_path(scv_file='cifar_10_n06')
            cifar_tr = cifar_train(train_path, train_label, transform=trans)
            return DataLoader(cifar_tr, batch_size=self.batch_size, shuffle=True, num_workers=10)
        else:
            raise ValueError('Invalid dataset name')

    def val_loader(self):
        if self.name == 'webvision':
            val_path, val_label = get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(val_path, val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'renotate_webvision':
            val_path, val_label = get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(val_path, val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'renotate_webvision_reduce':

            re_val_path, re_val_label = reduce_get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(re_val_path, re_val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'webvision_google_half':
            val_path, val_label = get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(val_path, val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'imagenet':
            imagenet_validation = imagenet_val(transform=self.trans.trans_val())

            return DataLoader(imagenet_validation, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'reduce_imagenet':
            imagenet_validation = reduce_imagenet_val(transform=self.trans.trans_val())

            return DataLoader(imagenet_validation, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'scan_top5':
            val_path, val_label = get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(val_path, val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)
        elif self.name == 'mini_webvision':
            val_path, val_label = get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(val_path, val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)
        elif self.name == 'mini_scan_top5':
            val_path, val_label = get_path(scv_file_path='path/val', file_num=5)

            webvision_val = Webvision(val_path, val_label, transform=self.trans.trans_val())

            return DataLoader(webvision_val, batch_size=self.batch_size, shuffle=False, num_workers=10)

        elif self.name == 'cifar_10':
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            cifar_train = datasets.CIFAR10('dataset', train=False, transform=trans, download=True)

            return DataLoader(cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'cifar_10_n02':
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            cifar_train = datasets.CIFAR10('dataset', train=False, transform=trans, download=True)

            return DataLoader(cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        elif self.name == 'cifar_10_n06':
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            cifar_train = datasets.CIFAR10('dataset', train=False, transform=trans, download=True)

            return DataLoader(cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=10)

        else:
            raise ValueError('Invalid dataset name')

def number_each_class(dataset):
    if dataset == 'webvision':
        classes = [0 for x in range(1000)]
        train_path, train_label = get_path(scv_file_path='path/train', file_num=120)
        for label in train_label:
            classes[label] += 1
        classes = np.asarray(classes)
        classes = torch.from_numpy(classes).type(torch.cuda.FloatTensor)
        return classes

    elif dataset == 'scan_top5':
        label_acc5 = []
        with open('final_top5.csv', 'rb') as f:
            reader = csv.reader(f)
            for path, label in reader:
                label_acc5.append(int(label))
        classes = [0 for x in range(1000)]

        for label in label_acc5:
            classes[label] += 1
        classes = np.asarray(classes)
        classes = torch.from_numpy(classes).type(torch.cuda.FloatTensor)
        return classes
    elif dataset == 'imagenet':
        pass

    else:
        raise ValueError('Invalid dataset name')















