import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
f = open('/home/jiangfanhan/data/ILSVRC2012_validation_ground_truth.txt', 'r')
tr_f = open('/home/jiangfanhan/data/transfer.txt', 'r')
img_path = '/home/jiangfanhan/data/imagenet_val_pic'
fake_labels = []
pic_path = []
trans_arr = []
true_labels = []
ignore_list = [80, 134, 152, 156, 158, 167, 186, 188, 193, 206, 209,
               214, 215, 220, 221, 238, 240, 255, 282, 332, 389, 392, 446, 465, 482, 522, 530,
               542, 543, 585, 590, 639, 729, 744, 790, 804, 810, 818, 826, 833, 848, 885, 901]
ignore_list.reverse()
origin = [x for x in range(1000)]
for i in ignore_list:
    origin.pop(i)


for line in f.readlines():
    fake_labels.append(int(line))

for line in tr_f.readlines():
    trans_arr.append(int(line))
trans_dict = {}
for i, x in enumerate(trans_arr):
    trans_dict[x] = i

for labels in fake_labels:
    true_labels.append(trans_dict[labels])


for dirpath, dirnames, filenames in os.walk(img_path):
    for name in filenames:
        path = os.path.join(dirpath, name)
        pic_path.append(path)
pic_path.sort()

ziped = zip(pic_path,true_labels)

k = 0
j = 0
for i in range(50000):
    if ziped[i-k][1] in ignore_list:
        ziped.pop(i - k)
        k += 1
re_pic_path, re_true_labels = zip(*ziped)
re_pic_path = list(re_pic_path)
re_true_labels = list(re_true_labels)

for i, label in enumerate(re_true_labels):
    re_true_labels[i] = origin.index(label)




class imagenet_val(Dataset):
    def __init__(self, path=pic_path, label=true_labels, transform=None):

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
        label = self.label[index]
        sample = (image,label)
        return sample

class reduce_imagenet_val(Dataset):
    def __init__(self, path=re_pic_path, label=re_true_labels, transform=None):
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
        label = self.label[index]
        sample = (image, label)
        return sample