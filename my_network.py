import torch
from torchvision import models
from inceptionresnetv2 import inceptionresnetv2
import my_resnet as myr
import resnet_for_cifar as res_cifar

def load_pretrained_model(network):
    if network == 'inception_resnetv2':
        net = inceptionresnetv2(num_classes=1000)
    elif network == 'resnet_50':
        net = models.resnet50(pretrained=True)
    elif network == 'resnet_18':
        net = models.resnet18(pretrained=True)
    elif network == 'alexnet':
        net = models.alexnet(pretrained=True)
    elif network == 'my_res_50':
        net = myr.resnet50(pretrained=True)
    else:
        raise ValueError ('invalid network name')
    return net

def load_scratch_model(network):
    if network == 'inception_resnetv2':
        net = inceptionresnetv2(num_classes=1000, pretrained=None)
    elif network == 'resnet_50':
        net = models.resnet50(pretrained=False)
    elif network == 'resnet_18':
        net = models.resnet18(pretrained=False)
    elif network == 'alexnet':
        net = models.alexnet(pretrained=False)
    elif network == 'my_res_50':
        net = myr.resnet50(pretrained=False)
    elif network == 'resnet_14_cifar_10':
        net = res_cifar.resnet14()
    elif network == 'resnet_32_cifar_10':
        net = res_cifar.resnet32()
    else:
        raise ValueError('invalid network name')
    return net