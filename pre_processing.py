import torch
from torchvision import transforms

class preprocessing(object):
    def __init__(self, network):

        self.network = network

    def trans_train(self):

        if self.network =='inception_resnetv2':

            return transforms.Compose([transforms.Scale(342),
                                       transforms.RandomCrop(299),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                       ])
        else:

            return transforms.Compose([transforms.Scale(256),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                       ])
    def trans_val(self):

        if self.network == 'inception_resnetv2':
            return transforms.Compose([transforms.Scale(342),
                                        transforms.CenterCrop(299),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])

        else:

            return transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
