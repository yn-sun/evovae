# coding=utf-8

import os
import torch as th 
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import *
from torchvision import transforms
import numpy as np 
import random
from utils import _args

_DATASET_PATH = os.path.expanduser('~/dataset')

def set_all_seed(rand_seed):
    # set the random seed
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    th.manual_seed(rand_seed)
    th.random.manual_seed(rand_seed)
    th.cuda.manual_seed(rand_seed)
    th.cuda.manual_seed_all(rand_seed)
    th.backends.cudnn.deterministic=True

def split_dataset_by_cls_num(d1, d2, labels, num_per_cls):
    """Split the dataset by the specified number of data per class
    Params:
    -------
    - d1            (th.utils.data.Dataset) :
    - d2            (th.utils.data.Dataset) :
    - labels        (list)                  : list of label, from `0`
    - num_per_cls   (int)                   : 
    Returns:
    --------
    - subset1       (th.utils.data.Subset)  : contains the dataset whose number per class is less or equal to `num_per_cls`
    - subset2       (th.utils.data.Subset)  : contains the data which is exclusive of `subset1`
    """
    cls_list = np.unique(labels)
    cls_i_idx_split1_list = []
    cls_i_idx_split2_list = []
    for cls_i in cls_list:
        if cls_i >=0:
            cls_i_idx = np.random.permutation((np.array(labels)==cls_i).nonzero()[0])
            cls_i_idx_split1 = cls_i_idx[:num_per_cls]
            cls_i_idx_split2 = cls_i_idx[num_per_cls:]

            cls_i_idx_split1_list += cls_i_idx_split1.tolist()
            cls_i_idx_split2_list += cls_i_idx_split2.tolist()
    # import pdb; pdb.set_trace()
    subset1 = Subset(d1, cls_i_idx_split1_list)           
    # subset1 = Subset(d1, list(range(len(d1))))           
    subset2 = Subset(d2, cls_i_idx_split2_list)
    return subset1, subset2

def split_dataset(d1, d2, train_val_ratio, shuffle=True):
    """Split the dataset by the specified number of data per class
    Params:
    -------
    - d1                (th.utils.data.Dataset) :
    - d2                (th.utils.data.Dataset) :
    - train_val_ratio   (float)                 : ratio between training samples number and validation samples number
    Returns:
    --------
    - subset1       (th.utils.data.Subset)  : contains the `1 - valida_ratio` data in `d1`
    - subset2       (th.utils.data.Subset)  : contains the data which is exclusive of `subset1`
    """
    assert len(d1) == len(d2), 'the two dataset must be consistent'
    num_data = len(d1)
    
    train_num = int(train_val_ratio/(train_val_ratio+1)*num_data)
    # valid_size = int(num_data * valid_ratio)
    all_idx = np.arange(num_data)
    
    if shuffle:
        all_idx = np.random.permutation(all_idx)
    # idx_train = all_idx[valid_size:]
    # idx_val = all_idx[:valid_size]
    idx_train = all_idx[:train_num]
    idx_val = all_idx[train_num:]

    subset1 = Subset(d1, idx_train)           
    subset2 = Subset(d2, idx_val)

    return subset1, subset2

def build_dataset(dataset_name, split_val=False, train_val_ratio=9, num_per_cls=None):
    """Build different datasets
    Params:
    -------
    - dataset_name (str)
    - num_per_cls  (int)
    """
    unsup_train=unsup_val=\
           unsup_test=super_train=\
           super_val=super_test=None

    # load the transformation
    # NOTE: if you want to add data augmentation, please reimplement this function...
    train_trans = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 2*x-1
            ])

    infer_trans = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 2*x-1
            ]) 

    # set the seed
    set_all_seed(_args.rand_seed)
    if dataset_name == 'CIFAR10':
        unsup_train = CIFAR10(_DATASET_PATH, train=True, transform= train_trans)

        d1 = CIFAR10(_DATASET_PATH, train=True, transform=train_trans)
        d2 = CIFAR10(_DATASET_PATH, train=True, transform=infer_trans)

        if num_per_cls is None:
            if split_val:
                super_train, super_val = split_dataset(d1, d2, train_val_ratio)
            else:
                super_train = d1
        else:
            super_train, super_val = split_dataset_by_cls_num(d1, d2, d1.targets, num_per_cls)

        super_test = CIFAR10(_DATASET_PATH, train=False, transform=infer_trans)

    elif dataset_name == 'Caltech101':
        img_size=224
        train_trans = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            lambda x: 2*x-1
            ])

        infer_trans = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            lambda x: 2*x-1
            ])
        
        d1 = Caltech101(_DATASET_PATH, transform=train_trans)
        d2 = Caltech101(_DATASET_PATH, transform=infer_trans)
        super_train, super_test = split_dataset_by_cls_num(d1, d2, d1.y, 30)
        unsup_train = super_train

    elif dataset_name == 'MNIST':
        unsup_train = MNIST(_DATASET_PATH, train=True, transform= train_trans)

        d1 = MNIST(_DATASET_PATH, train=True, transform=train_trans)
        d2 = MNIST(_DATASET_PATH, train=True, transform=infer_trans)
        
        if num_per_cls is None:
            if split_val:
                super_train, super_val = split_dataset(d1,d2,train_val_ratio)
            else:
                super_train = d1 
        else:
            super_train, super_val = split_dataset_by_cls_num(d1, d2, d1.targets, num_per_cls)
        
        super_test = MNIST(_DATASET_PATH,train=False,transform=infer_trans)

    elif dataset_name == 'STL10':
        unsup_train = STL10(_DATASET_PATH, split='train+unlabeled', transform=train_trans)

        d1 = STL10(_DATASET_PATH, split='train', transform=train_trans)
        d2 = STL10(_DATASET_PATH, split='train', transform=infer_trans)
        
        if num_per_cls is None:
            if split_val:
                super_train, super_val = split_dataset(d1,d2,train_val_ratio)
            else:
                super_train = d1 
        else:
            super_train, super_val = split_dataset_by_cls_num(d1, d2, d1.labels, num_per_cls)
        
        super_test = STL10(_DATASET_PATH, split='test', transform=infer_trans)
    
    elif dataset_name == 'SVHN':
        unsup_train = SVHN(_DATASET_PATH, split='train', transform= train_trans)

        d1 = SVHN(_DATASET_PATH, split='train', transform=train_trans)
        d2 = SVHN(_DATASET_PATH, split='train', transform=infer_trans)

        if num_per_cls is None:
            if split_val:
                super_train, super_val = split_dataset(d1, d2, train_val_ratio)
            else:
                super_train = d1
        else:
            super_train, super_val = split_dataset_by_cls_num(d1, d2, d1.labels, num_per_cls)

        super_test = SVHN(_DATASET_PATH, split='test', transform=infer_trans)
        
    else:
        raise ValueError('Unsupported dataset!')
    
    # set the seed
    set_all_seed(_args.rand_seed) # keep the random split to being the same

    return unsup_train, unsup_val,\
           unsup_test, super_train,\
           super_val, super_test



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    d1,_,_,sup_train,sup_val,_=build_dataset('CIFAR10',split_val=True) 
    im=sup_train.__getitem__(63)[0]
    im=(im.permute(1,2,0)*.5+.5).numpy()
    if im.shape[2]==1:
        im=im[...,0]
    plt.imshow(im)
    plt.show()
    print(sup_train.__len__())
    print(sup_val.__len__())