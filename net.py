# coding=utf-8
"""
@File    :   net.py,
@Time    :   2019-08-4,
@Author  :   Chen Xiangru,
@Version :   1.0,
@Contact :   None,
@License :   (C)Copyright None,
@Desc    :   about the networks 
"""

import os,sys,time
import numpy as np
import torch
import torch as th
import torchvision
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from torchvision.datasets import MNIST as DATASET
# from torchvision.datasets import CIFAR10 as DATASET
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10 
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from dataloader import build_dataset
from utils import get_logger,get_type_idx,_args,indi_copy,\
                PROC_HIDE_LOGGER
logger = get_logger(__name__)

# if not os.path.exists('./mlp_img'):
    # os.mkdir('./mlp_img')

def sample_model(model,latent_dim,path):
    model.eval()
    param_type = list(model.parameters())[0].device
    with th.no_grad():
        # t=th.nn.functional.softmax(th.randn(64,10),dim=1)
    #     t=th.zeros(64,10)
        z=th.randn(64,latent_dim).to(param_type)
        # zy=th.cat([z,t],dim=1).cuda(cu_did)
        # rec=model.decoder(zy)
        rec=model.decoder(z)
        rec=(rec+1)/2
    plt.figure()
    plt.imshow(make_grid(rec,nrow=8).cpu().detach().numpy().transpose(1,2,0))
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(path)

TRAIN_DATASET_MAP={
    'MNIST': torchvision.datasets.MNIST,
    'CIFAR10': torchvision.datasets.CIFAR10,
    'CIFAR100': torchvision.datasets.CIFAR100,
    'STL10': torchvision.datasets.STL10
}

TIME_DELAY_FOR_SHARE=1

# test the assymmetric padding
class A(th.nn.Module): 
    """
    Test the assymmetric padding for convolutional layer
    Params
    ------
    - fs (int): feature map size
    - ic (int): number of input channels
    - k  (int): kernel size
    - s  (int): stride size 
    """
    def __init__(self,fs,ic,oc,k,s): 
        super(A,self).__init__() 
        pad=(fs-1)*s-fs+k 
        pad,pad_=(pad//2,pad//2+pad%2) 
        pad=(pad,pad_,pad,pad_) 
        self.pad=pad 
        self.conv=th.nn.Conv2d(ic,oc,k,s) 

    def forward(self,x): 
        x=th.nn.functional.pad(x,pad=self.pad) 
        x=self.conv(x) 
        return x

class Identity(th.nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x

class Flatten(th.nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
        
    def forward(self,x):
        return x.view(x.size(0),-1)

class Unflatten(th.nn.Module):
    def __init__(self,c,h,w):
        super(Unflatten,self).__init__()
        self.h=h
        self.w=w
        self.c=c
        
    def forward(self,x):
        return x.view(x.size(0),self.c,self.h,self.w)

class DeConv2d(th.nn.Module):
    def __init__(self, ic, oc, act=None):
        super(DeConv2d,self).__init__()
        self.deconv=th.nn.ConvTranspose2d(ic,oc,2,2)
        self.act = act
    
    def forward(self,x):
        x=self.deconv(x)
        if self.act:
            x = self.act(x)

        return x 

class SameConv2d(th.nn.Module):
    """
    Conv2d wrapper with `same` padding.
    NOTE: fixed stride=1
    Params
    ------
    - ic  (int): input channles
    - oc  (int): output channles
    - ks  (int): kernel size
    """
    def __init__(self,ic,oc,ks,bn=True,act=None):
        super(SameConv2d,self).__init__()
        
        padding=(ks-1)//2
        padding_=(ks-1) % 2
        self.pad=(padding,padding+padding_,padding,padding+padding_)
        if bn:
            self.conv=th.nn.Conv2d(ic,oc,ks,1,bias=False)
            self.bn=th.nn.BatchNorm2d(oc)
        else:
            self.conv=th.nn.Conv2d(ic,oc,ks,1)
            self.bn=None 
        
        self.act=act

    def forward(self,x):
        x=th.nn.functional.pad(x,pad=self.pad)
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        # x=th.nn.functional.relu(x,inplace=True)
        if self.act:
            x = self.act(x)
               
        return x

class Dense(th.nn.Module):
    def __init__(self,inf,outf,act=None):
        super(Dense,self).__init__()
        self.linear=th.nn.Linear(inf,outf)
        self.act = act

    def forward(self,x):
        x=self.linear(x)
        if self.act:
            x=self.act(x)

        return x

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 28, 28)
    return x


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        self.encoder_mean = None
        
        self.encoder_var = None
        
        self.decoder = None

        self.fn = None

    def forward(self, x, supervised=False):
        x_mean = self.encoder_mean(x)
        if not supervised:
            x_var = self.encoder_var(x)
            epsi = th.randn(x_var.shape)
            # if x_var.is_cuda:
            #     epsi=epsi.cuda(x_var.device)
            epsi=epsi.to(x_var.device)
            
            x=x_mean+epsi*(x_var/2).exp()
            x = self.decoder(x)
            return x,x_mean,x_var
        else:
            x = self.fn(x_mean.view(len(x_mean),-1))
            return x

    def normal_ae_forward(self,x):
        """Forward like general vae
        """
        if hasattr(self,'share_head'):
            if self.share_head is not None:
                x = self.share_head(x)                
        latent_rep = self.encoder_mean(x)
        x_rec = self.decoder(latent_rep)

        return x_rec,latent_rep

    def extra_features_and_labels(self, dataloader, without_sigma=False):
        self.eval()
        features_list=[]
        label_list=[]
        did = list(self.parameters())[0].device
        
        for img,y  in dataloader:
            # img = img.view(img.size(0), -1)
            img = img.to(did)
            # ===================forward=====================
            x=img
            if hasattr(self,'share_head'):
                if self.share_head is not None:
                    x = self.share_head(x)                
            x_mean = self.encoder_mean(x)
            
            features = x_mean
            if not without_sigma:
                x_var = self.encoder_var(x)
                epsi = th.randn(x_var.shape)
                epsi=epsi.to(x_var.device)
                x=x_mean+epsi*(x_var/2).exp()
                features = x 
            features_list += [features.cpu().detach().numpy()]
            label_list += [y.numpy()]

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        return features,labels


    def unsup_parameters(self):
        params=[]
        for k,v in self.named_parameters():
            for t in ['shared_head','encoder_mean','encoder_var','decoder','gamma_log']:
                if k.startswith(t):
                    params+=[v]
        return  params
            
    def sup_parameters(self):
        return  th.nn.Sequential(*[
            _ for _ in [
                self.share_head,
                self.encoder_mean,
                self.fn] if _
        ]).parameters()

class CIFAR10VAE(BaseVAE):
    def __init__(self):
        super(CIFAR10VAE, self).__init__()
        self.encoder_mean = nn.Sequential(
            nn.Linear(3*32 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        
        self.encoder_var = nn.Sequential(
            nn.Linear(3*32 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3)) 
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 3*32 * 32), nn.Tanh())

        self.fn = th.nn.Sequential(th.nn.Linear(3,10),th.nn.Softmax(dim=1))


# class VAELoss(th.nn.Module):
#     def __init__(self):
#         super(VAELoss,self).__init__()
        
#     def forward(self,x,y,x_mean,x_var):
#         n=x.shape[0]
#         rec_loss=((x-y)**2).sum()
# #         kl_loss= (-(x_var**2+1e-6).log()+x_mean**2+x_var**2-1).sum()
#         kl_loss=(-x_var+x_mean**2+x_var.exp()-1).sum()
#         loss=rec_loss/(2*n)+kl_loss/(2*n)
#         return loss

class AELoss(th.nn.Module):
    def __init__(self):
        super(AELoss,self).__init__()
        
    def forward(self,x,y,x_mean,x_var):
        n=x.shape[0]
        rec_loss=((x-y)**2).sum()
#         kl_loss= (-(x_var**2+1e-6).log()+x_mean**2+x_var**2-1).sum()
        reconstruction_loss=rec_loss/(2*n)
        return reconstruction_loss

class VAELoss(th.nn.Module):
    def __init__(self):
        super(VAELoss,self).__init__()
        
    def forward(self,x,y,x_mean,x_var):
        n=x.shape[0]
        rec_loss=((x-y)**2).sum()
#         kl_loss= (-(x_var**2+1e-6).log()+x_mean**2+x_var**2-1).sum()
        kl_loss=(-x_var+x_mean**2+x_var.exp()-1).sum()
        loss=rec_loss/(2*n)+kl_loss/(2*n)
        reconstruction_loss=rec_loss/(2*n)
        kl_divergence=kl_loss/(2*n)
        return loss,reconstruction_loss,kl_divergence

class EvolveVAE(BaseVAE):
    """
    Evolve Variational autoencoder
    Params
    ------
    - indi          (dict): individuals
    - ic            (int) : input channles
    - im_size       (int) : im_size
    - out_class_num (int) : number of output class
    - msg_buf       (obj) :
    Returns
    -------
    """
    def __init__(self,indi,ic,im_size,out_class_num,msg_buf = None):
        super(EvolveVAE,self).__init__()
        
        indi = indi_copy(indi)
        self.indi = indi
        self.ic = ic
        self.im_size= im_size
        self.out_class_num = out_class_num
        self.msg_buf = msg_buf

        self.share_head=None
        self.encoder_mean=None
        self.encoder_var=None
        self.decoder=None

        #####################   
        dd_dd_idx_list=get_type_idx(indi['dd'],'d')
        num_of_dd = len(dd_dd_idx_list)
        

        mu_conv_idx_list = get_type_idx(indi['mu'],'c')
        head_conv_indx_list = get_type_idx(indi['head'],'c')
        if len(mu_conv_idx_list)==0:
            fc=indi['head'][head_conv_indx_list[-1]]['gene'][0]
        else:
            fc = indi['mu'][mu_conv_idx_list[-1]]['gene'][0]

        fh=fw=im_size//2**num_of_dd
        f_num=fc*fh**2

        dd_f_idx_list=get_type_idx(indi['dd'],'f')
        # trick: add the channel after the last fc in block of dd..
        indi['dd'].insert(dd_f_idx_list[-1]+1,{'type':'f','gene':[f_num]})
        # trick: add unflatten
        indi['dd'].insert(dd_f_idx_list[-1]+2,{'type':'u','gene':[fc,fh,fw]}) 
        # trick: convert 'latent' to 'latent1' and 'latent2'
        latent_dim = indi['latent'][0]['gene'][0]
        self.latent_dim=latent_dim
        del indi['latent']
        indi['latent1']=[{'type':'f','gene':[latent_dim]}]
        indi['latent2']=[{'type':'f','gene':[latent_dim]}]
        indi['dd']+=[{'type':'c','gene':[ic,3]}] # keep channel same

        lOut={k:[None]*len(v) for k,v in indi.items()}
        block_dict={k:[None]*len(v) for k,v in indi.items()}
        bInShapeMap = {
            'head': lambda : (im_size,im_size,ic),
            'mu': lambda : lOut['head'][-1] if len(indi['head'])  else (im_size,im_size,ic),
            'sig': lambda : lOut['head'][-1] if len(indi['head']) else (im_size,im_size,ic),
            'latent1': lambda: lOut['mu'][-1],
            'latent2': lambda: lOut['sig'][-1],
            'dd': lambda : lOut['latent1'][0],
        }
        bOutActMap = {
            'head': lambda idx: th.nn.ReLU(),
            'mu': lambda idx: th.nn.ReLU(),
            'sig': lambda idx: th.nn.ReLU(),
            'latent1': lambda idx: Identity(),
            'latent2': lambda idx: Identity(),
            'dd': lambda idx: th.nn.Tanh() if idx == len(lOut['dd'])-1 else th.nn.ReLU() 
        }
        outShapeMap={
            'c': lambda inS, gene: (inS[0],inS[1], gene[0]), 
            'p': lambda inS, gene: (inS[0]//2,inS[1]//2, inS[2]),
            'd': lambda inS, gene: (inS[0]*2, inS[1]*2, gene[0]),
            'f': lambda inS, gene: (gene[0],),
            'u': lambda inS, gene: (gene[1], gene[2], gene[0])
        }
        layerCreateMap = {
            'c': lambda inS,gene,act: SameConv2d(inS[2], gene[0], gene[1], bn=True, act=act),
            'p': lambda inS,gene,act: th.nn.Sequential(th.nn.AvgPool2d(2,2),th.nn.ReLU()) if gene[0] else th.nn.MaxPool2d(2,2),
            'd': lambda inS,gene,act: DeConv2d(inS[2], gene[0], act=act),
            'f': lambda inS,gene,act: Dense(inS[0],gene[0],act=act) if len(inS)==1 else th.nn.Sequential(Flatten(),Dense(np.prod(inS),gene[0],act=act)),
            'u': lambda inS,gene,act: Unflatten(*gene),
        }
        
        for bname in ['head','mu','sig','latent1','latent2','dd']:
            block = indi[bname]
            for i,unit in enumerate(block):
                uType = unit['type']
                uGene = unit['gene']
                if i==0:
                    inS = bInShapeMap[bname]()

                else:
                    inS = lOut[bname][i-1] 
                
                lOut[bname][i]=outShapeMap[uType](inS, uGene)

                act = bOutActMap[bname](i)
                # if inS is None:
                #     print('-------------------------')
                #     print(bname,'\n','block=',block,'\n','lOut',lOut,'\n',i)
                #     print('-------------------------')
                layer=layerCreateMap[uType](inS, uGene, act)
                block_dict[bname][i]=layer
        # combine
        self.share_head = th.nn.Sequential(*block_dict['head'])
        self.encoder_mean = th.nn.Sequential(*block_dict['mu'],*block_dict['latent1'])
        self.encoder_var = th.nn.Sequential(*block_dict['sig'],*block_dict['latent2'])
        self.decoder = th.nn.Sequential(*block_dict['dd'])

        self.fn=th.nn.Sequential(*[
            Dense(latent_dim,512,act=th.nn.ReLU()),
            th.nn.Dropout(.5),
            Dense(512,self.out_class_num,act=th.nn.Softmax(dim=1))
        ])
        

    @DeprecationWarning
    def probe_mean_shape(self):
        x=th.randn(2,self.ic,self.im_size,self.im_size)

        if self.share_head is not None:
            x=self.share_head(x)
        x=self.encoder_mean(x)
        _,feat_num=x.view(len(x),-1).shape
        return feat_num
        
    def forward(self,x,supervised=False):
        if self.share_head is not None:
            x=self.share_head(x)
        
        return BaseVAE.forward(self,x,supervised)

    @DeprecationWarning
    def create_layer(self,unit,last_unit):
        utype = unit['type']
        layer = None
        if utype == 'c':
            oc,ks=unit['gene']
            if last_unit is None:
                layer=CONV2D(self.ic,oc,ks)
            else:
                ic=last_unit['gene'][0]
                layer=CONV2D(ic,oc,ks)

        elif utype == 'p':
            ptype,=unit['gene']
            if ptype == 0:
                layer = th.nn.MaxPool2d(2,2)
            else:
                layer = th.nn.AvgPool2d(2,2)
        elif utype == 'd':
            ic= last_unit['gene'][0]
            oc,=unit['gene']
            layer= th.nn.ConvTranspose2d(ic,oc,2,2)
        elif utype == 'f':
            self._error('fully connected layer is not supported for now.')
        return layer 

    def _error(self, msg_str):
        if self.msg_buf is not None:
            self.msg_buf.print('ERROR: '+msg_str)
            time.sleep(TIME_DELAY_FOR_SHARE)
        if not PROC_HIDE_LOGGER:
            logger.error(msg_str)

    

class RunModel(object):
    """
    Running model for evolving
    Params
    ------
    - id            (int)               : identifier
    - indi          (list)              : the individual
    - did           (int)               : cuda device id
    - msg_buf       (ProcShareMessage)  :
    - DATASET       (dataset)           : torch dataset class 
    - OUT_CLASS_NUM (int)               : number of the output classes
    Returns
    -------
    """
    def __init__(self,id,indi,did,
                iters,generations,
                DATASET,
                OUT_CLASS_NUM=10,sup_train_epochs=None,unsup_train_epochs=None,
                num_per_cls=None,
                specified_model=None,
                msg_buf=None,
                # DATASET=TRAIN_DATASET_MAP[_args.dataset],
                ):
        self.id=id 
        self.indi=indi
        self.did=did 
        self.msg_buf=msg_buf
        self.ga_iters=iters
        self.ga_generations=generations
        # img_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     lambda x: 2*x-1
        #     ])

        self.TEST_DECODE=_args.net_decode_test
        self.only_cpu=_args.only_cpu   
        self.unsup_train_epochs = _args.unsupervised_train_epoch if unsup_train_epochs is None else unsup_train_epochs
        self.sup_train_epochs = _args.supervised_train_epoch if sup_train_epochs is None else sup_train_epochs
        
        self.batch_size = _args.batch_size
        self.learning_rate = _args.learning_rate

        self.lr_sche=_args.lr_schedule
        
        # train_val_ratio=_args.train_val_ratio
        # train_dataset=DATASET(os.path.expanduser('~/dataset'), train=True, 
        #     transform=img_transform,download=False)
        # total_trainset_num = train_dataset.__len__()
        # train_num = int(train_val_ratio/(train_val_ratio+1)*total_trainset_num)
        # valid_num = total_trainset_num - train_num
        
        # torch.manual_seed(_args.rand_seed) # keep the random split to being the same
        # train_dataset,valid_dataset= random_split(train_dataset,[train_num,valid_num])
        # torch.manual_seed(_args.rand_seed) 
        
        # self.train_dataloader = DataLoader(train_dataset, 
        #     batch_size=self.batch_size, shuffle=True)
        
        # self.valid_dataloader = DataLoader(valid_dataset,
        #     batch_size=self.batch_size, shuffle=False)
        
        # self.test_dataloader = DataLoader(DATASET(os.path.expanduser('~/dataset'), train=False,
        #     transform=img_transform,download=False), 
        #     batch_size=self.batch_size, shuffle=False)
        
        unsup_train_ds, unsup_val_ds, unsup_test_ds, sup_train_ds, sup_val_ds, sup_test_ds  \
            = build_dataset(DATASET, split_val=False, train_val_ratio=_args.train_val_ratio, num_per_cls=num_per_cls)
        
        self.unsup_train_dataloader = DataLoader(
            unsup_train_ds, 
            batch_size=self.batch_size, shuffle=True,
            num_workers=_args.num_workers)
        
        self.sup_train_dataloader = DataLoader(
            sup_train_ds,
            batch_size=self.batch_size, shuffle=True,
            num_workers=_args.num_workers)
        
        self.sup_test_dataloader = DataLoader(
            sup_test_ds, 
            batch_size=self.batch_size, shuffle=False,
            num_workers=_args.num_workers)

        _,c,h,w=next(iter(self.unsup_train_dataloader))[0].shape
        assert h==w
        self.model_input_shape = (h,w,c)
        model=EvolveVAE(indi,c,h,OUT_CLASS_NUM) if specified_model is None else specified_model
        if not self.only_cpu:
            model =model.cuda(self.did)
        
        self.model=model 
        self.unsupervised_criterion = VAELoss()
        self.supervised_criterion = lambda x,y: -(x[range(len(x)),y]+1e-6).log().mean() 
        
        unsup_opt_Adam = torch.optim.Adam(
            model.unsup_parameters(), lr=_args.learning_rate, weight_decay=1e-5)
        sup_opt_Adam = torch.optim.Adam(
            model.sup_parameters(), lr=_args.learning_rate, weight_decay=1e-5)

        # sup_opt_Adam = torch.optim.SGD(
        #     model.sup_parameters(), lr=_args.learning_rate, weight_decay=1e-5)

        self.unsupervised_opt = unsup_opt_Adam
        self.supervised_opt = sup_opt_Adam
        self.optimizer=None
        

        ##########for best unsupervised##############
        self.unsup_loss_w={'loss':float('inf'),w:None}
        ########################
        self._info('training model initialization ok! (net-id: {})'.format(self.id)) 
    
    def get_inf_time(self, input_shape = None, supervised=True):
        if input_shape is None:
            input_shape = (self.model_input_shape[-1],*self.model_input_shape[:-1])
        
        assert input_shape is not None 
        input_shape = (1,*input_shape)
        self.model.eval()
        with th.no_grad():
            # macs and params
            model_dev = next(self.model.parameters()).device

            input_data = th.randn(*input_shape).to(model_dev)

            # inference time
            iter_times = 10000
            timeall = []
            for i in range(iter_times):
                if i <= 100: 
                    # remove the influence of the first time
                    continue
                t0 = time.time()

                self.model(input_data, supervised)
                
                t1=time.time()

                timeall += [(t1-t0)*1000]
            timeall = np.array(timeall)
            delay_ms_mean, delay_ms_std = timeall.mean(), timeall.std()
        return delay_ms_mean, delay_ms_std
        
    def _info(self, msg_str):
        if self.msg_buf is not None:
            self.msg_buf.print(msg_str)
            time.sleep(TIME_DELAY_FOR_SHARE)

        if not PROC_HIDE_LOGGER:
            logger.info(msg_str)
        
    def _error(self, msg_str):
        if self.msg_buf is not None:
            self.msg_buf.print('ERROR: '+msg_str)
            time.sleep(TIME_DELAY_FOR_SHARE)

        if not PROC_HIDE_LOGGER:
            logger.error(msg_str)

    def _print(self, msg_str):
        if self.msg_buf is not None:
            self.msg_buf.print(msg_str)
            time.sleep(TIME_DELAY_FOR_SHARE)

        if not PROC_HIDE_LOGGER:    
            logger.info(msg_str)
    
    def get_lr_schedule(self,maxT):

        if self.lr_sche=='':
            return None 
        if self.lr_sche=='cosine':
            return th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,maxT)

        return None 

    def train_unsupervised(self):
        self.optimizer=self.unsupervised_opt
        lr_scheduler=self.get_lr_schedule(self.unsup_train_epochs)
        
        iters=1
        
        max_ll = float('-inf')
        for epoch in range(self.unsup_train_epochs):
            self.model.train()
            loss_list=[]
            rec_loss_list=[]
            kl_loss_list=[]

            for data in self.unsup_train_dataloader:
                img, _ = data
                # img = img.view(img.size(0), -1)
                if not self.only_cpu:
                    img = img.cuda(self.did)
                # ===================forward=====================
                output,x_mean,x_var = self.model(img)
                loss,rec_loss,kl_loss = self.unsupervised_criterion(img,output,x_mean,x_var)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # ===================log========================
                # logger.info('net-id: {}, epoch [{}/{}], iter {}, loss:{:.4f}'
                #    .format(self.id,epoch + 1, self.unsup_train_epochs, iters, loss.item() ))

                loss_list+=[loss.item()]
                rec_loss_list+=[rec_loss.item()]
                kl_loss_list+=[kl_loss.item()]

                iters+=1
                if self.TEST_DECODE:
                    return

            cur_lr=self.optimizer.param_groups[0]['lr']
            if lr_scheduler:
                lr_scheduler.step()
            # ===================log========================
            # self._info('**unsupervised training -- net-id: {}, epoch [{}/{}], loss:{:.4f}, lr:{:.6f}'
                    # .format(self.id,epoch + 1, self.unsup_train_epochs, loss.item(),cur_lr))

            self._info('**unsupervised training -- net-id: {}, epoch [{}/{}], mean-loss:{:.4f}, mean-rec_loss:{:.4f}, mean-kl_loss:{:.4f}, lr:{:.6f}'
                    .format(self.id,epoch + 1, self.unsup_train_epochs, np.mean(loss_list),np.mean(rec_loss_list),np.mean(kl_loss_list),cur_lr))
            
            if self.unsup_loss_w['loss']> np.mean(loss_list):
                self.unsup_loss_w['loss'] = np.mean(loss_list)
                self.unsup_loss_w['w']=self.model.state_dict()

            max_ll = max(max_ll,-np.mean(loss_list))
            if epoch % 10 == 0:
                pass 
                # pic = to_img(output.cpu().data)
                # save_image(pic, './mlp_img/image_{}.png'.format(epoch))
        
        im_save_path = os.path.join('sample_img', time.strftime("%Y-%m-%d-%X") + '-net-id-' + str(self.id)+ '.png')
        sample_model(self.model,self.model.latent_dim,path=im_save_path)
        if self.msg_buf is not None:
            self.msg_buf.save_file(im_save_path)
            time.sleep(10) # must wait for the message sync.
            
        fitness = max_ll
        return fitness

    def train_supervised(self):
        self.optimizer=self.supervised_opt
        lr_scheduler=self.get_lr_schedule(self.sup_train_epochs)
        best_acc = float('-inf')
        for epoch in range(self.sup_train_epochs):
            self.model.train()
            
            n=0
            tp=0
            for img,y  in self.sup_train_dataloader:
                # img = img.view(img.size(0), -1)
                if not self.only_cpu:
                    img = img.cuda(self.did)
                    y = y.cuda(self.did)
                # ===================forward=====================
                x = self.model(img,True)
                py=x.max(dim=1)[1]
                tp+=(py==y).sum().item()
                n+=len(x)
                loss = self.supervised_criterion(x,y)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            
                if self.TEST_DECODE:
                    return float(tp)/n
            
            cur_lr=self.optimizer.param_groups[0]['lr']
            if lr_scheduler:
                lr_scheduler.step()

            train_acc=float(tp)/n
            # valid_acc=self.test(self.valid_dataloader)
            # best_acc = max(best_acc, valid_acc)
            # test_acc=self.test(self.test_dataloader)
            test_acc = self.test(self.sup_test_dataloader)
            best_acc = max(best_acc, test_acc)
            # ===================log========================
            # self._info('**supervised training -- net-id: {}, epoch [{}/{}], loss:{:.4f}, train acc:{:.2f}%, valid acc:{:.2f}%, test acc:{:.2f}%, lr:{:.6f}'
            #         .format(self.id, epoch + 1, self.sup_train_epochs, loss.item(), train_acc*100, valid_acc*100,test_acc*100,cur_lr))
            self._info('**supervised training -- net-id: {}, epoch [{}/{}], loss:{:.4f}, train acc:{:.2f}%, test acc:{:.2f}%, lr:{:.6f}'
                    .format(self.id, epoch + 1, self.sup_train_epochs, loss.item(), train_acc*100,test_acc*100,cur_lr))

        # return valid_acc 
        return best_acc

    def test(self,data_loader):
        self.model.eval()
        
        n=0
        tp=0
        with th.no_grad():
            for img,y  in data_loader:
                # img = img.view(img.size(0), -1)
                if not self.only_cpu:
                    img = img.cuda(self.did)
                    y = y.cuda(self.did)
                # ===================forward=====================
                x = self.model(img,True)
                py=x.max(dim=1)[1]
                tp+=(py==y).sum().item()
                n+=len(x)
                # loss = self.supervised_criterion(x,y)
            
        acc=float(tp)/n        
        # ===================log========================
        # self._info('net-id: {}, loss:{:.4f}, test acc:{:.2f}%'
        #         .format(self.id, loss.item(), float(tp)/n *100 ) )        
        return acc 

    def get_fitness(self):
        """
        Get the fitness of this individual
        Params
        ------
        Returns
        -------
        - valid_acc (float): the accuracy on the validation dataset
        """
        self._info('begin unsupervised training. (net-id: {})'.format(self.id)) 
        fitness=self.train_unsupervised()
        self._info('get fitness. net-id: {}, unsupervised loss {:.2f}'.format(self.id,fitness)) 
        # self._info('begin supervised training. (net-id: {})'.format(self.id)) 
        # valid_acc=self.train_supervised()
        # self._info('get fitness. net-id: {}, valid acc {:.2f}%'.format(self.id,valid_acc*100)) 
        # test_acc=self.test(self.test_dataloader)
        # self._info('net-id: {}, final test acc {:.2f}%'.format(self.id,test_acc*100))
        return fitness 

if __name__ == "__main__":
    
    runm=RunModel()
    runm.train_unsupervised()
    runm.train_supervised()
    runm.test()
    # torch.save(model.state_dict(), './sim_BaseVAE.pth')
    # z=th.randn((128,3)).cuda(self.did)
    # x_=model.decoder(z)
    # pic = to_img(x_.cpu().data)
    # save_image(pic,'./mlp_img/image_test2.png')
