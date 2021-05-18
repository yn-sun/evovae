# coding=utf-8

import os,time,sys,multiprocessing
import tqdm
import matplotlib.pyplot as plt
import torch as th
import numpy as np

from utils import _args,get_logger
from net import RunModel, BaseVAE, Dense, build_dataset
# from scikitTSVM import SKTSVM
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier

logger = get_logger(__name__)


class KNNClassifier():
    def __init__(self,k,cu_did=None):
        self.k = k 
        self.cu_did=cu_did

    def pdist(self,x,y):
        x=th.tensor(x)
        y=th.tensor(y)
        if self.cu_did is not None:
            x=x.cuda(self.cu_did)
            y=y.cuda(self.cu_did)
        
        distance = (x*x).sum(dim=1)[:,None]\
                +(y*y).sum(dim=1)[None]\
                -2*x@y.t()
        distance = distance.sqrt()
        distance = distance.cpu().detach().numpy()
        th.cuda.empty_cache()

        return distance

    def fit(self,x,y):
        self.x=x 
        self.y=np.array(y) 
    
    def score(self,x,y):
        distance=self.pdist(x,self.x)
        idx=distance.argsort(axis=1)

        if self.k == 1:
            pred_y = self.y[idx[:,0]]
        else:
            _votes = self.y[idx[:, : self.k]]
            pred_y = []
            for line in _votes:
                wash, counts = np.unique(line,return_counts=True)
                pred_y += wash[counts.argmax()]

        pred_y = np.array(pred_y)
        return (pred_y == y).sum()/float(len(y))
        

class SSL_M1(BaseVAE):
    def __init__(self,im_shape,LATENT_SIZE):
        """
        Params:
        -------
        im_shape    (tupple)    : shape of the image
        LATENT_SIZE (int)       : dimension of the latent manifold
        """

        super(SSL_M1,self).__init__()
        self.im_shape=im_shape
        input_size = np.prod(im_shape)
        self.latent_dim=LATENT_SIZE
        self.share_head = th.nn.Sequential(
            th.nn.Linear(input_size, 600),
            th.nn.Softplus(),
            th.nn.Linear(600, 600),
            th.nn.Softplus(),
        )
        
        self.encoder_mean = th.nn.Sequential(
            th.nn.Linear(600, LATENT_SIZE)
        )
        
        self.encoder_var = th.nn.Sequential(
            th.nn.Linear(600, LATENT_SIZE)
        ) 
        
        self.decoder = th.nn.Sequential(
            th.nn.Linear(LATENT_SIZE, 600),
            th.nn.Softplus(),
            th.nn.Linear(600, 600),
            th.nn.Softplus(),
            th.nn.Linear(600, input_size),
#             th.nn.Sigmoid(),
            th.nn.Tanh()
        )
        
        self.fn=th.nn.Sequential(*[
            Dense(LATENT_SIZE,512,act=th.nn.ReLU()),
            th.nn.Dropout(.5),
            Dense(512,10,act=th.nn.Softmax(dim=1))
        ])
        
    def forward(self,x,supervised=False):
        x=x.view(x.size(0),-1)
        if self.share_head is not None:
            x=self.share_head(x)
        x=BaseVAE.forward(self,x,supervised)
        if supervised:
            return x
        else:
            output,x_mean,x_var=x
            return output.view(-1,*self.im_shape),x_mean,x_var


def get_output_dir():
    # dir_path = os.path.dirname(__file__)
    # workspace_name = data_path.split('/')[-3] 
    # dir_path = os.path.join(dir_path,'../','single_test',workspace_name)
    # return dir_path
    if raw_classifier:
        dir_path = os.path.dirname(os.path.dirname(data_path))
        dir_path = os.path.join(dir_path, raw_classifier)
    else:
        dir_path = data_path.replace('snapshots','single_test')
        dir_path = os.path.join(dir_path,'net_id-'+str(net_id))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

def get_unsup_model_path():
    dir_path = get_output_dir()
    model_name = dataset+'-'+'unsup_epoch_'+str(unsupervised_train_epoch)+'.pth'
    model_path = os.path.join(dir_path,model_name)

    return model_path

def get_features_path(phase):
    dir_path = get_output_dir()
    feature_name = '-'.join([
        dataset,
        'unsup_epoch_'+str(unsupervised_train_epoch),
        phase,
        'features-semi-'+ str(num_per_cls)+'.pth' if num_per_cls is not None else 'features.pth'
    ])
    feature_path = os.path.join(dir_path,feature_name)

    return feature_path
    
def unsup_train():
    super_train_data_len = rmodel.sup_train_dataloader.dataset.__len__()
    logger.info('number of supervised training datapoint:', super_train_data_len)
    
    unsup_loss=rmodel.train_unsupervised()
    logger.info('best unsupervised loss: %.5f' % (0-unsup_loss))
    
    model_path = get_unsup_model_path()
    # weights=rmodel.model.state_dict()
    weights = rmodel.unsup_loss_w['w']
    th.save(weights,model_path)
    logger.info('save weight file to ' + model_path)

def extract():
    model_path = get_unsup_model_path()
    logger.info('load weight file from '+ model_path)
    rmodel.model.load_state_dict(th.load(model_path))

    logger.info('extract the features in training set and test set')
    train_f, trian_y=rmodel.model.extra_features_and_labels(rmodel.sup_train_dataloader)
    test_f, test_y =rmodel.model.extra_features_and_labels(rmodel.sup_test_dataloader)

    feat_path = get_features_path('sup_train')
    write_data={'train_f':train_f,'train_y':trian_y,'test_f':test_f,'test_y':test_y}
    th.save(write_data,feat_path)
    logger.info('save the features to ' + feat_path)


# class FCTrainer(th.nn.Module):
#     def __init__(self,classifier):
#         super(FCClassifier,self).__init__()

#         self.classifier = classifier

#     def fit(self,x,y):

#     def predict(self,x):
        

def sup():
    ######################################################
    model_path = get_unsup_model_path()
    if os.path.exists(model_path):
        logger.info('load weight file from '+ model_path)
        rmodel.model.load_state_dict(th.load(model_path))
    else:
        logger.warn('missed weight file --> ? '+ model_path)
    best_acc=rmodel.train_supervised()
    logger.info('test accuracy is: %.5f'%(best_acc))
    exit()
    ######################################################

    feat_phase = 'sup_train'
    feat_path = get_features_path(feat_phase)
    logger.info('load the %s features from %s'%(feat_phase,feat_path))

    datum = th.load(feat_path)
    train_f,train_y,test_f,test_y = datum['train_f'],datum['train_y'],datum['test_f'],datum['test_y']

    

    # use tsvm
    # tsvm=SKTSVM()
    # tsvm=NuSVC()
    svm=SVC(C=2.5)
    logger.info('training svm')
    svm.fit(train_f,train_y)
    logger.info('prediction in the test dataset')
    pred_y=svm.predict(test_f)
    test_acc=(pred_y == test_y).sum()/float(len(test_y))
    logger.info('test accuracy is: %.5f'%(test_acc))

def trivial_classifier(method):
    classifier={
        # 'NN':KNeighborsClassifier(1),
        'NN':KNNClassifier(1,cu_did),
        'SVM':SVC(C=2.5)
    }[method]
    
    unsup_train_ds, unsup_val_ds, unsup_test_ds, sup_train_ds, sup_val_ds, sup_test_ds  \
        = build_dataset(dataset, split_val=False, train_val_ratio=_args.train_val_ratio, num_per_cls=num_per_cls)

    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]

    for im, label in sup_train_ds:
        train_x+=[im.view(-1).numpy()]
        train_y+=[label]

    for im, label in sup_test_ds:
        test_x += [im.view(-1).numpy()]
        test_y += [label]

    classifier.fit(train_x,train_y)
    
    if _args.get_inf_time:
        iter_times = 10000
        timeall = []
        input_data = test_x[0:1]
        input_y = test_y[0:1]
        for i in range(iter_times):
            if i <= 100: 
                # remove the influence of the first time
                continue
            t0 = time.time()
            classifier.score(input_data,input_y)
            t1=time.time()

            timeall += [(t1-t0)*1000]
        timeall = np.array(timeall)
        delay_ms_mean, delay_ms_std = timeall.mean(), timeall.std()
        print('*'*100)
        print('dataset:',dataset,'classifier:',method,'time: %.4f (%.4f)'%(delay_ms_mean,delay_ms_std))
        print('*'*100)
        exit()

    acc=classifier.score(test_x,test_y)

    logger.info('test accuracy is: %.5f'%(acc))


    
if __name__=='__main__':
    # param
    data_path = os.path.expanduser(_args.state_path)
    OUT_CLASS_NUM = _args.out_cls_num
    supervised_train_epoch=_args.supervised_train_epoch
    unsupervised_train_epoch=_args.unsupervised_train_epoch
    dataset = _args.dataset
    num_per_cls = _args.num_per_cls
    cu_did=_args.cuda_did
    raw_classifier = _args.raw_classifier

    iter_data = th.load(os.path.expanduser(data_path))
    pop=iter_data['pop']
    fitness=iter_data['fitness']
    if fitness is None:
        # net_id=np.random.choice(len(pop))
        net_id=7
    else:
        fitness[fitness==0]=float('-inf')
        net_id = fitness.argmax()

    best_indi = pop[net_id]
    if raw_classifier:
        if raw_classifier == 'ssl_m1':
            unsup_train_ds, unsup_val_ds, unsup_test_ds, sup_train_ds, sup_val_ds, sup_test_ds  \
                = build_dataset(dataset, split_val=False, train_val_ratio=_args.train_val_ratio, num_per_cls=num_per_cls)
            im, label= sup_train_ds.__getitem__(0)
            im_shape = im.shape
            smodel=SSL_M1(im_shape,50)
            rmodel=RunModel(0,None,cu_did,
                1,1,
                DATASET=dataset,
                OUT_CLASS_NUM=OUT_CLASS_NUM,sup_train_epochs=supervised_train_epoch,unsup_train_epochs=unsupervised_train_epoch,
                num_per_cls=num_per_cls,
                specified_model=smodel,
                msg_buf=None)
        elif raw_classifier == 'CNN':
            indi={
                'head': 
                [
                    {'type': 'c', 'gene': [64, 3]},
                    {'type': 'p', 'gene': [0]},
                    {'type': 'c', 'gene': [128, 3]},
                    {'type': 'p', 'gene': [0]},
                ],
                'mu': 
                [
                    {'type': 'c', 'gene': [256, 3]},
                    {'type': 'p', 'gene': [0]},
                ],
                'sig': 
                [
                    {'type': 'f', 'gene': [1]}
                ],
                'latent': 
                [
                    {'type': 'l', 'gene': [512]}
                ],
                'dd':
                [
                    {'type': 'f', 'gene': [1]},
                    {'type': 'f', 'gene': [1]},
                    {'type': 'c', 'gene': [1, 1]},
                    {'type': 'c', 'gene': [1, 1]},
                    {'type': 'd', 'gene': [1]}
                ]
            }
            rmodel=RunModel(0,indi,cu_did,0,0,dataset,OUT_CLASS_NUM=OUT_CLASS_NUM,
                    sup_train_epochs=supervised_train_epoch,
                    unsup_train_epochs=unsupervised_train_epoch,
                    num_per_cls=num_per_cls)
            
            best_acc=rmodel.train_supervised()
            logger.info('test accuracy is: %.5f'%(best_acc))
            
            exit()
        else:
            trivial_classifier(raw_classifier)
            exit()
    else:
        rmodel=RunModel(net_id,best_indi,cu_did,0,0,dataset,OUT_CLASS_NUM=OUT_CLASS_NUM,
                    sup_train_epochs=supervised_train_epoch,
                    unsup_train_epochs=unsupervised_train_epoch,
                    num_per_cls=num_per_cls)

    if _args.get_inf_time:
        inf_time_mean, inf_time_std = rmodel.get_inf_time()
        print('*'*100)
        print(rmodel.model)
        print(data_path,'==>','inf time: %.4f(±%.4f) (ms)'%(inf_time_mean, inf_time_std))
        print('*'*100)
        exit()
    single_phase = _args.single_phase
    {
        'unsup':unsup_train,
        'extract':extract,
        'sup': sup
    }[single_phase]()
