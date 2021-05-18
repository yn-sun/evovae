# coding=utf-8
"""
@File    :   main.py,
@Time    :   2019-08-3,
@Author  :   Chen Xiangru,
@Version :   1.0,
@Contact :   None,
@License :   (C)Copyright None,
@Desc    :   Some trivials
@Algo    :   1. initialize chromsome
             2. get fitness 
             3. mating pool selection
             4. evolve: cross over & mutation
             5. environmental selection
"""

import torch as th 
import numpy as np 
import random
import matplotlib.pyplot as plt 
import os 
import time 
import multiprocessing
from multiprocessing.pool import  ThreadPool
import threading
from tqdm import tqdm 
import argparse
from abc import abstractclassmethod
import pdb
import json

from utils import _args, indi_copy

# Global variables
RAND_SEED = _args.rand_seed
POP_SIZE = _args.pop_size
MATE_POOL_SIZE = POP_SIZE
DNA_SIZE = None
CROSS_RATE = _args.cross_rate
MUTATE_RATE = _args.mutate_rate
ETA_C = _args.eta_c
ETA_M = _args.eta_m
ELITISM = _args.elitism
GENERATIONS = _args.generations

FIT_LOCKER = multiprocessing.Lock()
DID_LOCKER = multiprocessing.Lock()
# DID_LOCKER = threading.Lock()


# CNN related
CONV_NUM_RANGE = [1,6]  # convolution layer's number range
CONV_FEAT_RANGE = [20,128] # convolution feature map's range
FC_NUM_RANGE = [1,2]
FC_FEAT_RANGE = [64,256]
POOL_NUM_RANGE = [1,2]   # pool layer's number range
KERNEL_SIZE_RANGE = [2,5]
LATENT_DIM_RANGE = [2,128]
GENE_RANGE = {'c':[[*CONV_FEAT_RANGE,int],[*KERNEL_SIZE_RANGE,int]], # 
              'p':[[0,1,int],], # here 0 denotes max pooling, 1 denotes avg pooling
              'd':[[*CONV_FEAT_RANGE,int],],
              'f':[[*FC_FEAT_RANGE,int],],
              'l':[[*LATENT_DIM_RANGE,int],], # latent dim size
            } # Gene range to keep the offsprings' value being in legal range
# cumulative distributed function for `add`, `delete`, `modify`
# MUTATE_CDF=np.array([0,1/3,1/3,1/3]).cumsum() \
    # [np.hstack([0,np.tile(np.arange(1,4)[:,None],[1,2]).reshape(-1)[:-1]]).reshape(-1,2)]

# device related
#GPU_LIST = [0,1]
#PROC_POOL_SIZE = 8
GPU_LIST = [0,1]
PROC_POOL_SIZE = 12

# set the random seed
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
th.manual_seed(RAND_SEED)
th.random.manual_seed(RAND_SEED)
th.cuda.manual_seed(RAND_SEED)
th.cuda.manual_seed_all(RAND_SEED)
th.backends.cudnn.deterministic=True

import ga 
from utils import tournament_select, nn_seperated_crossover, \
                  poly_mutate_float, get_type_idx, get_logger
from net import RunModel
from dnn_client import DNNManager
logger = get_logger(__name__)

class M1Prototype(ga.GenericGeneticAlgo):
    def __init__(self):
        super(M1Prototype,self).__init__(DNA_SIZE, POP_SIZE,
                                         CROSS_RATE, MUTATE_RATE,
                                         ETA_C, ETA_M, MATE_POOL_SIZE,
                                         ELITISM, GENERATIONS)
        self.pnum_range = POOL_NUM_RANGE
        self.cnum_range = CONV_NUM_RANGE
        self.kern_range = KERNEL_SIZE_RANGE
        self.feat_range = CONV_FEAT_RANGE
        self.gene_range = GENE_RANGE
        self.fc_feat_range = FC_FEAT_RANGE
        self.fnum_range = FC_NUM_RANGE
        self.latent_range = LATENT_DIM_RANGE
        # self.mutate_cdf = MUTATE_CDF
        
        self.supervised_train_epoch=_args.supervised_train_epoch
        self.unsupervised_train_epoch=_args.unsupervised_train_epoch
        # dispatch related
        self.gpu_list = GPU_LIST    
        self.proc_ps=PROC_POOL_SIZE   

        if _args.dist_train:
            self.server_list=_args.servers
            self.port=_args.port
        else:
            self.server_list=None

        # workspace path
        self.workspace = _args.workspace
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)

        # save the args
        self._CMD_ARGS=_args
        # initialize
        self._initialize()

    def _initialize(self):
        self.pop=[]
        for i in range(POP_SIZE):

            # 1. first to confirm the layer's number...
            pnum_l, pnum_u = self.pnum_range
            pnum = np.random.choice(range(pnum_l, pnum_u + 1))
            
            cnum_l, cnum_u = self.cnum_range
            # head branch 
            chd_num = np.random.choice(range(cnum_l, cnum_u+1))
            # mean branch
            cmu_num = np.random.choice(range(0, cnum_u-chd_num+1))
            # std brach
            csg_num = np.random.choice(range(0, cnum_u-chd_num+1))
            # decoder branch
            cdd_num = np.random.choice(range(cnum_l, cnum_u))

            # pooling layer number assignment 
            # headp_num = min(np.random.choice(range(0,max(1,chd_num//2+1))),pnum)
            # intep_num = max(0, pnum - headp_num)
            phd_num = np.random.choice(range(pnum_l, pnum_u + 1))        
            pmu_num = np.random.choice(range(0, pnum_u-phd_num+1))
            psg_num = np.random.choice(range(0, pnum_u-phd_num+1))

            # deconv
            dcd_num = np.random.choice(range(pnum_l, pnum_u + 1))
            
            # fc layers
            fnum_l,fnum_u = self.fnum_range
            fmu_num = np.random.choice(range(fnum_l, fnum_u+1))
            fsg_num = np.random.choice(range(fnum_l, fnum_u+1))
            fdd_num = np.random.choice(range(fnum_l, fnum_u+1))

            # 2. initialize the each block.
            block_attrs={
                'head':{
                    'lnum':
                        {
                            'c':chd_num,
                            'p':phd_num,
                        }
                    ,
                    'perm_range':(1,chd_num+phd_num)
                },
                'mu':{
                    'lnum':
                        {
                            'c':cmu_num,
                            'p':pmu_num,
                            'f':fmu_num
                        },
                    'perm_range':(1,cmu_num+pmu_num)
                },
                'sig':{
                    'lnum':
                        {
                            'c':csg_num,
                            'p':psg_num,
                            'f':fsg_num,
                        }
                    ,
                    'perm_range':(1,csg_num+psg_num)
                },
                'latent':{
                    'lnum':
                        {
                            'l':1
                        }
                    ,
                    'perm_range':(0,1)
                },
                'dd':{
                    'lnum':
                        {
                            'f':fdd_num,
                            'c':cdd_num,
                            'd':dcd_num,
                        }
                    ,
                    'perm_range':(fdd_num,fdd_num+cdd_num+dcd_num)
                },
            }
            one_indi={}
            for bname,block in block_attrs.items():
                block_layer_type_and_num = block['lnum']
                bllpr_l,bllpr_u = block['perm_range']
                block_layers=[]
                for utype,unum in block_layer_type_and_num.items():
                    utype_layers = self._rand_layer(utype,unum)
                    for utype_layer in utype_layers:
                        block_layers+=[
                            {
                                'type':utype,
                                'gene':utype_layer
                            }
                        ]
                block_layers[bllpr_l:bllpr_u] = np.random.permutation(block_layers[bllpr_l:bllpr_u])
                one_indi[bname]=block_layers
            self.pop+=[one_indi]

    def _rand_layer(self,t,num):
        return {
            'c':self._rand_conv_layer,
            'p':self._rand_pool_layer,
            'd':self._rand_deconv_layer,
            'f':self._rand_fc_layer,
            'l':self._rand_latent_num,
        }[t](num).tolist()
    
    def _rand_latent_num(self,num):
        ld_l, ld_u = self.latent_range
        latent_dims = np.random.choice(range(ld_l, ld_u+1), size = (num, ), replace = True)
        latent_dims = latent_dims[:,None]

        return latent_dims

    def _rand_conv_layer(self,num):
        kn_l, kn_u = self.kern_range
        fn_l, fn_u = self.feat_range

        ks = np.random.choice(range(kn_l, kn_u+1), size = (num, ), replace = True)
        fs = np.random.choice(range(fn_l, fn_u+1), size = (num, ), replace = True) 
        
        conv_layers = np.vstack([fs,ks]).T # [num,2]

        return conv_layers

    def _rand_pool_layer(self,num):
        # max pool (0) or avg pool (1)    
        pool_layers = np.random.choice(range(2), size = (num, ), replace = True)
        pool_layers = pool_layers[:,None]
        return pool_layers 


    def _rand_deconv_layer(self,num):
        fn_l, fn_u = self.feat_range
        deconv_layers = np.random.choice(range(fn_l,fn_u+1),size=(num,),replace=True )
        deconv_layers = deconv_layers[:,None]
        return deconv_layers
    
    def _rand_fc_layer(self,num):
        fcn_l, fcn_u=self.fc_feat_range
        fc_layers = np.random.choice(range(fcn_l,fcn_u+1),size=(num,), replace=True )
        fc_layers = fc_layers[:,None]
        return fc_layers
        
    def _get_rank(self,fitness):
        """
        Get rank from the fitness
        Args:
            fitness (np.ndarray(float)[n]):
        Return:
            rank    (np.ndarray(int)[n])    :
        """
        # infact you can get the order or rank by only once sort.
        rank=(1-fitness).argsort().argsort() # [n]
        return rank
    
    @staticmethod
    def _pool_func(args):
        i,indi=args

        if _args.only_cpu == '0':
            # find a suitable gpu
            DID_LOCKER.acquire()

            # uniques,counter=np.unique(self.dispatch_list_shared,return_counts=True)
            # uniques,counter=np.unique(gDispatchList,return_counts=True)
            uniques,counter=np.unique(gDispatchList,return_counts=True)
            find_place=False
            # for t in self.gpu_list:
            for t in gGPUList:
                if t not in uniques:
                    did=t 
                    find_place=True
                    break 
            if not find_place:
                if -1 in uniques:
                    uniques=uniques[1:]
                    counter=counter[1:]
                did=uniques[counter.argmin()]

            # did= self.last_did % self.num_gpu
            # self.dispatch_list_shared[i]=did
            gDispatchList[i]=did
            # self.last_did+=1
            DID_LOCKER.release()
        else:
            did=0
        
        np.random.seed(i)
        did=np.random.choice(GPU_LIST)
        did=int(did)       
        logger.info('the did is: %d (net-id: %d)' %(did,i) )

        try:
            model = RunModel(i,indi,did)
            cur_fit= model.get_fitness()

        except Exception as e:
            logger.error(str(e)+ '(net-id: %d) | indi: %s'%(i, str(indi) ))
            cur_fit=float('-inf')
            print(model.model)
            raise e
        finally:
            # DID_LOCKER.acquire()
            if not _args.only_cpu:
                gDispatchList[i]=-1
            # DID_LOCKER.release()

            # FIT_LOCKER.acquire()
            if not _args.net_decode_test:
                gFitness[i]=cur_fit
            # FIT_LOCKER.release()
            th.cuda.empty_cache()
            
    def _get_fitness(self,pop):
        """
        Get fitness
        Args:
            pop             (list)
        Return:
            fitness         (np.ndarray(float)[n])  : the fitness of the population for this task
            fitness_rank    (np.ndarray(int)[n])    : rank of the population
        """
        # **fast test evolution
        # fitness=np.random.rand(len(pop))
        # fitness_rank=fitness.argsort()[::-1]
        # return fitness,fitness_rank
        # **end fast test

        if self.server_list:
            return self._get_fitness2(pop)
        else:
            return self._get_fitness1(pop)

    def _get_fitness2(self,pop):
        """
        Novel distributed `Get fitness` function
        Args:
            pop             (list)
        Return:
            fitness         (np.ndarray(float)[n])  : the fitness of the population for this task
            fitness_rank    (np.ndarray(int)[n])    : rank of the population
        """
        # logger.info('2333?')
        dnnm = DNNManager(self,
                          pop,_args.supervised_train_epoch,_args.unsupervised_train_epoch,
                          self.server_list,self.port,
                          max_jobs=_args.max_jobs,min_mem=_args.min_mem,
                          wait_gpu_change_delay=_args.wait_gpu_change_delay,
                          gpu_scale=_args.gpu_scale)
                          
        fitness = dnnm.run()
        fitness[fitness==0]=float('-inf')
        fitness_rank = self._get_rank(fitness)

        return fitness,fitness_rank

    def _get_fitness1(self,pop):
        """
        original single machine `Get fitness` funcion
        Args:
            pop             (list)
        Return:
            fitness         (np.ndarray(float)[n])  : the fitness of the population for this task
            fitness_rank    (np.ndarray(int)[n])    : rank of the population
        """
        # logger.info('WARNING: use random fitness function')
        # fitness = np.random.randn(len(pop))
        # fitness = np.zeros((len(pop)))
        # self.dispatch_list=np.full(len(pop),-1)
        
        # self.dispatch_list_shared=multiprocessing.Array('i',len(pop),lock=True)
        # self.fitness_shared=multiprocessing.Array('d',len(pop),lock=True)

        global gDispatchList,gFitness,gGPUList
        gDispatchList=multiprocessing.Array('i',len(pop),lock=True)
        gFitness=multiprocessing.Array('d',len(pop),lock=True)
        gGPUList=multiprocessing.Array('i',len(self.gpu_list))
        for i in range(len(gDispatchList)):
            # self.dispatch_list_shared[i]=-1
            # self.fitness_shared[i]=0
            gDispatchList[i]=-1
            gFitness[i]=float('-inf')

        for i,gpu_ in enumerate(self.gpu_list):
            gGPUList[i]=gpu_

        logger.info('fetch the fitness of each individule')
        with multiprocessing.Pool(self.proc_ps) as p:
        # with multiprocessing.pool.ThreadPool(self.proc_ps) as p:
            p.map(self._pool_func,enumerate(self.pop) )
            # fitness=np.array(self.fitness_shared)
            if _args.net_decode_test:
                fitness=np.random.rand(len(pop))
            else:
                fitness=np.array(gFitness)
            del gDispatchList,gFitness,gGPUList,

        logger.info('fitness is done')

        ##################################
        # for i,indi in enumerate(pop):
            # try:
            #     model = RunModel(i,indi)
            #     cur_fit= model.get_fitness()
            # except Exception as e:
            #     logger.error(str(e))
            #     cur_fit=0
            # finally:
            #     fitness[i]=cur_fit

            #######################
            # model = RunModel(i,indi,0)
            # cur_fit= model.get_fitness()
            # fitness[i]=cur_fit

        fitness[fitness==0]=float('-inf')
        fitness_rank=self._get_rank(fitness)
        
        return fitness,fitness_rank

    def _mate_pool_select(self,pop,mp_size,fitness_rank):
        """
        Perform binary tournament selection to generate mating pool
        Args:
            pop             (list)                      :
            mp_size         (int)                       :
            fitness_rank    (np.ndarray(int)[n])        : 
        Return:
            mp              (list)                      :
        """
        mp,_=tournament_select(pop,mp_size,fitness_rank,cap=2)

        return mp 

    def _cross_over(self,mp,cross_rate,eta):
        """
        Simulated binary crossover (SBX)
        Args:
            mp          (list) :
            cross_rate  (float):
            eta         (float):
        Return:
            offspring   (list) : the generated offsprings
        """
        # 1. make pairs
        pair_idx=np.random.choice(len(mp),size=(len(mp),),replace=False)
        pair_idx=pair_idx.reshape(-1,2) # [m/2,2]

        # mpair=mp[pair_idx] # [m/2,2,d]
        offspring=[]
        block_names = list(mp[0].keys())
        for i,j in pair_idx:
            indi1,indi2=mp[i],mp[j]
            
            if np.random.rand()<self.cross_rate and not _args.no_cx:
                # sbx
                parts = []
                for block_name in block_names:
                    p1,p2=nn_seperated_crossover(indi1[block_name],indi2[block_name],self.gene_range,self.eta_c)
                    parts+=[(p1,p2)]
                
                for q in zip(*parts):
                    new_indi={}
                    for bn_idx, p1 in enumerate(q):
                        bname=block_names[bn_idx]
                        new_indi[bname]=p1
                    offspring+=[new_indi]
            else:
                # deep copy
                # offspring += [json.loads(json.dumps(indi1)), json.loads(json.dumps(indi2))]
                offspring += [indi_copy(indi1), indi_copy(indi2)]

        return offspring
                    
    def get_layer_num(self,indi):
        """
        Get the layers' number
        Params
        ______
        - indi          (list): individule
        Returns
        _______
        - num_cumulator (dict)
        """
        type_names=self.gene_range.keys()
        num_cumulator={}
        for k,v in indi.items():
            block_counter={_: 0 for _ in type_names}
            for l in v:
                ltype=l['type']
                block_counter[ltype]+=1
            num_cumulator[k]=block_counter
        return num_cumulator

    def _mutate(self,ofspg,mutate_rate,eta):
        """
        Perform mutation in `ofspg`, inplace operation.
        Mutation: there types of mutation--`add`, `delete`, `modify`,
        Note that `add` and `delete` operation is based on the whole 
        chromosome, both change the length of the chromosome. While 
        `modify` is based on the unit gene, length are kept the same. 
        Params
        ______
        - ofspg       (object)
        - mutate_rate (float)
        - eta         (float)
        Returns
        _______
        - ofspg       (object): offsprings after mutation
        """
        if _args.no_mut:
            return ofspg
        # probalbility for `add`, `delete`, `modify`
        for i, indi in enumerate(ofspg):
            # rand = np.random.rand()
            # for mutate_opt,(lower, upper) in enumerate(self.mutate_cdf):
            #     if lower <= rand < upper:
            #         # all use reference.
            #         if mutate_opt==0:
            #             self._mutate_add_(indi)
            #         elif mutate_opt==1:
            #             self._mutate_del_(indi)
            #         elif mutate_opt==2:
            #             self._mutate_mod_(indi,self.mutate_rate)
            #         else:
            #             assert 0    
            mutate_opt=np.random.choice(range(3))
            [self._mutate_add_,self._mutate_del_,self._mutate_mod_][mutate_opt](indi)
        return ofspg

    def _mutate_add_(self,indi):
        """
        add operation, inplace
        Params
        ______
        - indi    (dict)
        Returns
        _______
        - indi    (dict)
        """
        num_cumulator=self.get_layer_num(indi)
        # 1.choose block
        rand_bname = np.random.choice(list(indi.keys()))

        # 2.choose utype
        lower_upper = self._get_feasible_lu(indi)
        utype=np.random.choice(list(lower_upper[rand_bname].keys()))   

        # 3.add
        if num_cumulator[rand_bname][utype]<lower_upper[rand_bname][utype]['num_range'][1]\
            and lower_upper[rand_bname][utype]['num_range'][0]< lower_upper[rand_bname][utype]['num_range'][1]\
            :
            new_l = self._rand_layer(utype,1)[0]
            irl,iru = lower_upper[rand_bname][utype]['insert_range']
            new_pos = np.random.choice(range(irl,iru+1))
            indi[rand_bname].insert(new_pos,{'type':utype,'gene':new_l})
        
        return indi

    def _get_feasible_lu(self,indi):
        """Get feasible lower and upper bound, auxiluary function for _mutate_add_ and _mutate_del_
        """
        layer_counter=self.get_layer_num(indi)
        cnum_l,cnum_u = self.cnum_range
        pnum_l,pnum_u = self.pnum_range
        fnum_l,fnum_u = self.fnum_range

        low_upper={
            'head':{
                'c':{
                    'num_range':(
                        max(cnum_l-layer_counter['mu']['c'], cnum_l-layer_counter['sig']['c'] ),
                        min(cnum_u-layer_counter['mu']['c'], cnum_u-layer_counter['sig']['c'])
                    ),
                    'insert_range':(
                       0, layer_counter['head']['c']+layer_counter['head']['p']
                    )
                },
                'p':{
                    'num_range':(
                        max(pnum_l-layer_counter['mu']['p'], pnum_l-layer_counter['sig']['p'] ),
                        min(pnum_u-layer_counter['mu']['p'], pnum_u-layer_counter['sig']['p'])
                    ),
                    'insert_range':(
                        0, layer_counter['head']['c']+layer_counter['head']['p']
                    )
                }
            },
            'mu':{
                'c':{
                    'num_range':(
                        cnum_l-layer_counter['head']['c'],
                        cnum_u-layer_counter['head']['c']
                    ),
                    'insert_range':(
                       0, layer_counter['mu']['c']+layer_counter['mu']['p']
                    )
                },
                'p':{
                    'num_range':(
                        pnum_l-layer_counter['head']['p'],
                        pnum_u-layer_counter['head']['p']
                    ),
                    'insert_range':(
                       0, layer_counter['mu']['c']+layer_counter['mu']['p']
                    )
                },
                'f':{
                    'num_range':(
                        fnum_l,
                        fnum_u
                    ),
                    'insert_range':(
                       layer_counter['mu']['c']+layer_counter['mu']['p'],
                       layer_counter['mu']['c']+layer_counter['mu']['p']+layer_counter['mu']['f']
                    )
                },
            },
            'sig':{
                'c':{
                    'num_range':(
                        cnum_l-layer_counter['head']['c'],
                        cnum_u-layer_counter['head']['c']
                    ),
                    'insert_range':(
                       0, layer_counter['sig']['c']+layer_counter['sig']['p']
                    )
                },
                'p':{
                    'num_range':(
                        pnum_l-layer_counter['head']['p'],
                        pnum_u-layer_counter['head']['p']
                    ),
                    'insert_range':(
                       0, layer_counter['sig']['c']+layer_counter['sig']['p']
                    )
                },
                'f':{
                    'num_range':(
                        fnum_l,
                        fnum_u
                    ),
                    'insert_range':(
                       layer_counter['sig']['c']+layer_counter['sig']['p'],
                       layer_counter['sig']['c']+layer_counter['sig']['p']+layer_counter['sig']['f']
                    )
                },
            },
            'latent':{
                'l':{
                    'num_range':(
                        float('inf'),
                        float('-inf')
                    ),
                    'insert_range':(
                        -1,-1
                    )
                }
            },
            'dd':{
                'f':{
                    'num_range':(
                        fnum_l,
                        fnum_u
                    ),
                    'insert_range':(
                       0,
                       layer_counter['dd']['f']
                    )
                },
                'c':{
                    'num_range':(
                        cnum_l,
                        cnum_u
                    ),
                    'insert_range':(
                       layer_counter['dd']['f'],
                       layer_counter['dd']['f']+layer_counter['dd']['c']+layer_counter['dd']['d']
                    )
                },
                'd':{
                    'num_range':(
                        pnum_l,
                        pnum_u,
                    ),
                    'insert_range':(
                       layer_counter['dd']['f'],
                       layer_counter['dd']['f']+layer_counter['dd']['c']+layer_counter['dd']['d']
                    )
                }
            }
        }
        
        return low_upper

    def _mutate_del_(self,indi):
        """
        del operation, inplace
        Params
        ______
        - indi   (dict)
        Returns
        _______
        - indi   (dict)
        """
        num_cumulator=self.get_layer_num(indi)
        # 1.choose block
        rand_bname = np.random.choice(list(indi.keys()))

        # 2.choose utype
        lower_upper = self._get_feasible_lu(indi)
        utype=np.random.choice(list(lower_upper[rand_bname].keys()))   

        # 3.del
        if num_cumulator[rand_bname][utype]>lower_upper[rand_bname][utype]['num_range'][0]\
            and num_cumulator[rand_bname][utype]>0\
            and lower_upper[rand_bname][utype]['num_range'][0]< lower_upper[rand_bname][utype]['num_range'][1]\
                :
            # del_pos=np.random.choice(range(*lower_upper[rand_bname][utype]['insert_range']))
            idx_list=get_type_idx(indi[rand_bname],utype)
            del_idx = np.random.choice(idx_list)
            del indi[rand_bname][del_idx]
        return indi 

    def _mutate_mod_(self,indi):
        """
        mod operation, inplace
        perform generic polynomial mutation
        Params
        ______
        - indi          (dict)
        Returns
        _______
        - indi          (dict)
        """
        mutate_rate=self.mutate_rate
        for k,v in indi.items():
            for l in v:
                utype=l['type']
                range_list=self.gene_range[utype]
                for pos,(u,(lower,upper,numeric_type)) in enumerate(zip(l['gene'],range_list)):
                    if np.random.rand()<mutate_rate:
                        u=poly_mutate_float(u,lower,upper,self.eta_m)
                    l['gene'][pos]=numeric_type(u)
        return indi 

    def _environmental_select(self,pop,pop_fitness,ofspg,elitism):
        """
        Environmental selection
        Params
        ------
        - pop             (list)                  :
        - pop_fitness     (np.ndarray(float))     : the fitness of the pop
        - ofspg           (list)                  :
        - elitism         (float)                 :
        Returns
        -------
        - pop             (list)                  : the new population
        - fitness         (np.ndarray(float))     : the fitness of the new population
        - fitness_rank    (np.ndarray(int))       : the rank of the new population
        """ 
        n,m=len(pop),len(ofspg)
        tnum=n+m
        osg_fit,_=self._get_fitness(ofspg) # [m]
        # joint_p=np.concatenate([pop,ofspg],axis=0) # [n+m]
        joint_p=pop+ofspg
        joint_p=np.array(joint_p,dtype=list)
        joint_fit=np.concatenate([pop_fitness,osg_fit],axis=0) # [n+m]
        joint_rank=self._get_rank(joint_fit) # [n+m]
        eli_num=int(tnum*elitism)
        div_num=self.pop_size-eli_num
        
        # 1. select the elites
        eli_mask=joint_rank<eli_num
        elitism_slt=joint_p[eli_mask]
        elitism_fit=joint_fit[eli_mask]
        elitism_rank=joint_rank[eli_mask]

        # 2. perform binary tournament selection for the rest individules
        diver_mask=~eli_mask
        diver_slt=joint_p[diver_mask]
        diver_fit=joint_fit[diver_mask]
        diver_rank=joint_rank[diver_mask]

        div_num=div_num+1 if div_num & 1 else div_num
        # diver_slt_ts,diver_slt_idx=self._tournament_select(diver_slt,div_num,diver_rank,cap=2)
        diver_slt_ts,diver_slt_idx=tournament_select(diver_slt,div_num,diver_rank,cap=2)
        diver_fit_ts=diver_fit[diver_slt_idx]
        diver_rank_ts=self._get_rank(diver_fit_ts)+len(elitism_rank)

        # 3. combine the elites and the normal one.
        final_pop=np.concatenate([elitism_slt,diver_slt_ts],axis=0)
        final_fit=np.concatenate([elitism_fit,diver_fit_ts],axis=0)
        final_rank=np.concatenate([elitism_rank,diver_rank_ts],axis=0)

        final_pop=final_pop[:self.pop_size].tolist()
        final_fit=final_fit[:self.pop_size]
        final_rank=final_rank[:self.pop_size]

        # for this task

        return final_pop,final_fit,final_rank
    
    def run(self):
        self.iters=0
        logger.info('GA(ITER:%d/%d): SAVING SNAPSHOT'%(self.iters+1,self.generations)) 

        self.save_snapshot(os.path.join(self.workspace,'snapshots'))
        
        if _args.snapshot != '':
            self.load_snapshot(_args.snapshot)
            logger.info('GA(ITER:%d/%d): LOADING SNAPSHOT -- PATH: <-- %s'%(self.iters+1,self.generations,_args.snapshot)) 

        if self.generations==1:
            logger.info('GA(ITER:%d/%d): RUNNING %d RANDOM NETWORKS' % (self.iters+1,self.generations,self.pop_size))

            self.fitness,self.fitness_rank=self._get_fitness(self.pop)

            self.iters+=1
            self.save_snapshot(os.path.join(self.workspace,'snapshots'))
            
        else:
            while self.iters<self.generations:
                logger.info('GA(ITER:%d/%d): RUNNING' % (self.iters+1,self.generations))
                
                logger.info('GA(ITER:%d/%d): BEGIN STEP'%(self.iters+1,self.generations)) 
                self.step()
                logger.info('GA(ITER:%d/%d): END STEP'%(self.iters+1,self.generations)) 
                
                self.iters+=1
                logger.info('GA(ITER:%d/%d): SAVING SNAPSHOT'%(self.iters+1,self.generations)) 
                self.save_snapshot(os.path.join(self.workspace,'snapshots'))
        self.iters=0

def func(x):
    M1Prototype()
    logger.info(x)

if __name__ == "__main__":
    m1=M1Prototype()
    
    m1.run()
    # import pdb;
    # pdb.set_trace()
    # m1.load_snapshot('./snapshots/state_gen2')
    # while 1:
    #     m1.run()
    # with multiprocessing.Pool(24) as p:
        # p.map(func,range(1000))
    
    logger.info('just ok')
