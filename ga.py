# coding=utf-8
"""
@File    :   ga.py,
@Time    :   2019-08-4,
@Author  :   Chen Xiangru,
@Version :   1.0,
@Contact :   None,
@License :   (C)Copyright None,
@Desc    :   The generic ga 
"""

import os

import torch as th 
import numpy as np 
import random

import matplotlib.pyplot as plt 
from abc import abstractclassmethod
from utils import get_logger

logger = get_logger(__name__)

class GenericGeneticAlgo(object):
    def __init__(self,dna_size,pop_size,cross_rate,mutate_rate,eta_c,eta_m,mp_size,elitism,generations):
        """
        Constructor of the GenericGeneticAlgo
        Args:
            dna_size    (tuple) : the size of the dna
            pop_size    (int)   :
            cross_rate  (float) :
            mutate_rate (float) :
            eta_c       (int)   : distributed index for sbx
            eta_m       (int)   : distributed index for mutation
            mp_size     (int)   :
            elitism     (float) : proportion of the elitism
            generations (int)   : the generations of the algorithm
        """
        self.dna_size=dna_size
        self.pop_size=pop_size
        self.cross_rate=cross_rate
        self.mutate_rate=mutate_rate
        self.eta_c=eta_c
        self.eta_m=eta_m
        self.mp_size=mp_size
        self.elitism=elitism
        self.generations=generations

        self.pop=None
        self.fitness=None
        self.fitness_rank=None
        self.mp=None
        self.offspring=None

        self.start_state=True
        self.iters=0
    
    def save_snapshot(self,save_dir):
        """
        Save snapshot to target directory
        Params
        ------
        - save_dir    (str): path
        Returns
        -------
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        keys=[
            'dna_size',
            'pop_size',
            'cross_rate',
            'mutate_rate',
            'eta_c',
            'eta_m',
            'mp_size',
            'elitism',
            'generations',
            'pop',
            'fitness',
            'fitness_rank',
            'mp',
            'offspring',
            'start_state',
            'iters'
        ]
        running_states={}
        for key in keys:
            assert hasattr(self,key)
            running_states[key]=getattr(self,key)
        th.save(running_states,os.path.join(save_dir,'state_gen%d'%(self.iters)) )
    
    def load_snapshot(self,state_path):
        """
        Load snapshot from file
        Params
        ------
        - state_path    (str): path of the state file
        Returns:
        --------
        """
        
        running_states=th.load(state_path)
        for key,val in  running_states.items():
            setattr(self,key,val)

    @abstractclassmethod
    def _initialize(self):
        pass 
    
    @abstractclassmethod
    def _get_fitness(self,pop):
        """
        Get fitness
        Args:
            pop             (object)    : the population
        Return:
            fitness         (object)    : the fitness
            fitness_rank    (object)    : rank of the population
        """
    
    @abstractclassmethod
    def _mate_pool_select(self,pop,mp_size,fitness_rank):
        """
        Binary tournament selection
        Args:
            pop             (object)    :
            mp_size         (int)       :
            fitness_rank    (object)    : 
        Return:
            mp              (object)    :
        """

    def _evolve(self,mp,cross_rate,eta_c,mutate_rate,eta_m):
        """
        genetic operators
        Args:
            mp                  (object):
            cross_rate          (float) :
            eta_c               (int)   :
            mutate_rate         (float) :
            eta_m               (int)   :         
        Return:
            ofspg           (object)    :
        """
        ofspg=self._cross_over(mp,cross_rate,eta_c)
        ofspg=self._mutate(ofspg,mutate_rate,eta_m)

        return ofspg
        
    @abstractclassmethod
    def _cross_over(self,mp,cross_rate,eta):
        """
        Simulated binary crossover (SBX)
        Args:
            mp          (object):
            cross_rate  (float):
            eta         (float):
        Return:
            offspring   (object): the generated offsprings
        """ 
    
    @abstractclassmethod
    def _mutate(self,ofspg,mutate_rate,eta):
        """
        Polynomial mutation
        Args:
            ofspg       (object):
            mutate_rate (float)
            eta         (float)
        Return:
            ofspg       (object): offsprings after mutation
        """
    
    @abstractclassmethod
    def _environmental_select(self,pop,pop_fitness,ofspg,elitism):
        """
        Environmental selection
        Args:
            pop             (object):
            pop_fitness     (object): the fitness of the pop
            ofspg           (object):
            elitism         (float) :
        Return:
            pop             (object): the new population
            fitness         (object): the fitness of the new population
            fitness_rank    (object): the rank of the new population
        """ 

    def step(self):
        """
        Step once
        """
        # 1. get fitness 
        if self.start_state:
            logger.info('GA(ITER:%d/%d): BEGIN GET FITNESS' % (self.iters+1,self.generations))
            self.fitness,self.fitness_rank=self._get_fitness(self.pop)
            logger.info('GA(ITER:%d/%d): END GET FITNESS' % (self.iters+1,self.generations))

            self.start_state=False
        # 2. mating pool selection
        logger.info('GA(ITER:%d/%d): BEGIN MATING POOL SELECTION' % (self.iters+1,self.generations))
        self.mp=self._mate_pool_select(self.pop,self.mp_size,self.fitness_rank)
        logger.info('GA(ITER:%d/%d): END MATING POOL SELECTION' % (self.iters+1,self.generations))
        
        # 3. evolve
        logger.info('GA(ITER:%d/%d): BEGIN EVOLVATION' % (self.iters+1,self.generations))
        self.offspring=self._evolve(self.mp,self.cross_rate,self.eta_c,self.mutate_rate,self.eta_m)
        logger.info('GA(ITER:%d/%d): END EVOLVATION' % (self.iters+1,self.generations))

        # 4. environmental selection
        logger.info('GA(ITER:%d/%d): BEGIN ENVIRONMENTAL SELECTION' % (self.iters+1,self.generations))
        self.pop,self.fitness,self.fitness_rank=self._environmental_select(self.pop,self.fitness,self.offspring,self.elitism)
        logger.info('GA(ITER:%d/%d): END ENVIRONMENTAL SELECTION' % (self.iters+1,self.generations))

    @abstractclassmethod
    def run(self):
        pass 

# take an example...
class Extremum(GenericGeneticAlgo):
    def __init__(self,*args):
        super(Extremum,self).__init__((1,),100,.8,.1,20,20,100,.2,100)
        
        self.func=lambda x: x*np.cos(2*x)+np.sin(10*x)*np.exp(x/4)
        self.x_bound=[-6,6]
        self.y_bound=[-10,10]

        self.y=None 

        self.pop=self._initialize()

    def _initialize(self):
        """
        Initlialize the population
        Args:
            none
        Return:
            pop     (np.ndarray(float)[n,d])
        """
        l,u=self.x_bound
        pop=np.random.rand(self.pop_size,*self.dna_size)*(u-l)+l # [n,d] 
        # pop-=3
        pop=pop.clip(*self.x_bound)
        return pop

    def _get_rank(self,fitness):
        """
        Get rank from the fitness
        Args:
            fitness (np.ndarray(float)[n,1]):
        Return:
            rank    (np.ndarray(int)[n])    :
        """
        # infact you can get the order or rank by only once sort.
        rank=fitness[:,0].argsort().argsort() # [n]
        return rank

    def _get_fitness(self,pop):
        """
        Get fitness
        Args:
            pop             (np.ndarray(float)[n,d])
        Return:
            fitness         (np.ndarray(float)[n,d])  : the fitness of the population for this task
            fitness_rank    (np.ndarray(int)[n])    : rank of the population
        """
        y=self.func(pop) # [n,1] 
        self.y=y
        
        y=-y
        # y_rank=y[:,0].argsort().argsort() # [n]
        y_rank=self._get_rank(y)

        fitness=y
        fitness_rank=y_rank
        
        return fitness,fitness_rank
    
    def _tournament_select(self,pop,mp_size,fitness_rank,cap=2):
        """
        Tournament selection
        Args:
            pop             (np.ndarray(float)[n,d]):
            mp_size         (int)                   :
            fitness_rank    (np.ndarray(int)[n])    :
            cap             (int)                   : the number of individules in one comparison
        Return:
            slt_res         (np.ndarray(float)[n,d]): the selected results.
            slt_idxo        (np.ndarray(int)[n])    : the index based on pop
        """
        idx=np.random.choice(len(pop),size=(cap*mp_size,),replace=True)
        # NOTE: it might choose two same individules as one pair,
        # but it's probability is low
        idx=idx.reshape(-1,cap) 
        slt_res=pop[idx] # [m,cap,d]
        slt_rank=fitness_rank[idx] # [m,cap]
        slt_res_idx=slt_rank.argsort(axis=1)[:,0]
        slt_res=slt_res[range(len(slt_res)),slt_res_idx] # [m,d]
        
        slt_idxo=idx[range(len(slt_res)),slt_res_idx]
        return slt_res,slt_idxo

    def _mate_pool_select(self,pop,mp_size,fitness_rank):
        """
        Perform binary tournament selection to generate mating pool
        Args:
            pop             (np.ndarray(float)[n,d])    :
            mp_size         (int)                       :
            fitness_rank    (np.ndarray(int)[n])        : 
        Return:
            mp              (np.ndarray(float)[m,d])    :
        """
        mp,_=self._tournament_select(pop,mp_size,fitness_rank,cap=2)

        return mp 
    
    def _sbx(self,mpair,xl,xu,cross_rate,eta=1):
        """
        Simulated binary crossover
        Args:
            mpair       (np.ndarray(float)[m/2,2,d])    :
            xl          (float)                         :
            xu          (float)                         :
            cross_rate  (float)                         :
            eta         (int)                           :
        Return: 
            offspring   (np.ndarray(float)[m,d])        :
        """
        mp1,mp2=mpair[:,0],mpair[:,1] # [m/2,d]

        # crossover mask
        cx_mask=np.random.rand(len(mp1))<cross_rate # [m/2]
        sbx_mask=np.random.rand(*mp1.shape)<.5 # [m/2,d]
        nz_mask=np.abs(mp1-mp2) > 1e-14 # [m/2,d]
        
        # 2.1 calculate offspring 1
        x1=np.minimum(mp1,mp2)
        x2=np.maximum(mp1,mp2)
        rand=np.random.rand(*mp1.shape)
        
        beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)
        rand_mask=rand>(1.0/alpha)
        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
        beta_q[rand_mask]=((1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))[rand_mask]
        
        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

        # 2.2 calculate offspring 2
        beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)
        rand_mask=rand>(1.0/alpha)
        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
        beta_q[rand_mask]=((1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))[rand_mask]
        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

        c1=c1.clip(xl,xu)
        c2=c2.clip(xl,xu)

        # 2.3 swap for different directions changes
        smask=np.random.rand(*mp1.shape )<.5
        c1[smask],c2[smask]=(c2[smask],c1[smask])

        # 2.4 collect the offsprings
        sbx_nz_mask=cx_mask[:,None] & sbx_mask & nz_mask
        offspring=mpair
        offspring[:,0,:][sbx_nz_mask]=c1[sbx_nz_mask]
        offspring[:,1,:][sbx_nz_mask]=c2[sbx_nz_mask]
        offspring=offspring.reshape(len(c1)+len(c2),-1) # [m,d]

        return offspring

    def _cross_over(self,mp,cross_rate,eta):
        """
        Perform sbx
        Args:
            mp          (np.ndarray(float)[m,d]):
            cross_rate  (float)                 :
            eta         (int)                   :
        Return:
            offspring   (np.ndarray(flot)[m,d]) :
        """
        # 1. make pairs
        pair_idx=np.random.choice(len(mp),size=(len(mp),),replace=False)
        pair_idx=pair_idx.reshape(-1,2) # [m/2,2]

        mpair=mp[pair_idx] # [m/2,2,d]

        # 2. SBX
        xl,xu=self.x_bound
        offspring=self._sbx(mpair,xl,xu,cross_rate,eta)
        
        return offspring

    def _poly_mutate(self,x,xl,xu,eta):
        """
        Args:
            x   (np.ndarray(float)[n,d])    :
            xl  (float)                     :
            xu  (float)                     :
            eta (int)                       :
        Return:
            x   ((np.ndarray(float)[n,d]))  :
        """
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        mut_pow = 1.0 / (eta + 1.)
        rand=np.random.rand(*x.shape)
        dq_mask=rand>=.5

        xy = 1.0 - delta_1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
        delta_q = val ** mut_pow - 1.0

        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
        delta_q[dq_mask]=(1.0 - val ** mut_pow)[dq_mask]

        x = x + delta_q * (xu - xl)
        x = x.clip(xl,xu)

        return x

    def _mutate(self,ofspg,mutate_rate,eta):
        """
        Perform polynomial mutation
        Args:
            ofspg       (np.ndarray(float)[m,d]):
            mutate_rate (float)
            eta         (float)
        Return:
            ofspg       (np.ndarray(float)[m,d]):   
        """
        x=ofspg # [m,d]
        xl,xu=self.x_bound

        # decide which should be mutated
        mt_mask=np.random.rand(*x.shape) < mutate_rate

        x=self._poly_mutate(x,xl,xu,eta)

        ofspg[mt_mask]=x[mt_mask]

        return ofspg

    def _environmental_select(self,pop,pop_fitness,ofspg,elitism):
        """
        Environmental selection
        Args:
            pop             (np.ndarray(float)[n,d]):
            pop_fitness     (np.ndarray(float)[n,d]): the fitness of the pop
            ofspg           (np.ndarray(float)[m,d]):
            elitism         (float)                 :
        Return: 
            pop             (np.ndarray(float)[n,d]): the new population
            fitness         (np.ndarray(float)[n])  : the fitness of the new population
            fitness_rank    (np.ndarray(int)[n])    : the rank of the new population
        """
        n,m=len(pop),len(ofspg)
        tnum=n+m
        osg_fit,_=self._get_fitness(ofspg) # [m,1]
        joint_p=np.concatenate([pop,ofspg],axis=0) # [n+m,d]
        joint_fit=np.concatenate([pop_fitness,osg_fit],axis=0) # [n+m,d]
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
        diver_slt_ts,diver_slt_idx=self._tournament_select(diver_slt,div_num,diver_rank,cap=2)
        diver_fit_ts=diver_fit[diver_slt_idx]
        diver_rank_ts=self._get_rank(diver_fit_ts)+len(elitism_rank)

        # 3. combine the elites and the normal one.
        final_pop=np.concatenate([elitism_slt,diver_slt_ts],axis=0)
        final_fit=np.concatenate([elitism_fit,diver_fit_ts],axis=0)
        final_rank=np.concatenate([elitism_rank,diver_rank_ts],axis=0)

        final_pop=final_pop[:self.pop_size]
        final_fit=final_fit[:self.pop_size]
        final_rank=final_rank[:self.pop_size]

        # for this task

        return final_pop,final_fit,final_rank

    def run(self):
        iter_times=1
        plt.ion()
        plt.figure(1)
        # plt.axis([*self.x_bound,*self.y_bound])
        plt.xlim(self.x_bound)
        plt.ylim(self.y_bound)
        xs=np.linspace(*self.x_bound,1000)
        ys=self.func(xs)

        plt.plot(xs,ys,linewidth=2 )
        pop=self.pop
        popy=self.func(pop)[:,0]        
        plt.scatter(pop,popy,s=20,c='r')
        plt.pause(2)

        while iter_times<=self.generations:
            plt.cla()
            plt.plot(xs,ys,linewidth=2)
            
            self.step()
            pop=self.pop
            popy=self.func(pop) 
            plt.scatter(pop,popy,s=20,c='r')
            plt.xlim(self.x_bound)
            plt.ylim(self.y_bound)
            
            best_idx=popy.argmax()
            plt.pause(0.016)
            print('Iter: %d/%d | Best: (%.3f,%.3f)'%(iter_times,self.generations,pop[best_idx],popy[best_idx]))

            iter_times+=1

        plt.ioff()

class EvoCNN(GenericGeneticAlgo):
    def __init__(self):
        pass 


if __name__ == "__main__":
    te1=Extremum()
    te1.run()
