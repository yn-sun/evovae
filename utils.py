# coding=utf-8
"""
@File    :   utils.py,
@Time    :   2019-08-4,
@Author  :   Chen Xiangru,
@Version :   1.0,
@Contact :   None,
@License :   (C)Copyright None,
@Desc    :   some auxiliary functions 
"""
# from atfork.stdlib_fixer import fix_logging_module
# fix_logging_module()

import numpy as np 
import argparse
import sys,os,time
import json 

np.set_printoptions(threshold=sys.maxsize)

_parser = argparse.ArgumentParser()
_parser.add_argument("-dist_train","--dist_train", help="enable distributed trainig",action='store_true')
_parser.add_argument("-servers","--servers", help="server list", type=str, default='0.0.0.0:cxr:123:0|1')
_parser.add_argument("-port","--port", help="network port",type=int,default=10007)

_parser.add_argument("-rand_seed","--rand_seed", help="random seed",type=int,default=0)
_parser.add_argument("-pop_size","--pop_size", help="size of population",type=int,default=100)
_parser.add_argument("-cross_rate","--cross_rate", help="rate of crossover",type=float,default=.8)
_parser.add_argument("-mutate_rate","--mutate_rate", help="rate of mutation",type=float,default=.1)
_parser.add_argument("-eta_c","--eta_c", help="distributed index for simulated binary crossover",type=int,default=20)
_parser.add_argument("-eta_m","--eta_m", help="distributed index for polynomial mutation",type=int,default=20)
_parser.add_argument("-elitism","--elitism", help="proportional between elite and pop",type=float,default=.2)
_parser.add_argument("-generations","--generations", help="generations of iteration",type=int,default=100)
_parser.add_argument("-workspace","--workspace", help="workspace path",type=str,default= 'job-'+time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
_parser.add_argument("-snapshot","--snapshot", help="snapshot path",type=str,default= '' )
_parser.add_argument("-max_jobs","--max_jobs", help="maximum job number in each gpu",type=int,default=3)
_parser.add_argument("-min_mem","--min_mem", help="minimum left gpu memory for allocation",type=int,default=3000)
_parser.add_argument("-wait_gpu_change_delay","--wait_gpu_change_delay", help="waiting time for gpu info change",type=int,default=20)
_parser.add_argument("-gpu_scale","--gpu_scale", help="proportional between gpu number and size of process pool",type=int,default=3)
_parser.add_argument("-only_cpu","--only_cpu", help="only use cpu",action='store_true' )

_parser.add_argument("-dataset","--dataset", help="Dataset for training",type=str,default='CIFAR10',choices=['CIFAR10','CIFAR100','STL10','MNIST','Caltech101','SVHN'])
_parser.add_argument("-train_val_ratio","--train_val_ratio", help="ratio between training samples number and validation samples number",type=float,default=9)

_parser.add_argument("-supervised_train_epoch","--supervised_train_epoch", help="epoch of supervised training",type=int,default=100)
_parser.add_argument("-unsupervised_train_epoch",'--unsupervised_train_epoch',help="epoch of unsupervised training",type=int,default=100)
# _parser.add_argument("-batch_size",'--batch_size',help="batch size",type=int,default=128)
_parser.add_argument("-batch_size",'--batch_size',help="batch size",type=int,default=32)
_parser.add_argument("-num_workers",'--num_workers',help="number of workers",type=int,default=4)
_parser.add_argument("-learning_rate",'--learning_rate',help="batch size",type=float,default=1e-3)
_parser.add_argument("-lr_schedule",'--lr_schedule',help="learning rate strategy",type=str,default='',choices=['','cosine'])


_parser.add_argument("-net_decode_test",'--net_decode_test',help="test the decoding of the gene",action='store_true')

# about the server
_parser.add_argument("-server_ip","--server_ip", help="server ip address",type=str,default='0.0.0.0')
# _parser.add_argument("-server_port","--server_port", help="server listen port",type=int,default=10007)
_parser.add_argument("-server_buf","--server_buf", help="server receive buffer size",type=int,default=4096)
_parser.add_argument("-server_msgbuf","--server_msgbuf", help="server shared message buffer size",type=int,default=2048)

# about the logger...
_parser.add_argument("-hide_server_log","--hide_server_log", help="hide the server logs in sub-thread and sub-process",type=str,default='0',choices=['0','1'])

# NOTE: for single test
_parser.add_argument("-state_path","--state_path", help="path of the snapshot",type=str)
_parser.add_argument("-cuda_did","--cuda_did", help="cuda device id",type=int)
_parser.add_argument("-out_cls_num","--out_cls_num", help="number of classes",type=int)
_parser.add_argument("-num_per_cls","--num_per_cls", help="number per class",type=int,default=None)
_parser.add_argument("-f","--f", help="for the jupyter notebook",type=str)
_parser.add_argument("-single_phase","--single_phase", help="Phase of the model inference", type=str, choices=['unsup','extract','sup'])
_parser.add_argument("-raw_classifier","--raw_classifier", help="raw classifier", type=str, choices=['SVM','NN','CNN','ssl_m1',None], default=None)
_parser.add_argument("-get_inf_time","--get_inf_time", help="show the time of inference", default=False, action='store_true')

# NOTE: for ablation experiments
_parser.add_argument("-no_cx", "--no_cx", help="disable the crossover",default=False, action='store_true')
_parser.add_argument("-no_mut", "--no_mut", help="disable the mutation", default=False,action='store_true')

_args = _parser.parse_args()

def get_logger(name,use_customer_log=True):
    if use_customer_log:
        logger=Logger()
    else:   
        import logging
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # logging.basicConfig(level = logging.INFO,
        #                     format = "%(asctime)s - %(name)s - File \"%(pathname)s\", line %(lineno)s %(filename)s %(funcName)s %(levelname)s - %(message)s")
        logger = logging.getLogger(name)
    return logger
    
PROC_HIDE_LOGGER=True if _args.hide_server_log == '1' else False

def tournament_select(pop,mp_size,fitness_rank,cap=2):
    """
    Tournament selection
    Args:
        pop             (np.ndarray(float)[n,d]): population
        mp_size         (int)                   : size of mating pool
        fitness_rank    (np.ndarray(int)[n])    : the rank of each individules
        cap             (int)                   : the number of individules in one comparison
    Return:
        slt_res         (np.ndarray(float)[n,d]): the selected results.
        slt_idxo        (np.ndarray(int)[n])    : the index based on pop
    """
    idx=np.random.choice(len(pop),size=(cap*mp_size,),replace=True)
    # NOTE: it might choose two same individules as one pair,
    # but it's probability is low
    idx=idx.reshape(-1,cap)
    rev=False
    if not isinstance(pop,np.ndarray):
        rev=True 
        # some numpy bugs may occur.
        pop=np.array(pop,dtype=list)

    slt_res=pop[idx] # [m,cap,d]
    slt_rank=fitness_rank[idx] # [m,cap]
    slt_res_idx=slt_rank.argsort(axis=1)[:,0]
    slt_res=slt_res[range(len(slt_res)),slt_res_idx] # [m,d]
    
    slt_idxo=idx[range(len(slt_res)),slt_res_idx]

    if rev:
        slt_res=slt_res.tolist()
    return slt_res,slt_idxo


def sbx_np(mpair,xl,xu,cross_rate,eta=1):
    """
    Simulated binary crossover in vectorization
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


def sbx_float(x1,x2,xl,xu,eta):
    """
    Ordinary simulated binary crossover 
    Args:
        x1  (float):  
        x2  (float):
        xl  (float): lower bound
        xu  (float): upper bound
        eta (int)  : distributed index
    Return:
        c1  (float): the offspring
        c2  (float): the offspring
    """

    if np.random.rand()<.5  and abs(x1-x2)>1e-14:
    
        # 2.1 calculate offspring 1
        # x1=min(x1,x2)
        # x2=max(x1,x2)
        x1,x2 = (min(x1,x2), max(x1,x2))
        rand=np.random.rand()
        
        beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)

        if rand< (1.0/alpha): # rand_mask=rand>(1.0/alpha)
            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
        else:
            beta_q=((1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))
        
        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

        # 2.2 calculate offspring 2
        beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)
        if rand < (1.0/alpha):  # rand_mask=rand>(1.0/alpha)
        
            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
        else:
            beta_q=((1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))
        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

        c1=min(max(c1,xl),xu) # c1=c1.clip(xl,xu)
        c2=min(max(c2,xl),xu) # c2=c2.clip(xl,xu)

        # 2.3 swap for different directions changes
        if np.random.rand()<.5: # smask=np.random.rand(*mp1.shape )<.5
            c1,c2=c2,c1 # c1[smask],c2[smask]=(c2[smask],c1[smask])

        # 2.4 collect the offsprings
        # sbx_nz_mask=cx_mask[:,None] & sbx_mask & nz_mask
        # offspring=mpair
        # offspring[:,0,:][sbx_nz_mask]=c1[sbx_nz_mask]
        # offspring[:,1,:][sbx_nz_mask]=c2[sbx_nz_mask]
        # offspring=offspring.reshape(len(c1)+len(c2),-1) # [m,d]
        return c1,c2
    return x1,x2

def poly_mutate_np(x,xl,xu,eta):
    """Polynomial mutation in vectorization
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


def poly_mutate_float(x,xl,xu,eta):
    """Ordinary polynomial mutation
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
    rand=np.random.rand()
    
    if rand < .5 : # dq_mask=rand>=.5
        xy = 1.0 - delta_1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
        delta_q = val ** mut_pow - 1.0
    else:
        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
        delta_q=(1.0 - val ** mut_pow)

    x = x + delta_q * (xu - xl)
    x= min(max(x,xl),xu) #x = x.clip(xl,xu)

    return x

def nn_seperated_crossover(seq_net1,seq_net2,gene_range,eta,unit_types=['c','p','d','f','l']):
    """
    Seperated crossover for a sequential networks according to the unit type.
    This implementation follows the `Unit Alignment` strategy propopsed in
    this paper: `Sun, Yanan, et al. "Evolving deep convolutional neural networks 
    for image classification." IEEE Transactions on Evolutionary Computation (2019).`
    ``
    Params
    ______
    - seq_net1    (list): sequential networks' information in list,
                          e.g. [{'type':'c','gene':[oc,ks,...]},...}
    - seq_net2    (list): sequential networks' information in list 
                          e.g. [{'type':'c','gene':[oc,ks,...]},...}
    - gene_range (dict): range of the paricular genetic value.
                         e.g. {'c':[lower,upper,numeric_type],...}
    - eta        (int) : distributed index for the simulated binary crossover.
    - unit_types  (list): type list in the networks
    Returns
    _______
    - ofspg1      (list): 1st offspring
    - ofspg2      (list): 2nd offspring
    """
    if unit_types is None:
        unit_types = gene_range.keys()
    else:
        # pass 
        assert set(unit_types) == gene_range.keys()

    ofspg1 = [None]*len(seq_net1)
    ofspg2 = [None]*len(seq_net2)
    for utype in unit_types:
        # 1. collect into another list
        indi1_pos = []
        indi2_pos = []
        cx_list1=[]
        cx_list2=[]
        for i,unit in enumerate(seq_net1):
            utype_ = unit['type']
            if utype_ == utype:
                indi1_pos+=[i]
                cx_list1+=[unit]
        
        for i,unit in enumerate(seq_net2):
            utype_ = unit['type']
            if utype_ == utype:
                indi2_pos+=[i]
                cx_list2+=[unit]
        o1,o2=nn_aligned_crossover(cx_list1,cx_list2,gene_range,eta)
        
        # 2. put back
        for i,pos in enumerate(indi1_pos):
            ofspg1[pos]=o1[i]
        for i,pos in enumerate(indi2_pos):
            ofspg2[pos]=o2[i]
    
    return ofspg1,ofspg2

def nn_aligned_crossover(seq_net1,seq_net2,gene_range,eta):
    """
    Aligned crossover for the same type unit's list.
    Params
    ______
    - seq_net1   (list): e.g. [{'type':'c','gene':[oc,ks,...]},...}
    - seq_net2   (list): e.g. [{'type':'c','gene':[oc,ks,...]},...}
    - gene_range (dict): range of the paricular genetic value.
                         e.g. {'c':[lower,upper,numeric_type],...}
    - eta        (int) : distributed index for the simulated binary crossover.
    Returns
    _______
    - ofspg1     (list)
    - ofspg2     (list)
    """
    ofspg1 = []
    ofspg2 = []
    for u1,u2 in zip(seq_net1,seq_net2):
        utype=u1['type']
        range_list=gene_range[utype]
        
        g1_list=[]
        g2_list=[]
        for ug1,ug2,(lower,upper,numeric_type) in zip(u1['gene'],u2['gene'],range_list):
            o1,o2=sbx_float(ug1,ug2,lower,upper,eta)
            o1=numeric_type(o1)
            o2=numeric_type(o2)
            g1_list+=[o1]
            g2_list+=[o2]
        
        ofspg1 += [{'type':utype,'gene':g1_list}]
        ofspg2 += [{'type':utype,'gene':g2_list}]
    
    if len(ofspg1) < len(seq_net1):
        # ofspg1+=seq_net1[len(ofspg1):]
        for i in  seq_net1[len(ofspg1):]:
            ofspg1+=[{'type':i['type'],'gene': [_ for _ in i['gene']] }]
    
    if len(ofspg2) < len(seq_net2):
        # ofspg2+=seq_net2[len(ofspg2):]
        for i in seq_net2[len(ofspg2):]:
            ofspg2+=[{'type':i['type'],'gene': [_ for _ in i['gene']]}]
    
    return ofspg1,ofspg2

def indi_copy(indi):
    """Copy the individual
    Params
    ------
    - indi      (dict): individual
    Returns
    -------
    - indi_copy (dict): clone of the param `indi`
    """
    indi_copy = json.loads(json.dumps(indi))
    return indi_copy

def get_type_idx(l,type_str):
    """
    Get the index of a particular type uints in a sequential networks
    Params
    ------
    - l         (list)
    - type_str  (str)
    Returns
    -------
    - idx       (list)
    """
    idx=[]
    for i,unit in enumerate(l):
        utype=unit['type']

        if utype == type_str:
            idx+=[i]
    return idx 

class Logger():
    LEVEL_STRS=[
        'DEBUG',
        'INFO',
        'WARN',
        'ERROR',
        'NONE'
    ]
    
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    NONE = 4
    
    def __init__(self):
        self.log_level = 0
        
        self.FG_DATETIME = True
        self.FG_LINENO = True
        self.FG_FILENAME = True
        self.FG_FUNCNAME = True
        
        self.set_opt()

    def set_level(self,level):
        """
        log.set_level(log.DEBUG)
        """
        self.log_level = level

    def set_opt(self,datetime=True, lineno=True, filename=True, funcnane=True):
        self.FG_DATETIME = datetime
        self.FG_LINENO = lineno
        self.FG_FILENAME = filename
        self.FG_FUNCNAME = funcnane
    
    def _format_str(self,t, name, funcname, line, prefix, content):
        res_str=" - ".join(filter(lambda x:str(x) != "", (t, name, funcname, line, prefix,content))) 
        return res_str

    def _dolog(self,*args, **kw):
        def tostr(arg):
            if isinstance(arg, list):
                arg = "[" + ",".join(map(str, arg)) + "]"
            elif isinstance(arg, tuple):
                arg = "(" + ",".join(map(str, arg)) + ")"
            elif isinstance(arg, set):
                arg = "{" + ",".join(map(str, arg)) + "}"
            elif isinstance(arg, dict):
                arg = "{" + ",".join(["%s:%s" % (str(k), str(v)) for k,v in arg.iteritems()]) + "}"
            else:
                arg = str(arg)
            return arg

        reparg = " ".join(map(tostr, args))
        repkw = ""
        if kw:
            repkw = " | " + " ".join(["%s=%s" % (str(k), str(v)) for k,v in kw.iteritems()])

        repstr = reparg + repkw
        return repstr
    
    def _log_to(self,log_level,file=sys.stdout,flush=True,*args, **kw):
        if log_level > len(self.LEVEL_STRS): return
        prefix = "%s" % (self.LEVEL_STRS[log_level])

        # traceback info
        try:
            raise Exception
        except:
            f_tb = sys.exc_info()[2].tb_frame.f_back

            t = time.strftime("%Y-%m-%d %X") if self.FG_DATETIME else ""
            name = os.path.basename(f_tb.f_code.co_filename) if self.FG_FILENAME else ""
            funcname = f_tb.f_code.co_name if self.FG_FUNCNAME else ""
            line = str(f_tb.f_lineno) if self.FG_LINENO else "" # int

        res_str = self._format_str (t, name, funcname, line, prefix,self._dolog(*args, **kw))

        print(res_str,flush=flush,file=file)

    def debug(self,*args, **kw):
        log_level=self.DEBUG
        file=sys.stdout
        flush=True

#         self._log_to(self.DEBUG,sys.stdout,True,*args,**kw)
        
        if log_level > len(self.LEVEL_STRS): return
        prefix = "%s" % (self.LEVEL_STRS[log_level])

        # traceback info
        try:
            raise Exception
        except:
            f_tb = sys.exc_info()[2].tb_frame.f_back

            t = time.strftime("%Y-%m-%d %X") if self.FG_DATETIME else ""
            name = os.path.basename(f_tb.f_code.co_filename) if self.FG_FILENAME else ""
            funcname = f_tb.f_code.co_name if self.FG_FUNCNAME else ""
            line = str(f_tb.f_lineno) if self.FG_LINENO else "" # int

        res_str = self._format_str (t, name, funcname, line, prefix,self._dolog(*args, **kw))

        print(res_str,flush=flush,file=file)

    def info(self,*args, **kw):
#         self._log_to(self.INFO,sys.stdout,True,*args,**kw)
        log_level=self.INFO
        file=sys.stdout
        flush=True
        
        if log_level > len(self.LEVEL_STRS): return
        prefix = "%s" % (self.LEVEL_STRS[log_level])

        # traceback info
        try:
            raise Exception
        except:
            f_tb = sys.exc_info()[2].tb_frame.f_back

            t = time.strftime("%Y-%m-%d %X") if self.FG_DATETIME else ""
            name = os.path.basename(f_tb.f_code.co_filename) if self.FG_FILENAME else ""
            funcname = f_tb.f_code.co_name if self.FG_FUNCNAME else ""
            line = str(f_tb.f_lineno) if self.FG_LINENO else "" # int

        res_str = self._format_str (t, name, funcname, line, prefix,self._dolog(*args, **kw))

        print(res_str,flush=flush,file=file)

    def warn(self,*args, **kw):
#         self._log_to(self.WARN,sys.stdout,True,*args,**kw)
        log_level=self.WARN
        file=sys.stdout
        flush=True
        
        if log_level > len(self.LEVEL_STRS): return
        prefix = "%s" % (self.LEVEL_STRS[log_level])

        # traceback info
        try:
            raise Exception
        except:
            f_tb = sys.exc_info()[2].tb_frame.f_back

            t = time.strftime("%Y-%m-%d %X") if self.FG_DATETIME else ""
            name = os.path.basename(f_tb.f_code.co_filename) if self.FG_FILENAME else ""
            funcname = f_tb.f_code.co_name if self.FG_FUNCNAME else ""
            line = str(f_tb.f_lineno) if self.FG_LINENO else "" # int

        res_str = self._format_str (t, name, funcname, line, prefix,self._dolog(*args, **kw))

        print(res_str,flush=flush,file=file)

    def error(self,*args, **kw):
#         self._log_to(self.ERROR,sys.stderr,True,*args,**kw)
        log_level=self.ERROR
        file=sys.stderr
        flush=True
        
        if log_level > len(self.LEVEL_STRS): return
        prefix = "%s" % (self.LEVEL_STRS[log_level])

        # traceback info
        try:
            raise Exception
        except:
            f_tb = sys.exc_info()[2].tb_frame.f_back

            t = time.strftime("%Y-%m-%d %X") if self.FG_DATETIME else ""
            name = os.path.basename(f_tb.f_code.co_filename) if self.FG_FILENAME else ""
            funcname = f_tb.f_code.co_name if self.FG_FUNCNAME else ""
            line = str(f_tb.f_lineno) if self.FG_LINENO else "" # int

        res_str = self._format_str (t, name, funcname, line, prefix,self._dolog(*args, **kw))

        print(res_str,flush=flush,file=file)

if __name__ == "__main__":
    print('test utils')
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))
    # print(sbx_float(1.24,2.5,-1,5,2))

    # x=np.array([1,2,3,4],dtype=float)
    # y=np.array([5,6,7,8],dtype=float)
    # mpair=np.concatenate([x[:,None,None],y[:,None,None]],axis=1)
    # print(sbx_np(mpair,0,10,.8,4))

    # x=np.array([1,2,3,4],dtype=float)
    # print(poly_mutate_np(x,0,10,3))

    # print(poly_mutate_float(1.24,-1,5,2))
    # print(poly_mutate_float(1.24,-1,5,2))
    # print(poly_mutate_float(1.24,-1,5,2))
    # print(poly_mutate_float(1.24,-1,5,2))
    # print(poly_mutate_float(1.24,-1,5,2))

    # n1=[{'type':'c','gene':np.array([10,4])},{'type':'c','gene':np.array([80,4])},{'type':'c','gene':np.array([10,2])}]
    # n2=[{'type':'c','gene':np.array([20,5])},{'type':'c','gene':np.array([10,2])},{'type':'d','gene':np.array([20,5])} ]
    # gene_range={'c':np.array([[10,100,int],[2,5,int]]),'d':np.array([[10,100,int],[2,5,int] ]) }
    # o1,o2 = nn_aligned_crossover(n1,n2,gene_range,5)
    # o1,o2 = nn_seperated_crossover(n1,n2,gene_range,5)
    # print(o1,'|',o2)
