# coding=utf-8

import multiprocessing,subprocess,os,sys,time,threading
from multiprocessing import pool

import numpy as np
import random

import struct
import math
import codecs
import pickle
from utils import get_logger
logger = get_logger(__name__)
np.set_printoptions(threshold=sys.maxsize)

class NetMessage(object):
    TYPES = ['START', 'END', 'RESULT', 'LOGGER', 'HEART', 'FILE']

    @staticmethod
    def pack(msg_type, pay_load):
        """
        Pack the payload to binary format
        Params
        ------
        - msg_type   (str)   : the type of this message
        - pay_load   (str)   : the payload
        Returns
        -------
        - msg        (bytes) : message in bytes
        """
        NetMessage.check_type(msg_type)
        msg = {'type': msg_type, 'data': pay_load}
        msg = repr(msg).encode()

        return msg
    

    @staticmethod
    def unpack(msg):
        """
        Unpack the message
        Params
        ------
        - msg       (bytes)
        Returns
        -------
        - msg_type  (str)
        - pay_load  (str)
        """
        msg = eval(msg.decode())
        msg_type = msg['type']
        NetMessage.check_type(msg_type)
        pay_load = msg['data']
        
        return msg_type, pay_load
        
    @staticmethod
    def check_type(msg_type):
        """
        Check the message type
        Params
        ------
        - msg_type    (str)
        Return
        ------
        Raise
        -----
        - ValueError
        """
        if msg_type not in NetMessage.TYPES:
            raise ValueError('invalid message type, must be one of %s' % str(NetMessage.TYPES) )

class ProcShareMessage(object):
    """
    Shared messgae class bettween process
    Params
    ------
    - buf_size (int): the buffer size of the data
    """
    def __init__(self, buf_size = 2048):
        self._data_buf = multiprocessing.Array('c', buf_size, lock = True)
        self._rlock = multiprocessing.RLock()
        self.clear()

    def pack(self, valid, msg_str):
        """
        Pack the message into binary format, and write into the `_data_buf`
        Params
        ------
        - valid     (bool)                  : the valid flag
        - msg_str   (str)                   : the payload string data
        Returns
        -------
        - _data_buf (Array('c', buf_size))  : the buffer
        """
        self._rlock.acquire()

        msg_pack = {'valid':valid, 'data': msg_str}
        msg_pack = repr(msg_pack).encode()
        self._data_buf[:len(msg_pack)] = msg_pack
        self._data_buf[len(msg_pack):] = b'\0' * (len(self._data_buf) - len(msg_pack))
        
        self._rlock.release()

        return self._data_buf
 
    def print(self, msg_str):
        # return self.pack(True,msg_str)
        return self._write_logger(msg_str)

    def info(self, msg_str):
        # return self.pack(True,msg_str)
        return self._write_logger(msg_str)

    def save_file(self, file_path):
        return self._write_file(file_path)
        
    def _write_logger(self, msg_str):
        payload = {'msg_type':'logger','content':msg_str} 
        
        return self.pack(True,payload)

    def _write_file(self, file_path):
        """
        Params:
        -------
        - file_content  (object): python object
        - file_name     (str)   : name of the file
        """
        payload = {'msg_type':'file','file_path':file_path}

        return self.pack(True,payload)

    def unpack(self,clear):
        """
        Unpack the messgae
        Params
        ------
        - clear     (bool): wether clear the buffer
        Returns
        -------
        - valid     (bool)
        - msg_str   (str)
        """
        self._rlock.acquire()
        
        msg_ = eval(self._data_buf[:].decode().replace('\0',''))
        valid = msg_['valid']
        msg_str = msg_['data']

        if clear:
            self.clear()
        
        self._rlock.release()

        return valid, msg_str

    def clear(self):
        self.pack(False, '')


class GPUInfoHelper():
    """
    Helper for get the gpu info
    """
    def __init__(self,server_list,max_jobs=3,min_mem=3000):
        self.server_list=server_list
        self.gpus_info=None
        self.max_jobs=max_jobs
        self.min_mem=min_mem
        
    def get_gpus_info(self):
        """
        Return `gpus_info`
        Returns
        -------
        - gpus_info     (dict): the dict results of the query
        """
        # self.locker=threading.Lock()
        server_list=self.server_list.split(',')
        self.nvidia_infos=[None]*len(server_list)
        
        with multiprocessing.Pool(4) as p:
            self.nvidia_infos=p.map(self.thread_func,enumerate(server_list))
        
        # self.nvidia_infos = [ self.thread_func((idx, single_server)) for idx, single_server in enumerate(server_list)]
        gpus_info=[]

        for nvidia_info,server in zip(self.nvidia_infos,server_list):
            host_ip,_,_,gpu_str=server.split(':')
            gpu_enabled_list=gpu_str.split('|')
            nvidia_info=nvidia_info.split('\n')
            list1=list_temp=[]
            list2=[]
            for line in nvidia_info:
                if line.startswith(' '*5):
                    list_temp=list2
                    continue
                list_temp+=[line]

            num_gpu=(len(list1)-7)//3

            job_num_list=[0]*num_gpu
            job_num=len(list2)-6
            if job_num==1 and list2[4].split()[1]=='No':
                job_num=0
                # print('no job found')
            else:
                for i in range(job_num):
                    job_in_gpu=list2[4+i].split()[1]
                    job_in_gpu=int(job_in_gpu)
                    job_num_list[job_in_gpu]+=1

            for i in range(num_gpu):
                if str(i) not in gpu_enabled_list:
                    continue
                gpu_model=list1[7+3*i].split('|')[:]
                load_info=list1[8+3*i].split('|')[:]

                gpu_name=' '.join(gpu_model[1].split()[1:-1])
                used_mem,total_mem=load_info[2].split('/')

                used_mem=used_mem.split()[0]
                used_mem=used_mem[:len(used_mem)-3]
                total_mem=total_mem.split()[0]
                total_mem=total_mem[:len(total_mem)-3]
                total_mem=int(total_mem)
                used_mem=int(used_mem)
                left_mem=total_mem-used_mem
                gpus_info+=[{'name':gpu_name,'gpu_slot':i,'total_mem':total_mem,'used_mem':used_mem,'left_mem':left_mem,'job_num':job_num_list[i],'ip':host_ip}]
                #print('name:',gpu_name,' used:',used_mem,' total:',total_mem,' left:',left_mem)
            
        self.gpus_info=gpus_info
        
        return gpus_info
    
    def get_adequate_gpu(self,block=False):
        """
        Get adequate gpu info list,
        More memory are considered as the first
        Returns
        -------
        - adequatelist  (list)
        - block         (bool): if true, will loop to get the adequate gpu list
        """
        adequatelist=[]
        while len(adequatelist)<=0:
            adequatelist=[]
            self.get_gpus_info()
            random.shuffle(self.gpus_info)
            for gpu_info in self.gpus_info:
                left_mem=gpu_info['left_mem']
                job_num=gpu_info['job_num']
                
                if left_mem>=self.min_mem and job_num<=self.max_jobs:
                    adequatelist+=[gpu_info]
            if not block:
                break
        adequatelist=sorted(adequatelist,key=lambda x: x['left_mem'],reverse=True)
        return adequatelist
        
    def thread_func(self,args):
        i,server=args
        host=server.split(':')
        ip,user,token,gpu_str=host
        # NOTE: use this popen in a pool.map function may cause hang...
        # nvidia_info=subprocess.Popen(['sshpass','-p',token,'ssh','%s@%s'%(user,ip),'nvidia-smi'],stdout=subprocess.PIPE).stdout.read().decode()
        nvidia_info=os.popen('sshpass -p '+token+' ssh'+ ' %s@%s '%(user,ip)+' nvidia-smi').read()
        #print(nvidia_info,flush=True)
        # self.locker.acquire()
        # self.nvidia_infos[i]=nvidia_info
        # self.locker.release()

        return nvidia_info


class SocketBuffer():
    """
    SocketBuffer
    Params
    ------
    - sock     (socket): socket object
    - buf_size (int)   : buffer size in once sending
    """
    def __init__(self,sock,buf_size=1024):
        self.sock=sock
        self.buf_size=buf_size
        self.ack=b'ack'
        self.header_fmt='LLL'

    def close(self):
        """
        Close the socket
        """
        self.sock.close()

    def send(self,bstr):
        """
        Params
        ------
        - bstr (bytes)
        """
        # 1. send header
        byte_num=len(bstr)
        block_num=math.ceil(len(bstr)/float(self.buf_size))
        header=struct.pack(self.header_fmt,byte_num,block_num,self.buf_size)
        self.sock.send(header)
        data=self.sock.recv(len(self.ack))
        assert data==self.ack

        # 2. send data
        for i in range(block_num):
            start_pos=i*self.buf_size
            end_pos=min((i+1)*self.buf_size,byte_num)
            self.sock.send(bstr[start_pos:end_pos])

            data=self.sock.recv(len(self.ack))
            assert data==self.ack
        return byte_num

    def recv(self,ignored_buf_size):
        """
        receive
        Params
        ------
        - ignored_buf_size (int): ignored attribute for convenience of upper invoker
        """
        # 1. recv header
        header=self.sock.recv(len(struct.pack(self.header_fmt,*([0]*len(self.header_fmt)))))
        if header is None or header==b'':
            return None

        byte_num,block_num,buf_size=struct.unpack(self.header_fmt,header)
        self.sock.send(self.ack)

        # 2. receive data
        data=bytearray(b'')
        for i in range(block_num):
            start_pos=i*buf_size
            end_pos=min((i+1)*buf_size,byte_num)
            block_data=self.sock.recv(end_pos-start_pos)
            data[start_pos:end_pos]=block_data

            self.sock.send(self.ack)
        data=bytes(data)
        return data


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
