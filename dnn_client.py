# coding=utf-8

import numpy as np
from numpy import array
import random

import multiprocessing,subprocess,os,sys,time,threading
import socket
import argparse

from cs_utils import NetMessage,GPUInfoHelper,ProcShareMessage,SocketBuffer

import utils 
logger=utils.get_logger(__name__)
np.set_printoptions(threshold=sys.maxsize)

class DNNClient(object):
    def __init__(self, HOST, PORT, BUFSIZ, server_list, net_id, indi,super_train_epoch,unsuper_train_epoch,GA_MODEL):
        self.HOST = HOST
        self.PORT = PORT
        self.BUFSIZ = BUFSIZ
        self.SOCKADDR = (HOST, int(PORT))
        self.server_list = server_list
        # print(self.server_list)
        self.net_id=net_id
        self.indi=indi
        self.super_train_epoch=super_train_epoch
        self.unsuper_train_epoch=unsuper_train_epoch
        self.GA_MODEL=GA_MODEL

        self.gih=GPUInfoHelper(server_list)
        self.client_sock = None
        self.did=None

        self.result=ProcShareMessage(64)
        self.stop_state=ProcShareMessage(64)
    
    def urgent_stop(self):
        """
        Stop the client urgently
        """
        self.stop_state.print('STOP')
    
    def _need_stop(self):
        """
        Scan the stop state
        Returns
        -------
        - stop  (bool)
        """
        valid,data=self.stop_state.unpack(clear=True)
        if valid and data == 'STOP':
            return True 
        return False 
        
    def get_did(self):
        """
        Get the device id when runing
        Returns
        -------
        - did   (int): the nvidia device id
        """
        
        best_gpu_info=self.gih.get_adequate_gpu()[0]
        gpu_ip=best_gpu_info['ip']
        did=best_gpu_info['gpu_slot']
        self.SOCKADDR = (gpu_ip, int(self.PORT))
        logger.info(' '.join(['net_id:',str(self.net_id),"get adequate ip:",str(gpu_ip),'gpu slot:',str(did)])) 

        self.did=did
        return self.did
    
    def get_result(self):
        valid,data=self.result.unpack(clear=True)
        if valid:
            data=float(data)
        else:
            data=float('-inf')
        return data
        
    def run(self,ip,port,did):
        """
        run the client model
        Params
        - ip
        - port 
        - did 
        ------
        """
        self.SOCKADDR = (ip, int(port))
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.connect(self.SOCKADDR)
        self.client_sock=SocketBuffer(self.client_sock)
        try:
            init_payload={
                'id':self.net_id,
                'indi':self.indi,
                'did':did,
                'sup_train_epochs':self.super_train_epoch,
                'unsup_train_epochs':self.unsuper_train_epoch,
                'iters':self.GA_MODEL.iters,
                'generations':self.GA_MODEL.generations,
                'DATASET':self.GA_MODEL._CMD_ARGS.dataset
            }
            init_payload=str(init_payload)
            send_data = NetMessage.pack('START',init_payload)
            self.client_sock.send(send_data)
            while True:
                data = self.client_sock.recv(self.BUFSIZ)
                # print(data)
                if data is None or data == b'':
                    logger.error('get null message')
                    break 
                else:
                    dtype, recv_data = NetMessage.unpack(data)

                    if dtype == 'RESULT':
                        logger.info(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'net_id:',str(self.net_id),'ip:',ip,'port:',str(port),'did:',str(did),"get results:", recv_data]) )
                        logger.info(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'net_id:',str(self.net_id),"send `close` signal to server",'ip:',ip,'port:',str(port),'did:',str(did)]))
                        self.result.pack(True,recv_data)
                        # self.client_sock.send(NetMessage.pack('END',''))
                        break

                    elif dtype == 'LOGGER':
                        logger.info(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'net_id:',str(self.net_id),'ip:',ip,'port:',str(port),'did:',str(did),'receive the logger info:', recv_data]))
                        self.client_sock.send(NetMessage.pack('HEART',''))

                    elif dtype == 'FILE':
                        recv_file_name=recv_data['file_name']
                        recv_file_binary= recv_data['binary']

                        dir_name = self.GA_MODEL.workspace
                        dir_name = os.path.join(dir_name,'files','iter%d'%(self.GA_MODEL.iters+1),'net-id-%d'%(self.net_id) )
                        
                        if not os.path.exists(dir_name):
                            os.makedirs(dir_name)
                        file_path = os.path.join(dir_name,recv_file_name)
                        with open(file_path,'wb+') as f:
                            f.write(recv_file_binary)
                        
                        logger.info(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'net_id:',str(self.net_id),'ip:',ip,'port:',str(port),'did:',str(did),'receive the file transfer request, save to:', file_path]))
                        self.client_sock.send(NetMessage.pack('HEART',''))

                    elif dtype == 'HEART':
                        # logger.info('heart')
                        if self._need_stop():
                            # self.client_sock.send(NetMessage.pack('END',''))
                            break
                        else:
                            self.client_sock.send(NetMessage.pack('HEART',''))
                            time.sleep(1e-1)

                    elif dtype == 'END':
                        logger.info(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'net_id:',str(self.net_id),'ip:',ip,'port:',str(port),'did:',str(did),"Task on server is finised, break the loop"]))
                        break
                        
        except KeyboardInterrupt as e:
            # self.client_sock.send(NetMessage.pack('END',''))
            logger.error(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'ip:',ip,'port:',str(port),'did:',str(did),"Exited by user"]))
        finally:
            self.client_sock.close()
            logger.info(' '.join(['GA(ITER:%d/%d)'%(self.GA_MODEL.iters+1,self.GA_MODEL.generations),'ip:',ip,'port:',str(port),'did:',str(did),'socket in client has been closed']))

class DNNManager():
    def __init__(self,GA_MODEL,pop,super_train_epoch,unsuper_train_epoch,server_list,port,max_jobs=3,min_mem=3000,wait_gpu_change_delay=5,gpu_scale=3):
        """
        Params
        ------
        - GA_MODEL
        - pop
        - server_list
        - super_train_epoch
        - unsuper_train_epoch
        - port
        - max_jobs
        - min_mem
        - wait_gpu_change_delay
        - gpu_scale              (int): proportional between pool size and gpu number
        """
        self.GA_MODEL=GA_MODEL
        self.pop=pop
        self.server_list=server_list
        self.super_train_epoch=super_train_epoch
        self.unsuper_train_epoch=unsuper_train_epoch

        self.port=port
        self.GIH=GPUInfoHelper(server_list,max_jobs=max_jobs,min_mem=min_mem)
        self.pool_size=None
        self.proc_list=None
        self.client_list=None
        self.gpu_scale=gpu_scale
        self.wait_gpu_change_delay=wait_gpu_change_delay
    
    def run(self):
        """
        running wrapper
        Returns
        -------
        - fitness (np.array(float)[n]): the fitness of the pop
        """
        try:

            return self._running()
        
        except KeyboardInterrupt as ki:
            for client in self.client_list:
                if client:
                    client.urgent_stop()
            while self.count_alive(self.proc_list)>0:
                time.sleep(1e-3)

            raise ki 

    def _running(self):
        """
        run dnn manager to get the fitness
        Returns
        -------
        - fitness (np.array(float)[n]): the fitness of the pop
        """
        server_list=self.server_list
        pop=self.pop
        port=self.port
        GIH=self.GIH
        wait_gpu_change_delay=self.wait_gpu_change_delay
        cur_valid_gpu_num=len(GIH.get_adequate_gpu(block=True))
        self.pool_size=pool_size=int(cur_valid_gpu_num*self.gpu_scale)
        self.proc_list=proc_list=[None]*len(pop)
        self.client_list=client_list=[None]*len(pop)
        logger.info('nvidia-smi delay:%s | cur_valid_gpu_num:%d | gpu scale:%s | pool size:%d'%(self.wait_gpu_change_delay,cur_valid_gpu_num,self.gpu_scale,pool_size))


        for i,indi in enumerate(pop):
            client=DNNClient('localhost',port,4096,server_list,i,indi,self.super_train_epoch,self.unsuper_train_epoch,self.GA_MODEL)
            client_list[i]=client
            best_gpu_info=GIH.get_adequate_gpu(block=True)[0]
            gpu_ip=best_gpu_info['ip']
            did=best_gpu_info['gpu_slot']
            logger.info(' '.join(['net_id:',str(i),"get adequate ip:",gpu_ip,'port:',str(port),'gpu slot:',str(did)]))
            proc=multiprocessing.Process(target=client.run,args=(gpu_ip,port,did))
            proc_list[i]=proc
            proc.start()
            time.sleep(wait_gpu_change_delay)
            # time.sleep(20)
            while self.count_alive(proc_list)>=pool_size:
                time.sleep(1e-3)

        while self.count_alive(proc_list)>0:
            logger.info('waiting processing finish...: '+str(self.count_alive_list(proc_list)) )
            time.sleep(1e2)

        fitness = np.array([client.get_result() for client in client_list])
        return fitness

    def count_alive(self,proc_list):
        """
        Return the number of alive process
        """
        return len(self.count_alive_list(proc_list))
    
    def count_alive_list(self,proc_list):
        """
        Count the alive thread id
        Params
        ------
        - proc_list     (list)
        Returns
        -------
        - alive_list    (list)
        """
        alive_list=[]
        for i,proc in enumerate(proc_list):
            if proc is not None and proc.is_alive():
                alive_list+=[i]
        return alive_list

if __name__ == '__main__':
    pass 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-ip","--ip", help="ip address",type=str,default='localhost')
    # parser.add_argument("-port","--port", help="port",type=int,default=10007)
    # parser.add_argument("-buf","--buf", help="buffer size",type=int,default=4096)
    # parser.add_argument("-servers","--servers", help="server list",type=str,default='localhost:cxr:123:0|1')
    # args = parser.parse_args()

    # dnn_client = DNNClient(args.ip,args.port,args.buf,args.servers,int(np.random.rand()*1000),{'h':[{'type':'p','gene':np.random.rand(10)},{'type':'c','gene':np.random.randn(20000)}],'t':[{'type':'p','gene':np.random.rand(30)}]})
    # dnn_client.run(args.ip,args.port,0)
