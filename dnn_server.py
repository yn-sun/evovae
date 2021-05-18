# coding=utf-8

import numpy as np
from numpy import array
import matplotlib
#matplotlib.use("Agg")
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

import sys,os
import socket
from time import ctime
import time
import threading
import multiprocessing

import argparse

from net import RunModel
from cs_utils import NetMessage, ProcShareMessage, SocketBuffer
import utils

from utils import PROC_HIDE_LOGGER


logger = utils.get_logger(__name__)
np.set_printoptions(threshold=sys.maxsize)

# def empty_func(i,out):
#     # time.sleep(1)
#     dir_path='proc_flag'
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
#     file_path='%s/%d.txt'%(dir_path,i)
#     with open(file_path,'a+') as f:
#         f.write('%d is finisehd\n'%(i))
#         f.flush()
#     out.print('0.23')
#     exit() 

class DNNServer(object):
    """
    DNN server class, accept the dnn training request and run the models
    Params
    ------
    - HOST      (str): host name
    - PORT      (int): port number
    - BUFSIZ    (int): network buffer size
    - MSG_SIZE  (int): shared process message buffer size
    """
    def __init__(self,HOST = '0.0.0.0', PORT = 10007, BUFSIZ = 40960, MSG_SIZE = 2048):
    
        self.HOST = HOST
        self.PORT = PORT
        self.BUFSIZ = BUFSIZ
        self.MSG_SIZE = MSG_SIZE
        self.ADDR = (HOST, PORT)
    
    @staticmethod
    def nn_proc_func(data,out,msg_buf):
        """
        Process function for neural network running
        Params
        ------
        - data      (str)                   : contain netid and individule's information
        - out       (Array('d', 1))         : the shared output container
        - msg_buf   (Array('d', MSG_SIZE))  : the shared message buffer
        """
        fitness=0

        # run the network model...
        try:
            data=eval(data)
            net_id=data['id']
            indi=data['indi']
            did=data['did']
            # print('params',net_id,indi,flush=True)
            # model = RunModel(net_id,indi,did,msg_buf)
            data['msg_buf']=msg_buf
            model = RunModel(**data)

            fitness = model.get_fitness()
            
            if not PROC_HIDE_LOGGER:
                logger.info('net-id: %d, subprocess finished' % (net_id))
        except Exception as e:
            if not PROC_HIDE_LOGGER:
                logger.error(str(e))
            msg_buf.pack(True,"ERROR: %s, indi: %s , net-id: %s"%(str(e), str(indi), str(net_id)))
            time.sleep(120) # wait
        finally:
            time.sleep(120) # wait for client to receive the msg.
            out.print(str(fitness))
    
    def sock_thread_func(self,sock, addr):
        """
        Thread function for socket operation
        Params
        ------
        - sock (socket object)
        - addr (socket address)
        Returns
        -------
        """
        if not PROC_HIDE_LOGGER:
            logger.info('%s :sock thread is ok'%(str(addr) ))
        proc=None

        try:

            while True:
                recv_data = sock.recv(self.BUFSIZ)
                if recv_data is None:
                    break
                # print(recv_data,flush=True)
                dtype, recv_data = NetMessage.unpack(recv_data)
                
                if dtype=='END':
                    if not PROC_HIDE_LOGGER:
                        logger.info(' '.join([str(addr),':','client send closed signal']))
                    if proc.is_alive():
                        proc.terminate()
                    break

                elif dtype=='HEART':
                    is_out,out_float=out.unpack(clear=False)
                    if is_out:
                        if proc.is_alive():
                            proc.terminate()
                        out_float = out_float['content']                    
                        ret_val = float(out_float)
                        
                        send_data = NetMessage.pack('RESULT', '%f' % (ret_val))            
                        sock.send(send_data) 
                        
                    else:
                        if proc.is_alive():
                            msg_valid, msg_str = msg_buf.unpack(clear=False)
                            if msg_valid:
                                msg_buf.clear()
                                sm_type = msg_str['msg_type']
                                # sm_content = msg_str['content']
                                
                                if sm_type == 'logger':
                                    send_data = NetMessage.pack('LOGGER', msg_str['content'])
                                elif sm_type == 'file':
                                    file_path = msg_str['file_path']
                                    file_name = os.path.basename(file_path)
                                    with open(file_path,'rb') as f:
                                        send_data = NetMessage.pack('FILE', {'file_name':file_name,'binary':f.read()})
                                else:
                                    send_data=''
                                # send info message back to the client
                                sock.send(send_data)
                            else:
                                sock.send(NetMessage.pack('HEART',''))
                                #time.sleep(1e-3)

                            # logger.info("Model process (pid: %d) is still running"%(proc.pid))
                        else:
                            # here are bugs to make the whole proc hangs when the dnn process was killed by accidently.
                            sock.send(NetMessage.pack('HEART',''))             
                    # else:
                    #     ret_val = out[0]
                    #     send_data = NetMessage.pack('RESULT', '%f' % (ret_val))            
                    #     sock.send(send_data) 

                elif dtype == 'START':
                    # run the network model
                    out = ProcShareMessage(self.MSG_SIZE)
                    msg_buf = ProcShareMessage(self.MSG_SIZE)

                    proc = multiprocessing.Process(target = DNNServer.nn_proc_func, args = (recv_data, out, msg_buf))

                    # data=eval(recv_data)
                    # net_id=data['id']
                    # indi=data['indi']
                    # did=data['did']
                    # proc = multiprocessing.Process(target = empty_func, args=(data['id'],out))

                    #proc.start();proc.join()
                    proc.start();

                    # launch the heart
                    sock.send(NetMessage.pack("HEART",''))
                    
        except BrokenPipeError as e:
            if not PROC_HIDE_LOGGER:
                logger.error('An error occurs when communicates with "%s": %s'%(addr, str(e)))
            if proc.is_alive():
                proc.terminate()

        finally:
            sock.close()
            if not PROC_HIDE_LOGGER:
                logger.info(' '.join([str(addr), ':', 'socket is already closed']))


    def start_new_thread(self,sock, addr):
        """
        Start a new thread to process the task
        Params
        ------
        - sock (object) : the socket object
        Returns
        -------
        """
        threading.Thread(target=lambda : self.sock_thread_func(sock, addr)).start()
    
    def run(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.ADDR)
        self.server_socket.listen(5)
        self.server_socket.setsockopt( socket.SOL_SOCKET, socket.SO_REUSEADDR, 1 )
        try:
            while True:
                logger.info('Server waiting for connection...')
                client_sock, addr = self.server_socket.accept()
                client_sock = SocketBuffer(client_sock)
                logger.info('Client connected from: ' + str(addr))
                self.start_new_thread(client_sock, addr)
        except KeyboardInterrupt as ki:
            logger.info('Server shutdown by peer')

        finally:
            self.server_socket.close()
            
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-ip","--ip", help="ip address",type=str,default='0.0.0.0')
    # parser.add_argument("-port","--port", help="port",type=int,default=10007)
    # parser.add_argument("-buf","--buf", help="buffer size",type=int,default=4096)
    # parser.add_argument("-msgbuf","--msgbuf", help="message buffer size",type=int,default=2048)
    # args = parser.parse_args()
    # dnn_server = DNNServer(args.ip,args.port,args.buf,args.msgbuf)
    # dnn_server.run()

    from utils import _args
    dnn_server = DNNServer(_args.server_ip,_args.port,_args.server_buf,_args.server_msgbuf)
    dnn_server.run()

