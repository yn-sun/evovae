# EvoVAE

# 1. Environmental requirements
* Python version 3.6.9
* PyTorch version >= 1.5.0
* NVIDIA driver version <= 440


# 2. Dependencies Installation
## 2.1 Modify the server configuration file "servers.conf", for example:
```
IP1:user1:passwd1:0|1|2|3
IP2:user2:passwd2:0|1
```
here "IP1" means the ip address or the domain of the remote host, the `user` denotes a valid username of the remote host,
the `passwd1` is the corresponding password, `0|1|2|3` refers to the utilized GPU ids. You must modify this configuration customarily.

# 2.2 install the dependecies in each remote host in the "servers.conf"
### 2.2.1 python libraries
```
pip install -r requirements.txt
```
some libraries with erors can be installed separately according to the reporeted error. 

## 2.2.2 cli dependencies
```
sudo apt update && sudo apt install sshpass -y
```

## 2.2.3 Confirm SSH
It is better to confirm that `sshpass` command works well when you first connect to an unknown host.
```
ssh user@remote_ip
sshpass -p passwd ssh user@remote_ip ls
```

## 2.2.4 Confirm dependencies
```
python dnn_server.py
```

# 3. Download the datasets
1. Download by torchvision, here take MNIST as an example:
```
MNIST('/home/user1/dataset',download=True)
```
`user1` must be replaced by the valid useranme stored in the "servers.conf" file

2. The datasets could also be copied from another computer:
```
scp -r xxx@xxx:~/dataset ~/
```

# 4. Search the architectures
4.1 Specified the python executable path in `dist_train.sh` file, e.g.:
```
PYTHON_PATH="/usr/bin/python"
```
4.2 Add the following script to the tail of the `dist_train.sh`:

1) For MNIST:
```
baseline_workspace='./EXP_RECORD/MNIST'
if [ ! -d $baseline_workspace ]
then
	mkdir $baseline_workspace
fi
nohup python $python_script -dist_train -servers="${hosts}" -pop_size=20 -generations=20 -workspace=$baseline_workspace -max_jobs=4 -min_mem 2000 -gpu_scale 4 -wait_gpu_change_delay 20 -supervised_train_epoch 40 -unsupervised_train_epoch 20 -rand_seed=12345 -dataset=MNIST > $baseline_workspace/main.log 2>&1 & 
```

2) For CIFAR10:
```
baseline_workspace='./EXP_RECORD/CIFAR-10'
if [ ! -d $baseline_workspace ]
then
    mkdir $baseline_workspace
fi
nohup python $python_script -dist_train -servers="${hosts}" -pop_size=20 -generations=20 -workspace=$baseline_workspace -max_jobs=1 -min_mem 8000 -gpu_scale 1 -wait_gpu_change_delay 20 -supervised_train_epoch 40 -unsupervised_train_epoch 20 -rand_seed=12345 -dataset=CIFAR10 > $baseline_workspace/main.log 2>&1 & 
```
3) For SVHN:
```
baseline_workspace='../EXP_RECORD/SVHN'
if [ ! -d $baseline_workspace ]
then
    mkdir $baseline_workspace
fi
nohup python $python_script -dist_train -servers="${hosts}" -pop_size=20 -generations=20 -workspace=$baseline_workspace -max_jobs=4 -min_mem 2000 -gpu_scale 4 -wait_gpu_change_delay 20 -supervised_train_epoch 40 -unsupervised_train_epoch 20 -rand_seed=12345 -dataset=SVHN > $baseline_workspace/main.log 2>&1 & 
```

4.3 Execute the shell 
```
./dist_train.sh no
```

4.4 Terminate the training procedure:
```
./dist_train.sh yes
```
# 5. Train the final individial
Modify the `single_test.sh` file according to the directory which stored the information of searched architectures:
```
...
DATASET=MNIST
...
state_path="$./EXP_RECORD/MNIST/snapshots/state_gen20"
...
```

Execute the shell
```
./single_test.sh
```
This script will automatically conduct the Unsupervised pretraining and Supervised training