#!/bin/bash

function rand(){  
    min=$1  
    max=$(($2-$min+1))  
    num=$(($RANDOM+1000000000))  
    echo $(($num%$max+$min))  
}

DATASET=MNIST
# DATASET=SVHN
# DATASET=CIFAR10

if [ ${DATASET} == 'MNIST' ]; 
	then 
		g="10 60 100 200 300 500 1000 6000" # some dataset may not have 6000 samples or over than 6000 samples.
		OUT_CLASS=10
        state_path="$./EXP_RECORD/MNIST/snapshots/state_gen20"

	elif [ ${DATASET} == 'SVHN' ]; 
	then 
		g="10 60 100 200 300 500 1000 4000 14000" # some dataset may not have 6000 samples or over than 6000 samples.
		OUT_CLASS=10
        state_path="./EXP_RECORD/SVHN/snapshots/state_gen20"
	
    elif [ ${DATASET} == 'CIFAR10' ]; 
	then 
		g="10 60 100 200 300 500 1000 5000" # some dataset may not have 6000 samples or over than 6000 samples.
		OUT_CLASS=10
        state_path="./EXP_RECORD/CIFAR10/snapshots/state_gen20"

	elif [ ${DATASET} == 'STL10' ]; 
	then 
		g="1 6 10 20 30 50 100 500" # some dataset may not have 6000 samples or over than 6000 samples.
		OUT_CLASS=10
        state_path="./EXP_RECORD/STL10/snapshots/state_gen20"

	else
		echo "unsuported dataset --> ${DATASET}"
fi

log_dir=${state_path//snapshots/single_test} 
if [ ! -d ${log_dir} ]; then
    mkdir -p ${log_dir}
fi 

# unsupervsied training and save weights.
log_path=${log_dir}/unsup_all.log
python3 single_train.py \
   -state_path ${state_path} \
   -out_cls_num ${OUT_CLASS} \
   -supervised_train_epoch 100 \
   -unsupervised_train_epoch 400 \
   -dataset ${DATASET} \
   -cuda_did 0 \
   -single_phase unsup\
   2>&1 > "${log_path}" 

# semi for EvolveVAE...
for num_per_cls in ${g}
do
	for i in {1..5}
	do 
	    rand_seed=$(od -An -N1 -i /dev/random)
	    log_path=${log_dir}/semi-${num_per_cls}/sup_seed${rand_seed// /}.log
	    if [ ! -d $(dirname ${log_path}) ]; then
	        mkdir -p $(dirname ${log_path})
	    fi 
	    python3 single_train.py \
	        -state_path ${state_path} \
	        -out_cls_num ${OUT_CLASS} \
	        -supervised_train_epoch 100 \
	        -unsupervised_train_epoch 400 \
	        -dataset ${DATASET} \
	        -cuda_did $(rand 0 1) \
	        -single_phase sup\
	        -num_per_cls ${num_per_cls}\
	        -rand_seed ${rand_seed} \
	        2>&1 > "${log_path}" &
	done
    wait
done



