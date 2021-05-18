# /usr/bin/bash

##! @TODO: distributed model training. 
##! @VERSION: 1.0 
##! @AUTHOR: Chen Xiangru

# GLOBAL VARIABLES
CLOSE_SERVER=$1
if [ "$CLOSE_SERVER" != "yes" ] && [ "$CLOSE_SERVER" != 'no' ]
then
	echo "ERROR: PARAM \`CLOSE_SERVER\` must be \`yes\` or \`no\`"
	exit
fi

REAL_DIR=$(cd "$(dirname $0)"; pwd)
REAL_PATH="${REAL_DIR}/$0"

SERVER_CONFIG="servers.conf"

TARGET_DIR='.dnn_tmp'

PYTHON_PATH="/usr/bin/python3"

################################
# gather all the files' md5sum #
################################
check_md5() {
	local dir_name="$*"
	find "${dir_name}" -maxdepth 1  -name "[!.][!_]*" | while read -r file  
	do
		if [ -d "$file" ]
		then
			#echo -e "\"$file\" is directory \n"
			:
		elif [ -f "$file" ]
		then
			local md5_val=(`md5sum "${file}"`)
			local md5_val=${md5_val[0]}
			# org_md5_list=(${org_md5_list[@]} "${md5_val}")
			echo [\"${file}\"]=\"$md5_val\"
		fi
	done
	# echo ${org_md5_list[@]}
}

#################################################################
# utility functions
################################################################

# dir exists
dir_exists() {
	local dir_name="$*"
	if [ -d "${dir_name}" ]
	then
		echo yes
	else
		echo no
	fi
}

# create dir
create_dir() {
	local dir_name="$*"
	if [ ! -d "$dir_name" ]
	then
		mkdir "$dir_name"
	fi
}

# remove dir
rm_dir() {
	local dir_name="$*"
	if [ -d "$dir_name"]
	then
		rm -rf "$dir_name"
	fi
}

# check value in a particular array
valinarr() {
	if [ ${#@} -le 1 ]
	then
		return -1
	fi

	for elm in "${@:2}" 
	do
		if [ "$elm" == "$1" ]
		then
			return 0
		fi
	done

	return -1
}

# wether the array contains a particular element
# i.e. if array_contains "${x[*]}" 2; then echo "find"; fi;
array_contains() {
	local array=($1)
	for elm in ${array[@]}
	do
		if [ "$elm" == "$2"  ]
		then
			return 0
		fi
	done
	return -1
}

# execute the python
exec_python() {
	local python_path="${*:1:1}"
	local script_path="${*:2}"
	local script_name=`basename "${script_path}"`
	local dir_path=`dirname "${script_path}"`
	local logger_path="${dir_path}/${script_name}.out"
	
	cd "$dir_path"
	nohup $python_path "${script_path}" > "${logger_path}"  2>&1 > "${script_name}.out"  &
	#$python_path "${script_path}" 
}

# execute ps 
exec_server_pid() {
	ps -aux | grep "${*// /\ }" |  grep -v "grep" | awk '{print $2}'
}

# execute kill process
exec_kill_pid() {
	local pid="$1"
	if [ "$pid" != "" ]
	then
		kill "$pid"
	fi
	
}


####################
# remote execution #
####################
remote_exec() {
	if [ $# -le 4 ]
	then
		echo "ERROR: remote_exec [ip] [user] [passwd] [func_name] [param1] [param2]..." 1>&2
		return -1
	fi
	
	local ip_=$1
	local user_=$2
	local token_=$3
	local func_name=$4
	local params=($@)
	local params=(${params[@]:4})

	# echo $ip_,$user_,$token_,$func_name,"${params[@]}"

	local func_declare=`typeset -f $func_name`
	local exe_str="${func_declare}; ${func_name} ""${params[@]}"";"
	sshpass -p ${token_} ssh ${user_}@${ip_} "${exe_str}" 2>&1
}

synchronized_files() {
	local ip_=$1
	local user_=$2
	local token_=$3
	local folder1="$4"
	local folder2="$5"
	declare -A dict_md5file 
	declare -A rdict_md5file


	eval dict_md5file=(`check_md5 $folder1`)


	if [ $(remote_exec ${ip_} $user_ $token_ dir_exists "$folder2") == "yes" ]
	then
		:
		# echo "remote dir \"$5\"  exist!"
	else
		echo "${ip_}: remote dir \"$folder2\" dose not exists"
		echo "${ip_}: creating..."
		remote_exec ${ip_} ${user_} ${token_} create_dir "$folder2"
	fi

	# access the remote file list
	eval rdict_md5file=(`remote_exec $ip_ $user_ $token_ check_md5 "$folder2" `)
	
	local rfile_list
	for fname in "${!rdict_md5file[@]}"
	do
		local bname=$(basename "${fname}")
		rfile_list+=("${bname}")
	done

	for file in "${!dict_md5file[@]}"
	do
		local trans_flag=0
		local bname=`basename "$file"`
		local oppfile="$folder2/$bname"
		if valinarr "$bname" "${rfile_list[@]}"  
		then
			# echo "\"${bname}\" dose exist in the remote server"
			if [ ${dict_md5file["${file}"]} == ${rdict_md5file["${oppfile}"]} ]
			then
				:
				# echo "file is the same"
			else
				:
				echo "${ip_}: file \"${oppfile}\" has changed"
				local trans_flag=1
			fi
		else
			echo "${ip_}: \"${oppfile}\" dose not exists in the remote server"
			echo "${ip_}: transmition begins"
			local trans_flag=1
			
		fi
		if [ $trans_flag -eq 1 ]
		then

			{
				# echo "target dir: \"${5}\""
				sshpass -p ${token_} scp "${file}" ${user_}@${ip_}:"${folder2// /\\ }"  
			
				echo "${ip_}: \"${oppfile}\" transmition completed"
			}&
		fi
	done
	wait
	# echo "file checking  passed"
	
	# check the directory
	# echo "NOTE THAT"`find "$4" -maxdepth 1 -type d -name "[!.][!_]*"`
	# exit
	find "$folder1" -maxdepth 1 -type d -name "[!.][!_]*" | while read -r folder
	do
		# echo NOTE THAT$folder
		local lfolder="${folder}"
		local bfolder=`basename "${folder}"`
		local nfolder="${folder2}/${bfolder}"
		if [ "$folder1" != "${lfolder}" ]
		then
			# :
			synchronized_files ${ip_} ${user_} ${token_} "${lfolder}" "${nfolder}" &
		fi
	done
}

server_control() {
	local ip=$1
	local user=$2
	local token=$3
	local stop_server=$4
	local remote_dir="/home/${user}/${TARGET_DIR}"

	local check_pid=$(remote_exec ${ip} ${user} ${token} exec_server_pid "python ${remote_dir}/dnn_server.py")
	if [ "$check_pid" != "" ]
	then
		echo "${ip}: find the server pid: ${check_pid}"
		if [ ${stop_server} == "yes" ]
		then
			ri=("${check_pid}")
			for real_pid in ${ri[@]}
			do
			{	
				echo "${ip}: kill the pid: ${real_pid}"
				remote_exec ${ip} ${user} ${token} exec_kill_pid "${real_pid}"
			}&
			done
		fi
	else
		echo "${ip}: server is not running"
		# local python_path="/home/${user}/workspace/my_python3_env/bin/python"
		local python_path=${PYTHON_PATH}
		if [ ${stop_server} == 'no' ]
		then	
			remote_exec ${ip} ${user} ${token} exec_python "${python_path}" "${remote_dir}/dnn_server.py"
			local check_pid=$(remote_exec ${ip} ${user} ${token} exec_server_pid "python ${remote_dir}/dnn_server.py")
			if [ "$check_pid" != "" ]
			then
				echo "${ip}: start server successed"
			else
				echo "${ip}: ERROR: failed to start the server"
			fi
		fi
	fi
}


#######################################
# reserved loop body for future usage #
#######################################

for line in `cat $SERVER_CONFIG`
do
	ip_u_p=(${line//:/ })
	ip=${ip_u_p[0]}
	user=${ip_u_p[1]}
	token=${ip_u_p[2]}
	remote_dir="/home/${user}/${TARGET_DIR}"

	if [ $CLOSE_SERVER == 'yes' ]
	then
		server_control $ip $user $token 'yes'
		continue
	fi
	{	
		# 1. synchronize the files
		synchronized_files $ip $user $token "$REAL_DIR" "${remote_dir}" 
		#remote_exec localhost cxr 123 dir_exists "/home/cxr/.dnn_tmp/test space"
		
	}&
done
wait

hosts=(`cat $SERVER_CONFIG`)
hosts="${hosts[*]}"
hosts="${hosts// /,}"

if [ $CLOSE_SERVER == 'yes' ]
then
	exit
fi

echo "files synchronization ok!"

for line in `cat $SERVER_CONFIG`
do
	{	
	ip_u_p=(${line//:/ })
	ip=${ip_u_p[0]}
	user=${ip_u_p[1]}
	token=${ip_u_p[2]}
	remote_dir="/home/${user}/${TARGET_DIR}"
		
		# 2. start server on the remote machine
		#exec_python "$user" "${remote_dir}/dnn_server.py"
		server_control $ip $user $token 'no' &
		
	}
done
wait

echo "waiting server up..."
sleep 1
for line in `cat $SERVER_CONFIG`
do
	{	
	ip_u_p=(${line//:/ })
	ip=${ip_u_p[0]}
	user=${ip_u_p[1]}
	token=${ip_u_p[2]}
	remote_dir="/home/${user}/${TARGET_DIR}"
	
		# 3. run the client script...
		python_script="$REAL_DIR/dnn_client.py"
		#python "$python_script" -ip=$ip -servers="${hosts}"  & 
		#python "$python_script" -ip=$ip &
		#python "$python_script" -ip=$ip &
		
		wait
	}&
done
wait

python_script="$REAL_DIR/main.py"

# MNIST
baseline_workspace='./EXP_RECORD/MNIST'
if [ ! -d $baseline_workspace ]
then
	mkdir $baseline_workspace
fi
nohup python $python_script -dist_train -servers="${hosts}" -pop_size=20 -generations=20 -workspace=$baseline_workspace -max_jobs=4 -min_mem 2000 -gpu_scale 4 -wait_gpu_change_delay 20 -supervised_train_epoch 40 -unsupervised_train_epoch 20 -rand_seed=12345 -dataset=MNIST > $baseline_workspace/main.log 2>&1 & 

# CIFAR-10
# baseline_workspace='./EXP_RECORD/CIFAR-10'
# if [ ! -d $baseline_workspace ]
# then
# 	mkdir $baseline_workspace
# fi
# nohup python $python_script -dist_train -servers="${hosts}" -pop_size=20 -generations=20 -workspace=$baseline_workspace -max_jobs=1 -min_mem 8000 -gpu_scale 1 -wait_gpu_change_delay 20 -supervised_train_epoch 40 -unsupervised_train_epoch 20 -rand_seed=12345 -dataset=CIFAR10 > $baseline_workspace/main.log 2>&1 & 

# # SVHN
# baseline_workspace='../EXP_RECORD/SVHN'
# if [ ! -d $baseline_workspace ]
# then
# 	mkdir $baseline_workspace
# fi
# nohup python $python_script -dist_train -servers="${hosts}" -pop_size=20 -generations=20 -workspace=$baseline_workspace -max_jobs=4 -min_mem 2000 -gpu_scale 4 -wait_gpu_change_delay 20 -supervised_train_epoch 40 -unsupervised_train_epoch 20 -rand_seed=12345 -dataset=SVHN > $baseline_workspace/main.log 2>&1 & 