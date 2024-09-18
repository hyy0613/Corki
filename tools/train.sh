
#!/usr/bin/env bash
###
 # @Author: 颜峰 && bphengyan@163.com
 # @Date: 2023-05-19 17:19:11
 # @LastEditors: 颜峰 && bphengyan@163.com
 # @LastEditTime: 2023-05-22 09:54:42
 # @FilePath: /CO-MOT/tools/train.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
### 
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

# 打印所有指令
set -x

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/RoboFlamingo_ubuntu/

export COPPELIASIM_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RLBench/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export QT_DEBUG_PLUGINS=1

sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

sudo apt-get update -y
sudo apt-get install -y  libegl1-mesa libegl1-mesa-dev

sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get -y install meson
sudo apt-get -y build-dep mesa

sudo apt-get -y install freeglut3
sudo apt-get -y install freeglut3-dev

sudo apt-get install -y libgl1-mesa-dri

sudo apt update -y
sudo apt install -y xvfb
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99

sudo apt-get install -y libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev libqt5svg5-dev
# echo "keyboard-configuration  keyboard-configuration/layout select English (US)" | sudo debconf-set-selections
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y xorg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/dri
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/RoboFlamingo_ubuntu/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libxkbcommon-x11.so.0


PY_ARGS=${@:2} # 第2个输入参数后边的值

# 脚本运行失败，报错
set -o pipefail
#sed -e  ：直接在指令列模式上進行 sed 的動作編輯；
OUTPUT_BASE=$(echo $1 | sed -e "s/robot_flamingo\/configs/exps/g" | sed -e "s/.args$//g")
OUTPUT_BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/RoboFlamingo-origin/$OUTPUT_BASE
mkdir -p $OUTPUT_BASE

cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import tools.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import tools.json_parser as json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"
PYTHONPATH=$PYTHONPATH:../ \
# python tools/run_net.py \
#    --num_shards 8 \
#    --shard_id $node_rank \
#    --dist_url $dist_url \
#    --cfg configs/verb/MVIT_B_32x2_CONV.yaml

MASTER_ADDR=${MASTER_ADDR:-$master_addr}
MASTER_PORT=${MASTER_PORT:-$master_port}
NODE_RANK=${NODE_RANK:-$node_rank}
# let "NNODES=GPUS/GPUS_PER_NODE"

NODE_NUM=${#worker_strs[@]}  
echo "node num is $NODE_NUM"

if ((NODE_RANK == 0)); then
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    OUTPUT_DIR=$OUTPUT_BASE/run$RUN
    mkdir $OUTPUT_DIR && break
  done

  # clean up *.pyc files
  rmpyc() {
    rm -rf $(find -name __pycache__)
    rm -rf $(find -name "*.pyc")
  }

  # run backup
  echo "Backing up to log dir: $OUTPUT_DIR"
  rmpyc && cp -r robot_flamingo $1 $OUTPUT_DIR
  echo " ...Done"

  # tar src to avoid future editing
  cleanup() {
    echo "Packing source code"
    rmpyc
    # tar -zcf models datasets util main.py engine.py eval.py submit.py --remove-files
    echo " ...Done"
  }

  pushd $OUTPUT_DIR
  trap cleanup EXIT

  # log git status
  echo "Logging git status"
  git status > git_status
  git rev-parse HEAD > git_tag
  git diff > git_diff
  echo $PY_ARGS > desc
  echo " ...Done"

else
  # 3 minutes
  sleep 180
  for RUN in $(seq 100); do
    ls $OUTPUT_BASE | grep run$RUN && continue
    let "ITERRUN=$RUN-1"
    OUTPUT_DIR=$OUTPUT_BASE/run$ITERRUN
    break
  done
fi

args=$(cat $1)

# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 robot_flamingo/train/train_calvin.py
python -m torch.distributed.launch --nproc_per_node=8 --nnodes ${NODE_NUM} --node_rank ${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port 29502 --use_env robot_flamingo/train/train_calvin.py ${args} --run_name $OUTPUT_DIR  |& tee -a $OUTPUT_DIR/output.log
