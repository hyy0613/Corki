
#!/usr/bin/env bash

set -x

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/RoboFlamingo_ubuntu/


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/RoboFlamingo_ubuntu/lib


set -o pipefail

OUTPUT_BASE=$(echo $1 | sed -e "s/robot_flamingo\/configs/exps/g" | sed -e "s/.args$//g")
OUTPUT_BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/RoboFlamingo-origin/$OUTPUT_BASE
mkdir -p $OUTPUT_BASE

args=$(cat $1)

torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 robot_flamingo/train/train_calvin.py  ${args} --run_name $OUTPUT_BASE  |& tee -a $OUTPUT_BASE/output.log
