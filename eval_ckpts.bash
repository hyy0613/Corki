#!/bin/bash
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/RoboFlamingo_ubuntu/

# export http_proxy=http://10.70.10.73:8412   
# export https_proxy=http://10.70.10.73:8412  

#Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

sudo apt-get update -y -qq
sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev

sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get -y install meson
sudo apt-get -y build-dep mesa

sudo apt-get -y install freeglut3
sudo apt-get -y install freeglut3-dev

sudo apt update -y
sudo apt install -y xvfb


sudo apt-get install -y --reinstall libgl1-mesa-dri


Xvfb :99 -screen 0 1024x768x16 &

export DISPLAY=:99
export PYTHONPATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/RoboFlamingo-origin
export EVALUTION_ROOT=$(pwd)

python eval_ckpts.py