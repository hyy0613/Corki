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