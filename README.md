# bot-pysc2
work is based on this DQN example from openai: 

* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


## using python 3
you need the ssl system packages because IOT requires ssl

```
brew install python@3.8.0
pyenv install 3.8.0
pyenv virtualenv 3.8.0 bot-sc2-gym
pyenv local bot-sc2-gym
python3 -m pip install --upgrade pip
```

# install AdeptRL
```
git clone git@github.com:heronsystems/adeptRL.git
cd adeptRL/
pip install .
```

# rquirements installs (including those for AdeptRL)
https://github.com/heronsystems/adeptRL

```
python3 -m pip install -r requirements.txt
```


## pygame 1.9.6
`python3 -m pip install https://github.com/pygame/pygame/archive/1.9.6.zip`

## runing the move to beacon simulation
`python runner_ple.py`

docker run --gpus all --shm-size=1g --ulimit memlock=-1     --ulimit stack=67108864 -it --rm     -v $(pwd)/workspace:/workspace bot-sc2-gym:latest python run_adept.py --env PongNoFrameskip-v4 --logdir /workspace/adept

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace -v $(pwd)/sc2:/sc2 bot-sc2-gym:latest python sc2dnn.py


docker build . -t bot-sc2-gym:latest

docker run --cap-add=SYS_PTRACE --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace -v $(pwd)/sc2:/sc2 bot-sc2-gym:latest python sc2dnn.py

docker run -p 2000:2000 -p 6000:6000 --cap-add=SYS_PTRACE --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace -v $(pwd)/sc2:/sc2 bot-sc2-gym:latest gdb -ex r --args python sc2dnn.py


sudo apt-get purge nvidia*
sudo apt-get purge --auto-remove nvidia-cuda-toolkit

nvidia-dkms-460
cuda-drivers-460
cuda-drivers
cuda-runtime-11-2
nvidia-driver-460
cuda-11-2
cuda-demo-suite-11-2
cuda

sudo ubuntu-drivers autoinstall
sudo apt-get install cuda

# Nvidia Docker
sudo apt install curl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

gdb -ex r --args python sc2dnn.py <arguments>

