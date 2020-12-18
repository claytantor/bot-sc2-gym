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

## runing the move to beacon simulation
`python runner_ple.py`


