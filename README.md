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


Exception has occurred: RuntimeError
mat1 and mat2 shapes cannot be multiplied (1x4096 and 2592x36)
  File "/Users/claytongraham/data/github/claytantor/bot-sc2-gym/agents/dqn4/__init__.py", line 115, in forward
    return self.head(x.view(x.size(0), -1))
  File "/Users/claytongraham/data/github/claytantor/bot-sc2-gym/agents/dqn4/__init__.py", line 323, in select_action_tensor
    return self.policy_net(state).max(1)[1].view(1, 1)
  File "/Users/claytongraham/data/github/claytantor/bot-sc2-gym/agents/dqn4/__init__.py", line 295, in step
    tensor_action = self.select_action_tensor(state)
  File "/Users/claytongraham/data/github/claytantor/bot-sc2-gym/sc2dnn.py", line 57, in main
    step_action_tensor, step_action_pysc2 = agent.step(timesteps[0])
  File "/Users/claytongraham/data/github/claytantor/bot-sc2-gym/sc2dnn.py", line 156, in <module>
    main(sys.argv[1:])


cartpole:
self.head = Linear(in_features=512, out_features=2, bias=True)
self.head.data.shape=torch.Size([1, 512])

x.view.shape = torch.Size([1, 512])



sc2:







