import matplotlib.pyplot as plt
import gym
import random
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path

from dqn_agent import Agent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('CarRacing-v0')
env = gym.wrappers.Monitor(env, directory='recording', force=True)

env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space)

# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

# env.reset()
# frame = env.render(mode='rgb_array')

agent = Agent(action_size=4, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth', map_location=device))

max_steps = 1000

for i in range(10):
    state = env.reset()
    state = state[:84,:,:]
    acc_reward = 0.0
    for j in range(max_steps):
        action = agent.act(state)
        env_action = agent.convert_action(action).squeeze()
        env.render()
        state, reward, done, _ = env.step(env_action)
        acc_reward += reward
        state = state[:84,:,:]

        if done or acc_reward<0:
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True             
            break
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True             

env.monitor.close() 
env.close()
