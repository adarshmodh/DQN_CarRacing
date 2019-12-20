import gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision 

torch.manual_seed(1)

env = gym.make('CarRacing-v0')

print(env.action_space)
print(env.observation_space)

env.reset()

for _ in range(1500):
  env.render()

  action = env.action_space.sample()
  print(action)
  
  next_state, reward, done, _ = env.step(action)
  print(next_state[:84,:,:].shape, reward, done)

  plt.imshow(next_state[:84,:,:])
  plt.pause(0.0001)
	
  if done:
      break

plt.show()
env.close()

