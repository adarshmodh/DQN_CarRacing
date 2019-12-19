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


# data_folder = Path("/content/gdrive/My Drive/CIS680_2019/Project")

env = gym.make('CarRacing-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space)


from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

env.reset()
frame = env.render(mode='rgb_array')

agent = Agent(action_size=5, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = state[:84,:,:]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_action = agent.convert_action(action).squeeze()
            next_state, reward, done, _ = env.step(env_action)
            next_state = next_state[:84,:,:]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), data_folder /'checkpoint.pth')
            break
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
