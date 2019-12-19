import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from model import QNetwork
import gym

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('CarRacing-v0')

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                            transforms.CenterCrop((84,84)),                 
                            transforms.ToTensor()])
        # Q-Network
        self.qnetwork_local = QNetwork(action_size, seed).to(device)
        self.qnetwork_target = QNetwork(action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def convert_action(self,action):
        env_action = env.action_space.sample()

        if action == 0:           #turn left
          env_action[0] = -1.0
          env_action[1] = 0.0
          env_action[2] = 0.0  
        elif action == 1:         #turn right
          env_action[0] = 1.0
          env_action[1] = 0.0
          env_action[2] = 0.0
        elif action == 2:         #accelerate
          env_action[0] = 0.0
          env_action[1] = 1.0
          env_action[2] = 0.0
        elif action == 3:         #brake
          env_action[0] = 0.0
          env_action[1] = 0.0
          env_action[2] = 1.0
        elif action == 4:         #do nothing
          env_action[0] = 0.0
          env_action[1] = 0.0
          env_action[2] = 0.0

        return env_action

    def step(self, state, action, reward, next_state, done):
        # print(state.shape, action.shape, reward, next_state.shape, done)
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # print("now learning")
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # print(state.shape)    
        state = self.transform(state).unsqueeze(0).to(device)
        # print(state.shape)    

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # print(action_values)
        # return action_values.cpu().data.numpy()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        
        # print(actions,actions.shape,self.qnetwork_local(states).shape )
        Q_expected = self.qnetwork_local(states).gather(1,actions.unsqueeze(1))
        # print(Q_targets.shape,Q_expected.shape)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        # print(loss)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                            transforms.CenterCrop((84,84)),                
                            transforms.ToTensor()])
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for e in experiences:
          if e is not None:
            
            # print(e.state.shape, e.action, e.reward, e.next_state.shape, e.done)
            
            states.append(self.transform(e.state).to(device))
            next_states.append(self.transform(e.next_state).to(device))
            actions.append(torch.tensor(np.array(e.action).astype(np.long)).to(device))
            rewards.append(torch.Tensor(np.array(e.reward)).to(device))
            dones.append(torch.from_numpy(np.array(e.done).astype(np.uint8)).float().to(device))

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        # print(actions,rewards,dones)

        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # print(actions,rewards,dones)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
