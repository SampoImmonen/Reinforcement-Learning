from collections import deque, namedtuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

#namedtuples to store an experience tuple used to train a DQN
ActionPair = namedtuple('ActionPair', ('state', 'action', 'next_state', 'reward'))
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'isdone'))
PlayedEpisode = namedtuple('PlayedEpisode', ('reward', 'num_steps'))

class ReplayBuffer:
    """
    ReplayBuffer stores Experience and handles sampling a batch
    """
    def __init__(self, capacity):
        
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience):
        self.buffer.append(experience)
        
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size, as_tensor=False, device="cpu"):
        """
        sample a batch
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, isdones = zip(*batch)
        states, actions, next_states, rewards, isdones = np.stack(states), np.stack(actions), np.stack(next_states), np.stack(rewards), np.stack(isdones)
        if as_tensor:
            return (torch.tensor(states).to(device), torch.tensor(actions, dtype=torch.int64).to(device),
                   torch.tensor(next_states).to(device), torch.tensor(rewards).to(device),
                   torch.BoolTensor(isdones).to(device)
                   )
        return states, actions, next_states, rewards, isdones



class ExperienceSource:
    """
    Experience source used for training DQNs
    Very similar to PolicyExperienceSource used with A2C
    (These two should be morphed)
    """
    def __init__(self, env, n_steps=1, gamma=0.99, capacity=10000, device="cpu"):
        
        self.env = env
        self.state = self.env.reset()
        
        self.n_steps = n_steps
        self.gamma = gamma
            
        self.buffer = ReplayBuffer(capacity)
        
        self.device = device
        
        self.episode_steps = 0

        self.steps_done = 0
        self.episode_reward = 0

        
    def step(self, net):
        """
        takes a step in the environment and stores in the Replay Buffer
        returns episode reward is episode ends
        """

        state = self.state
        state_tensor  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            act = net(state_tensor).max(1)[1].item()
        
        self.episode_steps+=1
        obs, reward, isdone, _ = self.env.step(act)
        first_action = act
        total_reward = reward
        self.episode_reward+=reward

        #Loop to enroll bellman equation
        if (not isdone):
            for i in range(self.n_steps-1):
                
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    act = net(state_tensor).max(1)[1].item()
                obs, reward, isdone, _  = self.env.step(act)
                total_reward+=(self.gamma**(i+1))*reward

                self.episode_reward+=reward
                self.episode_steps+=1

                if isdone:
                    break

        exp = Experience(state, first_action, obs, total_reward, isdone)
        
        self.buffer.push(exp)  
        self.steps_done+=1
        
        if isdone:
            self.state = self.env.reset()
            episode_reward = self.episode_reward
            episode_steps = self.episode_steps

            self.episode_reward = 0
            self.episode_steps = 0

            return (episode_reward, episode_steps)
        
        self.state = obs
        
        return None
        
    def get_steps(self):
        return self.steps_done
        
            
    def sample(self, batch_size, as_tensor=False):
        """
        samples a batch from ReplayBuffer and transfers to device
        """
        states, acts, next_states, rewards, isdones = self.buffer.sample(batch_size, as_tensor)
        if as_tensor:
            return (states.to(self.device), acts.to(self.device),
                   next_states.to(self.device), rewards.to(self.device),
                   isdones.to(self.device)
                  )
        else:
            return (states, acts, next_states, rewards, isdones)


class ExperienceSourceForPolicy:
    """
    Experience source used with A2C. very Similar to Experience source used with dqn
    """
    def __init__(self, env, n_steps, gamma = 0.99, device="cpu"):
        
        self.env = env
        self.state = self.env.reset()
        
        self.episode_reward = 0
        self.episode_steps = 0
        
        self.n_steps = n_steps
        
        self.device = device
        self.steps_done = 0
        
        self.gamma = gamma
        
        
    @torch.no_grad()
    def step(self, policy):
        
        state = self.state
        obs_tens = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act = policy.get_action(obs_tens)
        
        obs, reward, isdone, _ = self.env.step(act)
        
        self.episode_steps +=1
        self.episode_reward+=reward
        
        first_action = act
        total_reward = reward
        
        if (not isdone):
            for i in range(self.n_steps-1):
                obs_tens = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                act = policy.get_action(obs_tens)
                obs, reward, isdone, _ = self.env.step(act)
                total_reward+=(self.gamma**(i+1))*reward
                self.episode_reward+=reward
                self.episode_steps+=1
                if isdone: 
                    break
                    
        exp = Experience(state, first_action, obs, total_reward, isdone)
        
        if isdone:
            self.state=self.env.reset()
            episode_reward = self.episode_reward
            episode_steps = self.episode_steps
            
            self.episode_steps = 0
            self.episode_reward = 0
                
            return exp, (episode_reward, episode_steps)


        self.state = obs
        
        return exp, None