from collections import deque, namedtuple
import random
import numpy as np
import torch

#namedtuple to store an experience tuple used to train a DQN
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

