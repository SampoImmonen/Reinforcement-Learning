import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""

TODO:
    better unified interface for policies
"""


class SamplingPolicy:
    """
    sampling policy for discete actions
    with a shared net for policy and value
    """
    def __init__(self, net):
        
        self.net = net
        
    @torch.no_grad()
    def get_action(self, state):
        """
        get sampled action from state
        currently supports only single action at a time
        """
        
        logits, _ = self.net(state)
        output_dim = logits.shape[1]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        #print(probs)
        return np.random.choice(output_dim, p=probs[0])

    def __call__(self, state):
        return self.net(state)


class SamplingSinglePolicy(SamplingPolicy):
    """
    Sampling policy for discrete actions for a policy net
    """
    def __init__(self, net):
        super().__init__(net)
        
    @torch.no_grad()
    def get_action(self, state):
        logits = self.net(state)
        output_dim = logits.shape[1]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        #print(probs)
        return np.random.choice(output_dim, p=probs[0])


class NormalSamplingPolicy:
    """
    Sampling policy for continuous actions sampled from a normal distribution
    """
    
    def __init__(self, net : nn.Module , clip_range : int = None):
        
        self.net = net
        self.clip_range = clip_range
        
    @torch.no_grad()
    def get_action(self, state):
        """
        returns sampled action
        """
        mean, var = self.net(state)
        std = torch.sqrt(var)
        
        action = torch.normal(mean, std)
        if self.clip_range != None:
            action = torch.clip(action, -self.clip_range, self.clip_range)
        return action.cpu().numpy().squeeze(axis=0)

    @torch.no_grad()
    def mean_action(self, state):
        """
        returns means as action
        """
        mean, _ = self.net(state)
        act = mean.squeeze(0).cpu().numpy()
        if self.clip_range!=None:
            act = np.clip(act, -self.clip_range, self.clip_range)

        return act



class ArgMaxPolicy:

    """
    simple argmax policy
    """

    def __init__(self, net):
        pass

    def get_action(self, state):
        pass

