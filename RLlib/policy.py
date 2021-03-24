import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplingPolicy:
    
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
    
    def __init__(self, net):
        super().__init__(net)
        
    @torch.no_grad()
    def get_action(self, state):
        logits = self.net(state)
        output_dim = logits.shape[1]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        #print(probs)
        return np.random.choice(output_dim, p=probs[0])