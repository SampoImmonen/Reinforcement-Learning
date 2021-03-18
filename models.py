import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    """
    Implements noisy linear layer. 
    taken from: https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/lib/dqn_model.py
    
    """
    def __init__(self, in_features, out_features, sigma_init=0.018, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)

    
class BasicConvNet(nn.Module):
    """
    A basic convolutional network to use in DQN
    """
    
    def __init__(self, input_shape, output_shape):
        super(BasicConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
    
        out_size = self._get_conv_out(input_shape)
    
        self.classifier = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )
    
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
class NoisyConvNet(nn.Module):
    """
    A convolutional network to use in DQN
    with noisy linear layers
    """
    
    def __init__(self, input_shape, output_shape):
        super(NoisyConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
    
        out_size = self._get_conv_out(input_shape)
    
        self.classifier = nn.Sequential(
            NoisyLinear(out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, output_shape)
        )
    
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
    
class DuelingNoisyConvNet(nn.Module):
    """
    A convolutional network to use in DQN
    with noisy linear layers
    """
    
    def __init__(self, input_shape, output_shape):
        super(DuelingNoisyConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
    
        out_size = self._get_conv_out(input_shape)
    
        self.adv = nn.Sequential(
            NoisyLinear(out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, output_shape)
        )
        
        self.val = nn.Sequential(
            NoisyLinear(out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
    
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        val = self.val(x)
        adv = self.adv(x)
        return val+adv-adv.mean()