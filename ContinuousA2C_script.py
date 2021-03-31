import gym
import math
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from RLlib.utils import ExperienceSourceForPolicy
from RLlib.models import ValueNet
from RLlib.actorcritic import get_batch_continuous

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

def test_policy(policy, env, count=10, device="cpu"):
    rewards = []

    for ep in range(count):
        obs = env.reset()
        ep_reward = 0
        while True:
            obs = torch.tensor(obs).unsqueeze(0)
            act = policy.mean_action(obs)
            obs, reward, isdone, _ = env.step(act)
            ep_reward+=reward
            if isdone:
                break

        rewards.append(ep_reward)
    return sum(rewards)/len(rewards)



class ContinuousPolicyNet(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=512):
        super(ContinuousPolicyNet, self).__init__()
        
        self.base = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ELU(),
                                  nn.Linear(hidden_size, hidden_size), nn.ELU())
        
        self.mean = nn.Sequential(nn.Linear(hidden_size, output_dim), nn.Tanh())
        self.var  = nn.Sequential(nn.Linear(hidden_size, output_dim), nn.Softplus())
        
    def forward(self, input):
        x = self.base(input)
        mean = self.mean(x)
        var = self.var(x)
        return mean, var
    
    
def gaussianlogprobs(mean, var, act):
    t1 = -((act-mean).pow(2))/(2*var.clamp(min=1e-3))
    t2 = -torch.sqrt(2*math.pi*var).log()
    return t1+t2
    
def gaussianentropy(var):
    """input shape (batch_size, act_size)"""
    return -((2*math.pi*var).log()+1)/2





"""
Script implementing A2C algorithm for LunarLander-v2 Environment

"""


if __name__ == "__main__":

    #Add CLI
    print("Initialzing paramaters")
    
    #basic parameters
    gamma = 0.99
    lr = 0.001
    beta = 0.0001
    batch_size = 32
    sync_interval = 50
    test_interval = 100
    reward_steps = 2
    clip_grad = 0.1
    
    #environment
    env = gym.make('LunarLanderContinuous-v2')
    
    #models
    policynet = ContinuousPolicyNet(8, 2)
    valuenet = ValueNet(8)
    targetvaluenet = ValueNet(8)
    targetvaluenet.load_state_dict(valuenet.state_dict())

    #policy and experience soure
    policy = NormalSamplingPolicy(policynet, 1)
    exp_source = ExperienceSourceForPolicy(env, n_steps = reward_steps)

    #Optimizers
    policyoptimizer = torch.optim.Adam(policynet.parameters(), lr=lr)
    valueoptimizer = torch.optim.Adam(valuenet.parameters(), lr=lr)
    
    episode_rewards = []
    episodes_done=0
    batch = []
    best = -200
    num_steps = 0

    start_time = time.time()
    print("start training")
    while True:
            exp, rew = exp_source.step(policy)
            batch.append(exp)
            if rew!=None:
                reward, steps = rew
                episode_rewards.append(reward)
                if len(episode_rewards)%10==0 and len(episode_rewards) > 0:
                    if sum(episode_rewards[-10:])/10 > best:
                        print(sum(episode_rewards[-10:])/10)
                        print(len(episode_rewards))
                        print(time.time()-start_time)
                        best = sum(episode_rewards[-10:])/10
                        print(35*'-')
                    if sum(episode_rewards[-10:])/10 > 199:
                        print("Solved")
                        break
                        
            if len(batch) == batch_size:
                
                num_steps+=1
                
                states, acts, rewards = get_batch_continuous(batch, targetvaluenet, n_steps=reward_steps)
                batch.clear()
                policyoptimizer.zero_grad()
                valueoptimizer.zero_grad()
                
                mean, var = policynet(states)
                value = valuenet(states)
                
                value_loss = F.mse_loss(value.squeeze(-1), rewards)
                
                log_probs = gaussianlogprobs(mean, var, acts)
                
                adv_v = rewards.unsqueeze(-1)-value.detach()
                
                #print(adv_v.shape)
                #print(log_probs.shape)
                policy_loss = -(adv_v*log_probs).mean()
                
                #policy_loss.backward(retain_graph=True)
                
                ent = gaussianentropy(var)
                entropy_loss = beta*ent.mean()
                
                p_loss = policy_loss+entropy_loss

                p_loss.backward()
                value_loss.backward()
                        
                nn.utils.clip_grad_norm_(policynet.parameters(), clip_grad)
                nn.utils.clip_grad_norm_(valuenet.parameters(), clip_grad)
                
                policyoptimizer.step()
                valueoptimizer.step()
                

                #optimizer.step()
                if num_steps%sync_interval == 0:
                    targetvaluenet.load_state_dict(valuenet.state_dict())
                
                if num_steps%test_interval == 0:
                    rew = test_policy(policy, env)
                    print(f"mean test reward: {rew}")