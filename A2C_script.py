import gym
import time
import torch
import torch.nn.functional as F
import torch.nn as nn

from RLlib.models import PolicyNet, ValueNet
from RLlib.utils import ExperienceSourceForPolicy
from RLlib.policy import SamplingSinglePolicy
from RLlib.actorcritic import get_batch



if __name__ == "__main__":

    #Add CLI
    print("Initialzing paramaters")
    
    #basic parameters
    gamma = 0.99
    lr = 0.001
    beta = 0.005
    batch_size = 32
    sync_interval = 1000
    reward_steps = 4
    clip_grad = 0.1
    
    #environment
    env = gym.make('LunarLander-v2')
    
    #models
    policy_net = PolicyNet(8, 4)
    value_net = ValueNet(8)
    target_value_net = ValueNet(8)
    target_value_net.load_state_dict(value_net.state_dict())

    #policy and experience soure
    policy = SamplingSinglePolicy(policy_net)
    exp_source = ExperienceSourceForPolicy(env, n_steps = reward_steps)

    #Optimizers
    policyoptimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    valueoptimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    
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
                
                states, acts, rewards = get_batch(batch, target_value_net, n_steps=reward_steps)
                batch.clear()
                policyoptimizer.zero_grad()
                valueoptimizer.zero_grad()
                
                logits = policy_net(states)
                vals = value_net(states)
                
                value_loss = F.mse_loss(vals.squeeze(-1), rewards)
                
                log_probs = F.log_softmax(logits, dim=1)
                adv_v = rewards-vals.detach()
                log_p_a = log_probs[range(batch_size), acts]
                
                policy_loss = -(adv_v*log_p_a).mean()
                
                #policy_loss.backward(retain_graph=True)
                
                probs = F.softmax(logits, dim=1)
                ent = (probs*log_probs).sum(dim=1).mean()
                
                entropy_loss = beta*ent
                
                p_loss = policy_loss+entropy_loss
                
                p_loss.backward()
                value_loss.backward()
                        
                    
                nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad)
                nn.utils.clip_grad_norm_(value_net.parameters(), clip_grad)
                
                policyoptimizer.step()
                valueoptimizer.step()
                

                #optimizer.step()
                if num_steps%sync_interval == 0:
                    target_value_net.load_state_dict(value_net.state_dict())
