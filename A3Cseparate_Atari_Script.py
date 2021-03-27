import gym
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp

from RLlib.models import ConvPolicyNet
from RLlib.utils import ExperienceSourceForPolicy
from RLlib.policy import SamplingSinglePolicy
from RLlib.actorcritic import get_batch
from RLlib.wrappers import make_env_basic

from collections import namedtuple

EpisodeReward = namedtuple('EpisodeReward', ('reward'))

def experience_func(policynet, targetvaluenet, train_queue, device="cpu"):
    
    env = make_env_basic("PongNoFrameskip-v4")
    print("process started")
    policy = SamplingSinglePolicy(policynet)
    exp_source = ExperienceSourceForPolicy(env, n_steps = N_STEPS, gamma=GAMMA, device=device)
    
    batch = []
    
    while True:
        
        exp, rew = exp_source.step(policy)
        batch.append(exp)
        
        if rew!=None:
            reward, _ = rew            
            train_queue.put(EpisodeReward(reward))
    
        if len(batch) < PR_BATCH_SIZE:
            continue
            
        data = get_batch(batch, targetvaluenet, n_steps=N_STEPS, gamma=GAMMA, device=device)
        
        train_queue.put(data)
        batch.clear()


GAMMA = 0.99
PR_BATCH_SIZE = 8
BATCH_SIZE = 32
N_STEPS = 4
NUM_PROCS = 4


lr = 0.001
beta = 0.005
sync_interval = 500
clip_grad = 0.1
#Cuda does not work with windows
device = 'cpu'



if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    #os.environ['CUDA_LAUNCH_BLOCKING']="1"

    #Setting parameters
    env = make_env_basic("PongNoFrameskip-v4")
    policynet = ConvPolicyNet((4, 84, 84), env.action_space.n)
    policynet.share_memory()

    valuenet = ConvPolicyNet((4, 84, 84), 1)
    targetvaluenet = ConvPolicyNet((4, 84, 84), 1)
    targetvaluenet.load_state_dict(valuenet.state_dict())

    targetvaluenet.share_memory()

    train_queue = mp.Queue(maxsize=NUM_PROCS)

    print("Starting experience processes")
    procs = []
    for _ in range(NUM_PROCS):
        proc = mp.Process(target=experience_func, 
                          args=(policynet, targetvaluenet, train_queue, device))
        proc.start()
        procs.append(proc)

    
    policyoptimizer = torch.optim.Adam(policynet.parameters(), lr=lr, eps=1e-3)
    valueoptimizer = torch.optim.Adam(policynet.parameters(), lr=lr, eps=1e-3)
    #Main Training Loop
    print("starting training loop")

    batch_states = []
    batch_acts = []
    batch_rewards = []
    
    best = -200
    num_steps = 0
    start_time = time.time()
    episode_rewards = []
    try:
        while True:
            data = train_queue.get()
            
            if isinstance(data, EpisodeReward):
                episode_rewards.append(data.reward)
                print(data, time.time()-start_time)
                if len(episode_rewards)%10==0 and len(episode_rewards) > 0:
                    if sum(episode_rewards[-10:])/10 > best:
                        print(sum(episode_rewards[-10:])/10)
                        print(len(episode_rewards))
                        print(time.time()-start_time)
                        best = sum(episode_rewards[-10:])/10
                        print(35*'-')
                    if sum(episode_rewards[-10:])/10 > 18:
                        print("Solved")
                        break
                continue
            
            states, acts, rewards = data
            batch_states.append(states)
            batch_acts.append(acts)
            batch_rewards.append(rewards)

            if len(batch_states)*PR_BATCH_SIZE < BATCH_SIZE:
                continue


            num_steps+=1
            states = torch.cat(batch_states)
            acts = torch.cat(batch_acts)
            rewards = torch.cat(batch_rewards)
            batch_states.clear()
            batch_acts.clear()
            batch_rewards.clear()

            policyoptimizer.zero_grad()

            logits = policynet(states)
            vals = valuenet(states)
            
            value_loss = F.mse_loss(vals.squeeze(-1), rewards)

            #Policy loss
            log_probs = F.log_softmax(logits, dim=1)
            adv_v = rewards-vals.detach()
            log_p_a = log_probs[range(BATCH_SIZE), acts]
            policy_loss = -(adv_v*log_p_a).mean()

            #Entropy loss
            probs = F.softmax(logits, dim=1)
            ent = (probs*log_probs).sum(dim=1).mean()

            entropy_loss = beta*ent

            p_loss = policy_loss+entropy_loss
            
            p_loss.backward()
            value_loss.backward()
            
            nn.utils.clip_grad_norm_(valuenet.parameters(), clip_grad)
            nn.utils.clip_grad_norm_(policynet.parameters(), clip_grad)

            policyoptimizer.step()
            valueoptimizer.step()

            if num_steps%sync_interval==0:
                targetvaluenet.load_state_dict(valuenet.state_dict())

    finally:
        for proc in procs:
            proc.terminate()
            proc.join()