import time
import torch

from wrappers import make_env_basic
from utils import ExperienceSource
from models import DuelingNoisyConvNet
from dqn import calculate_loss
from logger import Logger



if __name__ == '__main__':

    # Add CLI

    #define parameters
    config = {
        'env' : 'BreakoutNoFrameskip-v4',
        'frameskip' : 4,
        'lr' : 0.0001,
        'batch_size' : 32,
        'train_threshold': 10000,
        'buffer_capacity': 100000,
        'gamma' : 0.99,
        'n_steps':2,
        'device': 'cuda',
        'step_interval': 1,
        'log_interval': 10,
        'sync_interval': 10,
        'doubleQ': False,
        'log_dir' : 'BreakoutTest',
        'dueling': True,
        'noisy': True
    }

    print("initializing parameters")
    device = config['device']

    env = make_env_basic(config['env'], config['frameskip'], episodic=True, firereset=True)
    output_dim = env.action_space.n

    #Implement Logic to choose network
    net = DuelingNoisyConvNet((config['frameskip'], 84,84), output_dim).to(device)
    net.load_state_dict(torch.load("BreakoutTest/checkpoint_eps3050_reward16.7.pt"))
    target_net = DuelingNoisyConvNet((config['frameskip'], 84,84), output_dim).to(device)
    target_net.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    threshold = config['train_threshold']
    sync_interval = config['sync_interval']

    n_steps = config['n_steps']
    gamma = config['gamma']
    capacity = config['buffer_capacity']

    experience_source = ExperienceSource(env, capacity=capacity,gamma=gamma, n_steps=n_steps, device=device)
    
    batch_size = config['batch_size']
    step_interval = config['step_interval']
    log_interval = config['log_interval']
    log_dir = config['log_dir']

    num_iters = 0
    #info_interval = 10

    doubleQ = config['doubleQ']

    logger = Logger(config)


    print("starting training")
    while True:
        #print("starting iter")
        episode_reward = experience_source.step(net)

        logger.push_episode(episode_reward, net)        

        steps_done = experience_source.get_steps()
        if  steps_done < threshold:
            continue
        
        if steps_done%step_interval==0:
            batch = experience_source.sample(batch_size, as_tensor=True)
            
            optimizer.zero_grad()
            loss = calculate_loss(batch, net, target_net, gamma**(n_steps), doubleQ=doubleQ)
            loss.backward()
            optimizer.step()

            logger.push_loss(loss.item())

            num_iters+=1
            if num_iters%sync_interval==0:
                target_net.load_state_dict(net.state_dict())
