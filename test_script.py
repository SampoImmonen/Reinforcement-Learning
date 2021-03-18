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
    print("initializing parameters")
    device = "cuda"
    

    env = make_env_basic("BoxingNoFrameskip-v4", crop=(30,30))
    
    output_dim = env.action_space.n

    net = DuelingNoisyConvNet((4, 84,84), output_dim).to(device)
    target_net = DuelingNoisyConvNet((4, 84,84), output_dim).to(device)
    target_net.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    threshold = 10000
    sync_interval = 1000
    gamma  = 0.99

    experience_source = ExperienceSource(env, capacity=50000, device=device)
    
    best_reward = 0
    batch_size = 32
    num_iters = 0
    episode_rewards = []
    #info_interval = 10

    t1 = time.time()

    logger = Logger("BoxingBasic", 10, complete_limit=50)


    print("starting training")
    while True:
        #print("starting iter")
        episode_reward = experience_source.step(net)

        logger.push_episode(episode_reward, net)        

        if experience_source.get_steps() < threshold:
            continue
        
        batch = experience_source.sample(batch_size, as_tensor=True)
        
        optimizer.zero_grad()
        loss = calculate_loss(batch, net, target_net, gamma, doubleQ=False)
        loss.backward()
        optimizer.step()

        logger.push_loss(loss.item())

        num_iters+=1
        if num_iters%sync_interval==0:
            target_net.load_state_dict(net.state_dict())
