import torch
import torch.nn as nn
import gym
import torch.multiprocessing as mp
import time
from collections import namedtuple
from models import DuelingNoisyConvNet
from wrappers import make_env_basic
from utils import Experience, PlayedEpisode, ReplayBuffer
from dqn import calculate_loss

def play_proc3(net, queue):
    env = make_env_basic('PongNoFrameskip-v4')
    state = env.reset()
    print("init process")
    episode_reward = 0
    num_steps = 0
    net.cuda()
    while True:
        obs_tens = torch.tensor(state).unsqueeze(0).float().to("cuda")
        with torch.no_grad():
            act = net(obs_tens).max(1)[1]
        
        next_state, reward, isdone, _ = env.step(act.item())

        episode_reward+=reward
        num_steps +=1

        exp = Experience(state, act.item(), next_state, reward, isdone)
        queue.put(exp)

        if isdone:
            next_state = env.reset()
            queue.put(PlayedEpisode(episode_reward, num_steps))
            episode_reward=0
            num_steps=0
        state = next_state

def train(net, exp_queue):
    

    sync_interval = 1000
    net.cuda()
    target_net = DuelingNoisyConvNet((4,84,84), 6).cuda()
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    buffer = ReplayBuffer(capacity=50000)

    threshold = 10000
    batch_size = 32*4
    gamma = 0.99    
    
    tracker = EpisodeTracker(10, 15)
    training_steps = 0

    while True:
        exp = exp_queue.get()
        if isinstance(exp, PlayedEpisode):
            tracker.push(exp)

        else:
            buffer.push(exp)

        if len(buffer) < threshold:
            continue

        batch = buffer.sample(batch_size, as_tensor=True, device="cuda")

        optimizer.zero_grad()
        loss = calculate_loss(batch, net, target_net, gamma)
        loss.backward()
        optimizer.step()
        training_steps+=1
        if training_steps%sync_interval==0:
            target_net.load_state_dict(net.state_dict())


class EpisodeTracker:

    def __init__(self, interval, success_score):

        self.interval = interval
        self.success_score = success_score

        self.t1 = time.time()
        self.episodes = []

    def push(self, episode):
        self.episodes.append(episode)
        if len(self.episodes)%self.interval == 0:
            mean = sum([episode.reward for episode in self.episodes[-self.interval:]])/self.interval
            completed_episodes = sum([ep.reward > self.success_score for ep in self.episodes[-self.interval:]])
            print(f"episodes played: {len(self.episodes)} mean reward:{mean}, completed episodes: {completed_episodes} time:{time.time()-self.t1}")

        return 


if __name__ == '__main__':

    #warnings.simplefilter("ignore", category=UserWarning)
    print("starting script")
    mp.set_start_method('spawn')

    #input and output size for pong
    net = DuelingNoisyConvNet((4,84,84), 6)

    
    net.share_memory()

    

    # Shared queue and playing process
    exp_queue = mp.Queue(maxsize=8)
    play_proc = mp.Process(target=play_proc3, args=(net, exp_queue))
    train_proc = mp.Process(target=train, args=(net, exp_queue))
    play_proc.start()
    train_proc.start()

    play_proc.join()
    train_proc.join()