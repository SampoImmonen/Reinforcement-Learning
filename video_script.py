import gym
from models import DuelingNoisyConvNet
from wrappers import make_env_basic
from video_tools import show_episode
import torch

if __name__ == '__main__':
    
    
    env = make_env_basic('BreakoutNoFrameskip-v4', episodic = True, firereset=True)
    model = DuelingNoisyConvNet((4, 84,84), env.action_space.n)
    model.load_state_dict(torch.load('BreakoutTest/checkpoint_eps4890_reward155.0.pt'))

    show_episode(model, env, save_video=True, images_path="./BreakoutTest/images")