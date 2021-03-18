import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import FrameStack
from PIL import Image

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward+=reward
            if done:
                break
            
        return obs, total_reward, done, info
    
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)


    def observation(self, observation):
        observation = observation[:, :, 0] * 0.299 + observation[:, :, 1] * 0.587 + observation[:, :, 2] * 0.114
        return observation/255.0
    
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape, crop=None):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape=(shape, shape)
        else:
            self.shape=tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space=Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.crop = crop
        
    def observation(self,observation):
        if self.crop != None:
            observation = observation[self.crop[0]:-self.crop[0], self.crop[1]:-self.crop[1]]
        #transforms = T.Compose(
        #    [T.Resize(self.shape), T.Normalize(0, 255)]
        #)
        #observation = transforms(observation)
        img = Image.fromarray(observation)
        img2 = img.resize(self.shape, Image.NEAREST)
        return np.asarray(img2)
                         
class TensorFromObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
    def observation(self, observation):
        return torch.tensor(observation.__array__(), dtype=torch.float)
    
class ArrayFromObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
    def observation(self, observation):
        return observation.__array__()


def make_env_basic(name, frameskip=4, crop=None):
    env = gym.make(name)
    env = SkipFrame(env, frameskip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84, crop=crop)
    env = FrameStack(env, num_stack=frameskip)
    env = ArrayFromObs(env)
    return env

