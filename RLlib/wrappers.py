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

class FireResetEnv(gym.Wrapper):
    """
    From https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

def make_env_basic(name, frameskip=4, crop=None, episodic=False, firereset=False):
    env = gym.make(name)

    if episodic:
        env = EpisodicLifeEnv(env)
    if firereset:
        env = FireResetEnv(env)
        
    env = SkipFrame(env, frameskip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84, crop=crop)
    env = FrameStack(env, num_stack=frameskip)
    env = ArrayFromObs(env)
    return env

