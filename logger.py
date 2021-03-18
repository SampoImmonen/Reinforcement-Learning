import os
import torch
import time


def check_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)

class Logger:
    def __init__(self, save_dir, interval, best_reward = 0,complete_limit=None, stop_limit=None):
       
        self.save_dir = save_dir
        
        check_dir(save_dir)

        self.interval = interval
        self.episodes = []
        self.episode_losses=[]
        self.episode_loss = []
        self.best_reward = best_reward
        
        self.complete_limit = complete_limit
        self.stop_limit = stop_limit
        self.log_length=True
        self.start_time = time.time()
        
        
    def push_episode(self, episode, net):
        if episode!=None:
            print(episode+(time.time()-self.start_time,))
            self.episodes.append(episode+(time.time()-self.start_time,))
    
            if len(self.episode_loss) > 0:
                mean_loss = sum(self.episode_loss)/len(self.episode_loss)
                self.episode_losses.append(mean_loss)
                self.episode_loss = []
                
            if len(self.episodes)%self.interval==0:
                interval = time.time()-self.start_time
                mean_stats = self._mean_stats()
                mean_reward, mean_length, mean_loss = mean_stats                
                print(f"{len(self.episodes)} episodes played, mean reward:{mean_reward}, mean ep length:{mean_length}, time elapsed:{interval}")
                if self.complete_limit!=None:
                    completed_episodes = sum([episode[0]>self.complete_limit for episode in self.episodes[-self.interval:]])
                    print(f"episodes completed: {completed_episodes}")
                if mean_reward > self.best_reward:
                    print("new best average")
                    num_eps = len(self.episodes)
                    torch.save(net.state_dict(), self.save_dir+"/"+f"checkpoint_eps{num_eps}_reward{mean_reward}.pt")
                    self.best_reward = mean_reward
                    
                    
    def push_loss(self, loss):
        """
        input has to be loss.item() other wise computational graph if stored
        """
        self.episode_loss.append(loss)
                
        return
    def _mean_stats(self):
        mean_reward = sum([episode[0] for episode in self.episodes[-self.interval:]])/self.interval
        mean_length = sum([episode[1] for episode in self.episodes[-self.interval:]])/self.interval
        mean_loss = sum(loss for loss in self.episode_losses[-self.interval:])/self.interval 
        return (mean_reward, mean_length, mean_loss)