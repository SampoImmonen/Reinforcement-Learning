import json
import os
import time
import torch



def check_dir(name):
    """
    function to check if directory exists
    if not then creates directory
    """
    if not os.path.exists(name):
        os.makedirs(name)
        return True
    return False


class Logger:

    """
    class used for tracking results, printing information and saving checkpoints
    """
    def __init__(self, config, best_reward = -21,complete_limit=None, stop_limit=None):
       
        self.save_dir = config['log_dir']
        check_dir(self.save_dir)   
        i = 1

        #Create new directory for every run 
        while True:
            run_path = os.path.join(self.save_dir, f"run{i}")
            if check_dir(run_path):
                break
            i+=1
        self.save_dir=run_path

        
        config_path = os.path.join(self.save_dir, "config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f)

        self.episode_stats_path = os.path.join(self.save_dir, 'episode_stats.txt')

        self.interval = config['log_interval']
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

                self._writetofile()

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
                    if self.stop_limit!=None and mean_reward > self.stop_limit:
                        return True
        return False
                    
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

    def _writetofile(self):

        with open(self.episode_stats_path, 'a') as f:
            for ep in self.episodes[-self.interval:]:
                f.write(str(ep)+'\n')

        return