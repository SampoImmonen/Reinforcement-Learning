import numpy as np
import torch



def get_batch(batch, net, n_steps, gamma=0.99):
    """
    inputs: batch (list of Experiences), net (NN to calculate value estimate), n_steps (for how many steps is the reward discounted), gamma
    output: Tensors for states, actions and discounted rewards
    """
    
    states, acts, next_states, rewards, isdones = zip(*batch)
    states, acts, next_states, rewards, isdones = np.stack(states), np.stack(acts), np.stack(next_states), np.stack(rewards), np.stack(isdones)
    done_mask = np.logical_not(isdones)
    #print(acts.shape)
    #print(done_mask.shape)
    
    tens = torch.FloatTensor(next_states[done_mask])
    Q_vals = net(tens)
    #print(Q_vals.shape)
    #print(rewards.shape)
    rewards[done_mask] = rewards[done_mask]+Q_vals.data.cpu().numpy()[:,0]*(gamma**n_steps)
    
    
    return (torch.FloatTensor(states), torch.tensor(acts, dtype=torch.int64), torch.FloatTensor(rewards))