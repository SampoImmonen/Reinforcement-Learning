import torch
import torch.nn as nn

def calculate_loss(batch, net, target_net, gamma, doubleQ = False):
    
    states, acts, next_states, rewards, isdones = batch
    
    #print(states.shape)
    #print(acts.shape)
    #print(next_states.shape)
    #print(rewards.shape)
    #print(isdones.shape)
    
    batch_size = states.shape[0]
    
    state_action_values = net(states).gather(1, acts.unsqueeze(1)).squeeze(-1)
    
    #print(state_action_values.shape)
    with torch.no_grad():
        
        if doubleQ:
            
            next_state_acts = net(next_states).max(1)[1]
            next_state_values = target_net(next_states).gather(1, next_state_acts.unsqueeze(1)).squeeze(1)
            
        else:
            next_state_values = target_net(next_states).max(1)[0]
        
    next_state_values[isdones] = 0.0
    #print(next_state_values)
    #print(next_state_values.shape)
    #mask = torch.ones(batch_size)-isdones
    expected_Q_values = next_state_values.detach()*gamma+rewards
    
    loss = nn.MSELoss()(state_action_values.float(), expected_Q_values.float())
    
    return loss