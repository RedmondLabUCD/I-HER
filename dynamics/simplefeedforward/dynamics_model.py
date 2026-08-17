import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# define simple feed-forward dynamics model
class DynamicsModel(nn.Module):
    def __init__(self, obs_size, act_size, hidden=512):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(obs_size + act_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, obs_size)

    def forward(self, obs, a):
        x = torch.cat([obs, a], dim=-1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prediction = self.fc3(x)
        return prediction
    
# # define simple feed-forward dynamics model
# class DynamicsModelDelta(nn.Module):
#     def __init__(self, obs_size, act_size, hiddens=2, hidden_size=512):
#         print('WARNING: old dynamics model class!!!!')
#         super(DynamicsModelDelta, self).__init__()
#         self.fc1 = nn.Linear(obs_size + act_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, obs_size)

#     def forward(self, obs, a):
#         x = torch.cat([obs, a], dim=-1).float()
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         prediction = self.fc3(x)
#         return prediction
    
# define simple feed-forward dynamics model
class DynamicsModelDelta(nn.Module):
    def __init__(self, obs_size, act_size, hiddens=2, hidden_size=512):
        super(DynamicsModelDelta, self).__init__()
        assert hiddens > 0, "Must have at least 1 hidden layer"
        self.hidden_layers = nn.ModuleList([nn.Linear(obs_size + act_size, hidden_size)])
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for i in range(hiddens-1)])
        self.fc_final = nn.Linear(hidden_size, obs_size)

    def forward(self, obs, a):
        x = torch.cat([obs, a], dim=-1).float()
        for fc in self.hidden_layers:
            x = F.relu(fc(x))
        delta = self.fc_final(x)
        return delta
    
    
# # define simple feed-forward dynamics model
# class DynamicsGMM(nn.Module):
#     def __init__(self, obs_size, act_size, hidden=512, gaussians=5):
#         super(DynamicsGMM, self).__init__()
#         self.gaussians = 5
#         self.obs_size = obs_size
        
#         self.fc1 = nn.Linear(obs_size + act_size, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, (2 * obs_size + 1) * gaussians)

#     def forward(self, obs, a):
#         x = torch.cat([obs, a], dim=-1).float()
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         gmm_outs = self.fc3(x)
        
#         stride = self.gaussians * self.obs_size
#         mus = gmm_outs[:stride]
#         mus = mus.view(self.gaussians, self.obs_size)
#         sigmas = gmm_outs[stride:2*stride]
#         sigmas = sigmas.view(self.gaussians, self.obs_size)
#         pi = gmm_outs[2*stride:]
#         logpi = F.log_softmax(pi, dim=-1)
        
#         return mus, sigmas, logpi
    
#     def gmm_loss(mus, sigmas, logpi, next_obs):
#         batch = batch.unsqueeze(-2) # add dim 1 where mus have gs gaussians
#         normal_dist = Normal(mus, sigmas)
#         g_log_probs = normal_dist.log_prob(batch)
#         g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
#         max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
#         g_log_probs = g_log_probs - max_log_probs
    
#         g_probs = torch.exp(g_log_probs)
#         probs = torch.sum(g_probs, dim=-1)
    
#         log_prob = max_log_probs.squeeze() + torch.log(probs)
        
#         return - torch.mean(log_prob) / self.obs_size