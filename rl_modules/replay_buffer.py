import threading
import numpy as np
import torch

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
        # if self.current_size == self.size:
        #     print('Uh oh, imaginary replay max size reached')
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions
    
    # def mdrnn_sample(self, batch_size, seq_split=False):
    #     temp_buffers = {}
    #     idxs = np.random.randint(0, self.current_size, batch_size)
    #     with self.lock:
    #         for key in self.buffers.keys():
    #             temp_buffers[key] = self.buffers[key][idxs]
    #     temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
    #     temp_buffers['obs'] = temp_buffers['obs'][:, :-1, :]
    #     temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
    #     if seq_split:
    #         temp_buffers['obs'] = np.concatenate((temp_buffers['obs'][:,:25,:], temp_buffers['obs'][:,25:,:]), axis=0)
    #         temp_buffers['actions'] = np.concatenate((temp_buffers['actions'][:,:25,:], temp_buffers['actions'][:,25:,:]), axis=0)
    #         temp_buffers['obs_next'] = np.concatenate((temp_buffers['obs_next'][:,:25,:], temp_buffers['obs_next'][:,25:,:]), axis=0)
    #     return temp_buffers
        

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

class BasicReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.iter_buf = np.zeros((size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.overflowing = False

    def store(self, obs, act, next_obs, update_iter):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.iter_buf[self.ptr] = update_iter
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        if self.size >= self.max_size and not self.overflowing:
            print('Uh oh, real replay max size reached')
            self.overflowing = True

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    # Warning: only works if max replay size isn't reached.
    # # once reached, new experiences are added to front
    # def sample_batch_biasrecent(self, batch_size=32):
    #     # Twice as likely to sample last experience as 1st experience
    #     probabilities = np.linspace(1, 2, num=self.size)
    #     # Ensure probailities sum to 1
    #     probabilities /= np.sum(probabilities)
    #     idxs = np.random.choice(self.size, size=batch_size, p=probabilities)
    #     batch = dict(obs=self.obs_buf[idxs],
    #                  obs2=self.obs2_buf[idxs],
    #                  act=self.act_buf[idxs],)
    #     return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    # Warning: only works if max replay size isn't reached.
    # once reached, new experiences are added to front.
    # Works if each update generally uses equal amount of steps(?)
    def sample_batch_biasrecent(self, batch_size=32, bias=2):
        # Twice as likely to sample last batch of experiences as 1st batch
        probs = self.iter_buf[0:self.size].copy()
        if probs[self.size-1] > 0:
            probs /= probs[self.size-1]
            probs = probs * (bias - 1)
        probs += 1 # + 1 to ensure prob isn't 0
        # Ensure probailities sum to 1
        probs /= np.sum(probs)
        idxs = np.random.choice(self.size, size=batch_size, p=probs)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def create_split(self, split=0.8):
        idxs = np.arange(self.size)
        np.random.shuffle(idxs)
        split = int(split*self.size)
        train_idxs, val_idxs = idxs[0:split], idxs[split:self.size]
        self.train_data = {
            'obs': self.obs_buf[train_idxs],
            'obs2': self.obs2_buf[train_idxs],
            'act': self.act_buf[train_idxs],
            }
        self.val_data = {
            'obs': self.obs_buf[val_idxs],
            'obs2': self.obs2_buf[val_idxs],
            'act': self.act_buf[val_idxs],
            }
        
    def shuffle_train(self):
        # TODO: verify working!!
        idxs = np.arange(self.train_data['obs'].shape[0])
        np.random.shuffle(idxs)
        self.train_data['obs'], self.train_data['obs2'], self.train_data['act'] = \
            self.train_data['obs'][idxs], self.train_data['obs2'][idxs], self.train_data['act'][idxs]
    
    # def normalize_obs(self, clip_range=5):
    #     self.obs_means = np.mean(self.obs_buf, axis=0)
    #     self.obs_stds = np.std(self.obs_buf, axis=0)
    #     for i in range(self.obs_stds.shape[0]):
    #         if self.obs_stds[i] == 0.0: self.obs_stds[i] = 1
    #     self.obs_buf = np.clip((self.obs_buf - self.obs_means) / self.obs_stds, -clip_range, clip_range)
    #     self.obs2_buf = np.clip((self.obs2_buf - self.obs_means) / self.obs_stds, -clip_range, clip_range)