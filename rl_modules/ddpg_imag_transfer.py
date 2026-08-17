import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

# from worldmodels.mdrnn import MDRNNCell, MDRNN, gmm_loss
# # from worldmodels.fetchreach_rnn import FetchReachRNN
# from worldmodels.fetchpush_rnn import FetchPushRNN
# import torch.nn.functional as f
# from worldmodels.misc import save_checkpoint
# from pprint import pprint
# from worldmodels.learning import ReduceLROnPlateau
# import torch.nn as nn

from dynamics.simplefeedforward.dynamics_model import DynamicsModelDelta
from worldmodels.fetchpush_ensembledelta import FetchPushEnsembleDelta

"""
ddpg with HER (MPI-version)
Use dynamics model trained in pick-place task to train agent in push task
Contrain all actions input to imagined env as follows:
action[-1] = -1
This ensures gripper is not opened at any time

"""
class ddpg_imag_transfer:
    def __init__(self, args, env, env_params):
        self.args = args
        self.realenv = env
        
        # Setup ensemble dynamics imagination env
        obs = env.reset()
        obs_size = obs['observation'].shape[0]
        act_size = env.action_space.shape[0]
        dy_model_state = torch.load(self.args.ensemble_path)
        self.ensemble_models = []
        for i in range(5):
            dy_model = DynamicsModelDelta(obs_size, act_size)
            dy_model.load_state_dict(dy_model_state['state_dict{}'.format(i)])
            self.ensemble_models += [dy_model]
        # setup normaliser
        self.real_o_norm = normalizer(size=obs_size, default_clip_range=args.clip_range)
        self.real_o_norm.mean = dy_model_state['obs_mean']
        self.real_o_norm.std = dy_model_state['obs_std']
        real_delta_norm = normalizer(size=obs_size, default_clip_range=args.clip_range)
        real_delta_norm.mean = dy_model_state['delta_mean']
        real_delta_norm.std = dy_model_state['delta_std']
        # setup imagined env. Force gripper closed, as in push env
        self.env = FetchPushEnsembleDelta(self.ensemble_models, env, self.real_o_norm, real_delta_norm, args.clip_range, close_gripper=True)
        
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        
        if self.args.load_ac:
            o_mean, o_std, g_mean, g_std, actor_state, critic_state = torch.load(self.args.ac_path, map_location=lambda storage, loc: storage)
            # create the actor network
            self.actor_network.load_state_dict(actor_state)
            self.critic_network.load_state_dict(critic_state)
        
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffers
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # self.buffer_real = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        
        if self.args.load_ac:
            # Load in norms!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.o_norm.mean, self.o_norm.std = o_mean, o_std
            self.g_norm.mean, self.g_norm.std = g_mean, g_std
        
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.exp_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
                
        self.logs = {
                    'epoch': [],
                    'imag_rate': [],
                    'real_rate': [],
                    'ri': [],
                    'a_loss': [],
                    'q_loss': [],
                    }

    def learn(self):
        """
        train the network
        
        """
        print('Training Push agent within PickAndPlace ensemble')
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            actor_loss, critic_loss, self.ri = [], [], 0
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    a_loss, q_loss = self._update_network()
                    actor_loss += [a_loss]
                    critic_loss += [q_loss]
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            imag_success_rate = self._eval_agent(real=False)
            real_success_rate = self._eval_agent(real=True)
            self._update_logs(epoch, imag_success_rate, real_success_rate, np.mean(actor_loss), np.mean(critic_loss))
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch: {} imag rate: {:.2f} real rate: {:.2f} ri: {:.4f} a_loss: {:.2f} q_loss {:.2f}'.format(datetime.now(), epoch, imag_success_rate, real_success_rate, self.ri, np.mean(actor_loss), np.mean(critic_loss)))
                np.save(self.model_path + '/logs.npy', self.logs)
                if self.args.save_models == 1:
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                                self.model_path + '/ac_epoch{}.pt'.format(epoch))

    def _update_logs(self, epoch, imag_success_rate, real_success_rate, actor_loss, critic_loss):
        self.ri = self.ri / (self.args.n_batches * self.args.num_rollouts_per_mpi * self.args.n_cycles)
        self.logs['epoch'].append(epoch)
        self.logs['imag_rate'].append(imag_success_rate)
        self.logs['real_rate'].append(real_success_rate)
        self.logs['ri'].append(self.ri)
        self.logs['a_loss'].append(actor_loss)
        self.logs['q_loss'].append(critic_loss)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1] # TODO: fix!
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        if self.args.include_ri:
            r_intrinsic = - self.get_intrinsic_reward(transitions['obs'], transitions['actions'])
            transitions['r'] += r_intrinsic
            self.ri += np.mean(r_intrinsic)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma) # TODO: fix to account for negative intrinsics?
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()
        return actor_loss.detach().numpy(), critic_loss.detach().numpy()

    def get_intrinsic_reward(self, obs, action, scale=0.5, clip=0.8, close_gripper=True):
        if close_gripper:
            action = action.copy()
            action[..., -1] = -1
        obs = self.real_o_norm.normalize(obs)
        predictions = []
        for model in self.ensemble_models:
            predictions += [model(torch.tensor(obs), torch.tensor(action)).detach().numpy()]
        predictions = np.clip(np.array(predictions), -self.args.clip_range, self.args.clip_range)
        mean_prediction = np.mean(predictions, axis=0)
        dist_from_mean = np.sum(np.square(predictions - mean_prediction), axis=-1)
        r_intrinsic = np.expand_dims(np.mean(dist_from_mean, axis=0), axis=1)
        return np.clip(r_intrinsic*scale, 0, clip)

    # do the evaluation
    def _eval_agent(self, real=False):
        if real:
            env = self.realenv
        else:
            env = self.env
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, r, _, _ = env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(r+1) # info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
