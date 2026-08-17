import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer, BasicReplayBuffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer, RunningMeanStd
from her_modules.her import her_sampler

import torch.nn.functional as f
from worldmodels.misc import save_checkpoint
from pprint import pprint
from worldmodels.learning import ReduceLROnPlateau
import torch.nn as nn

from dynamics.simplefeedforward.dynamics_model import DynamicsModel
from worldmodels.fetchpush_dynamics import FetchPushDynamics
import copy

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent_iter_dynamics:
    def __init__(self, args, env, env_params):
        self.args = args
        self.realenv = env
        
        obs = self.realenv.reset()
        self.obs_size = obs['observation'].shape[0] # + obs['desired_goal'].shape[0]
        self.act_size = self.realenv.action_space.shape[0]
        # setup dynamics model
        self.dy_model = DynamicsModel(self.obs_size, self.act_size)
        self.dy_optimizer = torch.optim.Adam(self.dy_model.parameters(), lr=1e-3)
        model_state = torch.load('dynamics/simplefeedforward/dynamicsmodel_norm.tar')
        self.dy_model.load_state_dict(model_state['state_dict'])
        self.dy_optimizer.load_state_dict(model_state["optimizer"])
        og_realmeans = model_state['means']
        og_realstds = model_state['stds']
        og_norm_count = model_state['env_steps']
        self.mseloss = nn.MSELoss()
        # setup real replay buffer and obs rms
        self.realobs_rms = RunningMeanStd(og_realmeans, og_realstds, og_norm_count)
        self.buffer_real = BasicReplayBuffer(self.obs_size, self.act_size, self.args.buffer_size)
        # setup imagined env
        self.env = FetchPushDynamics(self.dy_model, self.realenv, self.realobs_rms, self.args.clip_range)
        self.old_actors = []
        
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
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
        # create the replay buffer
        self.buffer_imag = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions) # is replay buffer shared between processes?
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network
        
        """
        # torch.autograd.set_detect_anomaly(True)
        # self.test_dynamics()
        # self.collect_real_exp()
        # self.update_dynamics(1)
        # self.test_dynamics()
        # return
        self.collect_real_exp() # Add random actions to buffer
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            if epoch % 10 == 0 and epoch > 1:
                # Collect real experience with current actor and update dynamics model
                self.collect_real_exp()
                self.update_dynamics(epoch)
                # Clear imagined replay buffer and refill using updated dynamics model and all old actors
                self.buffer_imag = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
                self.refill_imagined_replay_buffer(epoch)
            # store current actor
            old_actor = {
              "actor": copy.deepcopy(self.actor_network), # actor(self.env_params).load_state_dict(self.actor_network.state_dict()),
              "o_mean": self.o_norm.mean.copy(),
              "o_std": self.o_norm.std.copy(),
              "g_mean": self.g_norm.mean.copy(),
              "g_std": self.g_norm.std.copy(),
            }
            self.old_actors += [old_actor]
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
                self.buffer_imag.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model6dynamics{}.pt'.format(epoch))

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
        num_transitions = mb_actions.shape[1]
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
        transitions = self.buffer_imag.sample(self.args.batch_size)
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
            clip_return = 1 / (1 - self.args.gamma)
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

    def preproc_inputs_old(self, obs, g, o_mean, o_std, g_mean, g_std):
        o_clip = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
        g_clip = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        o_norm = np.clip((o_clip - o_mean) / (o_std), -self.args.clip_range, self.args.clip_range)
        g_norm = np.clip((g_clip - g_mean) / (g_std), -self.args.clip_range, self.args.clip_range)
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return inputs

    def refill_imagined_replay_buffer(self, epoch):
        # TODO: don't go over replay limit
        rollouts_per_epoch = self.args.n_cycles * self.args.num_rollouts_per_mpi
        for old_actor in self.old_actors:
            actor = old_actor['actor']
        # ############################## biased to more recent actors code between these line
        # rollouts_to_generate = epoch * self.args.n_cycles * self.args.num_rollouts_per_mpi
        # proportions = np.linspace(1, 2, num=len(self.old_actors))
        # proportions /= np.sum(proportions)
        # actor_rollouts = np.rint(proportions * rollouts_to_generate)
        # for i in range(len(self.old_actors)):
        #     old_actor = self.old_actors[i]
        #     actor = old_actor['actor']
        #     rollouts = int(actor_rollouts[i])
        # ############################
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            for _ in range(rollouts_per_epoch):
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
                        input_tensor = self.preproc_inputs_old(obs, g, old_actor['o_mean'], old_actor['o_std'], old_actor['g_mean'], old_actor['g_std'])
                        pi = actor(input_tensor)
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
            self.buffer_imag.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
            # TODO: clear normaliser and re-initialise?

    def update_dynamics(self, epoch):
        batch_size = 512 # TODO: could reduce size to speed up
        steps = int(1e4)
        cum_loss = 0
        for i in range(steps):
            batch = self.buffer_real.sample_batch_biasrecent(batch_size) # TODO: revert!!
            obs, a, obs2 = batch['obs'], batch['act'], batch['obs2']
            obs = np.clip((obs - self.realobs_rms.cur_mean) / self.realobs_rms.cur_std, -self.args.clip_range, self.args.clip_range)
            obs2 = np.clip((obs2 - self.realobs_rms.cur_mean) / self.realobs_rms.cur_std, -self.args.clip_range, self.args.clip_range)
            
            self.dy_optimizer.zero_grad()
            predictions = self.dy_model(obs, a)
            loss = self.mseloss(predictions.float(), obs2.float())
            loss.backward()
            sync_grads(self.dy_model)
            self.dy_optimizer.step()
            # print('step {}, loss: {}'.format(i, loss.item()))
            cum_loss += loss.item()
        self.env.dy_model.load_state_dict(self.dy_model.state_dict())
        print('Prediction loss: {}'.format(cum_loss/steps))
        if MPI.COMM_WORLD.Get_rank() == 0:
            # print('Prediction loss: {}'.format(cum_loss/steps))
            model_state = {
                "state_dict": self.dy_model.state_dict(),
                "optimizer": self.dy_optimizer.state_dict(),
                "means": self.realobs_rms.cur_mean,
                "stds": self.realobs_rms.cur_std,
                "env_steps": self.realobs_rms.cur_count}
            torch.save(model_state, 'dynamicsmodel6_norm{}.tar'.format(epoch))

    def collect_real_exp(self):
        rollouts = 1000 # TODO: make arg
        for _ in range(rollouts):
            # reset the environment
            observation = self.realenv.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    action = self._select_actions(pi)
                # feed the actions into the environment
                observation_new, _, _, _ = self.realenv.step(action)
                obs_new = observation_new['observation']
                # Store experience to replay buffer
                self.buffer_real.store(obs, action, obs_new)
                # re-assign the observation
                obs = obs_new
        # Calculate new individual obs mean and std
        self.realobs_rms.update_from_replay(self.buffer_real.obs_buf[:self.buffer_real.size,:])
        # Calculate means and stds across processes
        if MPI.COMM_WORLD.Get_size() == 2:
            #TODO only works for -np 2!!!!
            self.realobs_rms.combine_rms_across_processes()
        
    def test_dynamics(self):
        for _ in range(5):
            observation = self.realenv.reset()
            self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    action = self._select_actions(pi)
                observation, _, _, info = self.realenv.step(action)
                obs = observation['observation']
                self.env.step(action)
                self.env.render()
                self.realenv.render()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, r, _, _ = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(r+1) # info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
