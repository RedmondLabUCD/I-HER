import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads, mpi_average
from rl_modules.replay_buffer import replay_buffer, BasicReplayBuffer
from rl_modules.models import actor, critic

from mpi_utils.normalizer import normalizer

from her_modules.her import her_sampler, goal_distance_reward

import torch.nn as nn

from dynamics.simplefeedforward.dynamics_model import DynamicsModelDelta
from worldmodels.fetchpush_ensembledelta import FetchPushEnsembleDelta
# from worldmodels.fetchreach_ensembledelta import FetchReachEnsembleDelta
import copy
import sys

"""
ddpg with HER (MPI-version)

"""
class ddpg_differ_real_delta:
    def __init__(self, args, env, env_params):
        self.args = args
        self.real_env = env
        self.env_params = env_params
        
        obs = self.real_env.reset()
        self.obs_size = obs['observation'].shape[0] # + obs['desired_goal'].shape[0]
        self.act_size = self.real_env.action_space.shape[0]
        # setup dynamics model
        self.ensemble_models, self.ensemble_optimizers = self.create_ensemble(ensemble_size=self.args.ensemble_size)
        self.mseloss = nn.MSELoss()
        # setup real replay buffer and obs rms
        self.real_o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.real_delta_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.buffer_real_dynamics = BasicReplayBuffer(self.obs_size, self.act_size, self.args.buffer_size)
        self.old_actors = []
        self.ri = 0
        self.times_updated = 0
        
        # her sampler
        if args.simulator != 'mujoco':
            self.real_env.compute_reward = goal_distance_reward
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.real_env.compute_reward)
        # Real buffer for ddpg use
        self.buffer_real_ddpg = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # Add an input to tell agent if it is in real or imagination (env_is_real)
        agent_env_params = env_params.copy()
        agent_env_params['obs'] += 1
        
        # setup imagined env
        self.imag_env = FetchPushEnsembleDelta(self.ensemble_models, self.real_env, self.real_o_norm, self.real_delta_norm,\
                                               self.args.clip_range, real_buffer=self.buffer_real_ddpg, env_name=args.env_name)
        
        # create the network
        self.actor_network = actor(agent_env_params)
        self.critic_network = critic(agent_env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(agent_env_params)
        self.critic_target_network = critic(agent_env_params)
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
        # create the replay buffer
        self.buffer_imag = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        
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
        print('beginning differ_real_delta learning')
        for epoch in range(self.args.n_epochs):
            actor_loss, critic_loss, self.ri = [], [], 0
            sys.stdout.flush()
            sys.stderr.flush()
            # Update dynamics
            if epoch % self.args.epochs_per_dy_update == 0:
                # temp_real_exp = self.collect_real_exp(epoch)
                self.collect_real_exp(epoch)
                self.update_dynamics(epoch, batch_size=self.args.dynamics_batch_size)
                if epoch > 0 and self.args.refill:
                    refill_size = self.buffer_imag.current_size
                    # Clear imagined replay buffer and refill using updated dynamics model and all old actors
                    self.buffer_imag = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
                    self.refill_imagined_replay_buffer(refill_size)
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('[{}] buff size pre-clear: {}, post-refill: {}'.format(datetime.now(), refill_size, self.buffer_imag.current_size))
                refresh_norms = True
            # Update policy
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                # TODO: could parallelize imag data collection here
                for _ in range(self.args.num_rollouts_per_mpi):
                    if self.args.distinguish:
                        # Tell agent whether it is in imagination
                        env_is_real = np.random.binomial(1, self.args.p_imagwreal)
                    else:
                        # If not distinguishing, this input is always 0
                        env_is_real = 0
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.imag_env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            # Tell agent it is in imagination
                            input_tensor = torch.cat((input_tensor, torch.tensor(env_is_real, dtype=torch.float32).reshape(1,1)), dim=1)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.imag_env.step(action)
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
                # Only update normalizer with real experiences right before policy updates -
                # otherwise policy is not adjusted to new norms when collecting experience
                if refresh_norms:    
                    # self._update_normalizer(temp_real_exp)
                    # temp_real_exp = None
                    self._refresh_normalizer()
                    refresh_norms = False
                for _ in range(self.args.n_batches):
                    # train the network
                    a_loss, q_loss = self._update_network(epoch)
                    actor_loss += [a_loss]
                    critic_loss += [q_loss]
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # store current actor
            old_actor = {
              "actor": copy.deepcopy(self.actor_network), # actor(self.env_params).load_state_dict(self.actor_network.state_dict()),
              "o_mean": self.o_norm.mean.copy(),
              "o_std": self.o_norm.std.copy(),
              "g_mean": self.g_norm.mean.copy(),
              "g_std": self.g_norm.std.copy(),
            }
            self.old_actors += [old_actor]
            # start to do the evaluation
            imag_success_rate = self._eval_agent(real=False) # TODO: parallelize
            real_success_rate = self._eval_agent(real=True)
            self._update_logs(epoch, imag_success_rate, real_success_rate, np.mean(actor_loss), np.mean(critic_loss))
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch: {} imag rate: {:.2f} real rate: {:.2f} ri: {:.4f} a_loss: {:.2f} q_loss {:.2f}'.format(datetime.now(), epoch, imag_success_rate, real_success_rate, self.ri, np.mean(actor_loss), np.mean(critic_loss)))
                np.save(self.model_path + '/logs.npy', self.logs)
                if self.args.save_models == 1 and (epoch % self.args.save_every) == 0:
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                                self.model_path + '/ac_epoch{}.pt'.format(epoch))

    def _update_logs(self, epoch, imag_success_rate, real_success_rate, actor_loss, critic_loss):
        self.ri = self.ri / (self.args.n_batches * self.args.n_cycles)
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
                                            size=action.shape)
        # choose if use the random actions
        shape = list(action.shape)
        shape[-1] = 1
        choice = np.random.binomial(1, self.args.random_eps, tuple(shape)) * (random_actions - action)
        action += choice
        return action

    def _refresh_normalizer(self):
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)
        buffers = [self.buffer_imag, self.buffer_real_ddpg]
        for buffer in buffers:
            batch = [buffer.buffers[key][:buffer.current_size] for key in buffer.buffers.keys()]
            self._update_normalizer(batch)

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0] * mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions) # mistake???
        # obs, g = transitions['obs'], transitions['g'] # TODO: use mb_obs?
        obs, g = mb_obs, transitions['g']
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
    def _update_network(self, epoch):
        # TODO: fix proportions
        total_imag_rollouts = (epoch+1) * self.args.n_cycles *self.args.num_rollouts_per_mpi
        proportion_real = self.buffer_real_ddpg.current_size / (total_imag_rollouts + self.buffer_real_ddpg.current_size)
        proportion_real = np.clip(self.args.bias_real * proportion_real, 0, 0.9)
        # proportion_real = self.buffer_real_ddpg.current_size / (self.buffer_imag.current_size + self.buffer_real_ddpg.current_size)
        batch_size_real = np.max((int(proportion_real * self.args.batch_size), 1))
        batch_size_imag = int((1 - proportion_real) * self.args.batch_size)
        # sample the episodes
        transitions_imag = self.buffer_imag.sample(batch_size_imag)
        transitions_real = self.buffer_real_ddpg.sample(batch_size_real)
        # encode whether transition is real 
        transitions = {}
        for key in transitions_imag.keys():
            transitions[key] = np.concatenate((transitions_imag[key], transitions_real[key]), axis=0)
        if self.args.include_ri:
            r_intrinsic = self.get_intrinsic_reward(transitions['obs'], transitions['actions'])
            transitions['r'] += r_intrinsic
            self.ri += np.mean(r_intrinsic)
        # print('(real) transitions[obs][batch_size_imag,0]: {}'.format(transitions['obs'][batch_size_imag,0]))
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
        if self.args.distinguish:
            # Inform whether real or imaginary
            # Remember extra input is env_is_real = 0 (imag) or 1 (real)
            imag = torch.zeros(batch_size_imag,1)
            real = torch.ones(batch_size_real,1)
            env_is_real = torch.cat((imag,real), dim=0)
        else:
            env_is_real = torch.zeros(batch_size_imag+batch_size_real,1)
        inputs_norm_tensor = torch.cat((inputs_norm_tensor,env_is_real), dim=1)
        inputs_next_norm_tensor = torch.cat((inputs_next_norm_tensor,env_is_real), dim=1)
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
        return actor_loss.detach().numpy(), critic_loss.detach().numpy()

    def get_intrinsic_reward(self, obs, action, scale=0.5, clip=0.8):
        obs = self.real_o_norm.normalize(obs)
        predictions = []
        for model in self.ensemble_models:
            predictions += [model(torch.tensor(obs), torch.tensor(action)).detach().numpy()]
        predictions = np.clip(np.array(predictions), -self.args.clip_range, self.args.clip_range)
        mean_prediction = np.mean(predictions, axis=0)
        dist_from_mean = np.sum(np.square(predictions - mean_prediction), axis=-1) # TODO: mean rather than sum to ensure is agnostic to obs size
        r_intrinsic = np.expand_dims(np.mean(dist_from_mean, axis=0), axis=1)
        return np.clip(r_intrinsic*scale, 0, clip)
        
    def preproc_inputs_old(self, obs, g, o_mean, o_std, g_mean, g_std):
        o_clip = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
        g_clip = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        o_norm = np.clip((o_clip - o_mean) / (o_std), -self.args.clip_range, self.args.clip_range)
        g_norm = np.clip((g_clip - g_mean) / (g_std), -self.args.clip_range, self.args.clip_range)
        inputs = np.concatenate([o_norm, g_norm], axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,inputs.shape[-1])
        return inputs

    ################### New
    def refill_imagined_replay_buffer(self, refill_rollouts, max_batch_size=1024, max_refill_steps=1e6):
        # TODO: make more efficient by taking advantage of the seperate threads
        # Apply cap to refill
        refill_rollouts = np.minimum(refill_rollouts, int(max_refill_steps/self.env_params['max_timesteps']))
        # TODO: don't go over replay limit
        rollouts_per_epoch = self.args.n_cycles * self.args.num_rollouts_per_mpi
        rollouts_per_actor = int(self.args.bias_recent_refill * rollouts_per_epoch)
        assert rollouts_per_actor < max_batch_size
        refill_actors = int(np.ceil(refill_rollouts / rollouts_per_actor)) # ceil means it over-fills when decimal number
        actor_idx = len(self.old_actors) - refill_actors
        # Start from earliest actor so experience is stored in buffer in correct order
        while actor_idx < len(self.old_actors):
            old_actor = self.old_actors[actor_idx]
            actor_idx += 1
            actor = old_actor['actor']
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            if self.args.distinguish:
                # Tell agent it's in imagination
                env_is_real = 0 # TODO: tell agent its real sometimes!!!!!
                env_is_real = np.random.binomial(1, self.args.p_imagwreal, size=(rollouts_per_actor, 1))
            else:
                env_is_real = np.zeros((rollouts_per_actor, 1))
            # Begin parallelized rollouts
            observation = self.imag_env.reset(rollouts=rollouts_per_actor)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self.preproc_inputs_old(obs, g, old_actor['o_mean'], old_actor['o_std'], old_actor['g_mean'], old_actor['g_std'])
                    # Tell agent it's in imagination
                    input_tensor = torch.cat((input_tensor, torch.ones(input_tensor.shape[0], 1) * env_is_real), dim=-1).float()
                    pi = actor(input_tensor)
                    action = self._select_actions(pi)
                    # feed the actions into the environment
                    observation_new, _, _, info = self.imag_env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                mb_obs.append(obs.copy())
                mb_ag.append(ag.copy())
                mb_g.append(g.copy())
                mb_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            mb_obs.append(obs.copy())
            mb_ag.append(ag.copy())
            # convert them into arrays
            mb_obs = np.transpose(mb_obs, (1, 0, 2))
            mb_ag = np.transpose(mb_ag, (1, 0, 2))
            mb_g = np.transpose(mb_g, (1, 0, 2))
            mb_actions = np.transpose(mb_actions, (1, 0, 2))
            # store the episodes
            self.buffer_imag.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            # self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions]) # Not performed in order to allow normalizer to update gradually
            # self.imag_env.render_mb(mb_obs, mb_ag, mb_g)
            # TODO: clear normaliser and re-initialise?
            if self.buffer_imag.current_size >= (max_refill_steps // self.env_params['max_timesteps']):
                return


    def create_ensemble(self, ensemble_size=5): # TODO: size back to 5!!!!!!
        ensemble_models = []
        ensemble_optimizers = []
        for i in range(ensemble_size):
            model = DynamicsModelDelta(self.obs_size, self.act_size, hiddens=self.args.dynamics_hiddens)
            sync_networks(model) #TODO: verify these sync properly!!!!!!!
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            ensemble_models += [model]
            ensemble_optimizers += [optimizer]
        return ensemble_models, ensemble_optimizers          

    
    def update_dynamics(self, epoch, batch_size=512, scale_initial=5):
        steps = int(self.args.dynamics_steps/MPI.COMM_WORLD.Get_size())
        if epoch == 0:
            steps = steps * scale_initial
        e_loss = [0,0,0,0,0] # TODO: remove
        for j in range(len(self.ensemble_models)):
            cum_loss = 0
            for i in range(steps):
                batch = self.buffer_real_dynamics.sample_batch_biasrecent(batch_size, self.args.bias_recent_sample)
                obs, a, obs2 = batch['obs'], batch['act'], batch['obs2']
                delta = obs2 - obs
                obs = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
                obs_norm = torch.tensor(self.real_o_norm.normalize(obs.detach().numpy())).float()
                delta_norm = torch.tensor(self.real_delta_norm.normalize(delta.detach().numpy())).float()
                
                self.ensemble_optimizers[j].zero_grad()
                delta_pred = self.ensemble_models[j](obs_norm, a)
                loss = self.mseloss(delta_pred.float(), delta_norm.float())
                if not self.args.squared_l2_loss:
                    loss = torch.sqrt(loss)
                loss.backward()
                sync_grads(self.ensemble_models[j])
                self.ensemble_optimizers[j].step()
                
                cum_loss += loss.item()
            e_loss[j] += cum_loss
        self.times_updated += 1
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] Steps: {}, Prediction loss: {:.4f}'.format(datetime.now(), steps, e_loss[0]/steps))
            if self.args.save_models == 1:
                model_state = {
                    "obs_mean": self.real_o_norm.mean,
                    "obs_std": self.real_o_norm.std,
                    "delta_mean": self.real_delta_norm.mean,
                    "delta_std": self.real_delta_norm.std
                    }
                for i in range(len(self.ensemble_models)):
                    model_state['state_dict{}'.format(i)] = self.ensemble_models[i].state_dict()
                    model_state['optimizer{}'.format(i)] = self.ensemble_optimizers[i].state_dict()
                torch.save(model_state, self.model_path + '/Eof{}_epoch{}.tar'.format(len(self.ensemble_models), epoch))

    def collect_real_exp(self, epoch):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        rollouts = int(np.ceil(self.args.real_rollouts/MPI.COMM_WORLD.Get_size())) # comment to explain division!!
        # Collect a certain amount under real and certain under imag policy
        real_or_imag = np.ones(rollouts) # proportion 'real'
        real_or_imag[:int(rollouts*(1-self.args.p_realwreal))] = 0 # proportion 'imag'
        for r in range(rollouts):
            if self.args.distinguish:
                # Decide whether to act under real or imag policy
                env_is_real = real_or_imag[r]
            else:
                env_is_real = 0
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.real_env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if epoch == 0:
                    action = self.real_env.action_space.sample()
                else:
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g)
                        # Tell agent its in real env
                        input_tensor = torch.cat((input_tensor, torch.tensor(env_is_real, dtype=torch.float32).reshape(1,1)), dim=1)
                        pi = self.actor_network(input_tensor)
                        action = self._select_actions(pi)
                # feed the actions into the environment
                observation_new, _, _, _ = self.real_env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # Store experience to replay buffer
                self.buffer_real_dynamics.store(obs, action, obs_new, self.times_updated)
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
        self.buffer_real_ddpg.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        # self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions]) # Normalizer is only updated with this exp after refills
        # Calculate new individual obs mean and std
        mb_obs = np.clip(mb_obs, -self.args.clip_obs, self.args.clip_obs)
        self.real_o_norm.update(mb_obs)
        self.real_o_norm.recompute_stats()
        # delta = ob2 - obs1
        mb_delta = mb_obs[:,1:,:] - mb_obs[:,:-1,:]
        self.real_delta_norm.update(mb_delta)
        self.real_delta_norm.recompute_stats()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('[{}] rollouts collected: {}'.format(datetime.now(), self.args.real_rollouts))
        return [mb_obs, mb_ag, mb_g, mb_actions]
        
    # def test_dynamics(self):
    #     for _ in range(5):
    #         input()
    #         observation = self.real_env.reset()
    #         self.imag_env.reset()
    #         obs = observation['observation']
    #         g = observation['desired_goal']
    #         for _ in range(self.env_params['max_timesteps']):
    #             action = self.real_env.action_space.sample()
    #             observation, _, _, info = self.real_env.step(action)
    #             obs = observation['observation']
    #             self.imag_env.step(action)
    #             self.imag_env.render()
    #             self.real_env.render()

    # def load_ac(self):
    #     # load the model param!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     model_path = 'dynamics/pickplace/ensemble13/controllers/model31dynamics145.pt'
    #     o_mean, o_std, g_mean, g_std, actor_state = torch.load(model_path, map_location=lambda storage, loc: storage)
    #     # create the actor network
    #     self.actor_network.load_state_dict(actor_state)
    #     # self.critic_network.load_state_dict(critic_state)
    #     # Load in norms!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     self.o_norm.mean, self.o_norm.std = o_mean, o_std
    #     self.g_norm.mean, self.g_norm.std = g_mean, g_std
        
    # def load_ensemble(self):
    #     # Setup ensemble dynamics imagination env
    #     obs = self.real_env.reset()
    #     obs_size = obs['observation'].shape[0]
    #     act_size = self.real_env.action_space.shape[0]
    #     dy_model_state = torch.load('dynamics/pickplace/ensemble13/ensembles/31EnsembleOf5_epoch145.tar')
    #     ensemble_models = []
    #     for i in range(5):
    #         dy_model = DynamicsModelDelta(obs_size, act_size)
    #         dy_model.load_state_dict(dy_model_state['state_dict{}'.format(i)])
    #         ensemble_models += [dy_model]
    #     # setup normaliser
    #     real_o_norm = normalizer(size=obs_size, default_clip_range=self.args.clip_range)
    #     real_o_norm.mean = dy_model_state['obs_mean']
    #     real_o_norm.std = dy_model_state['obs_std']
    #     real_delta_norm = normalizer(size=obs_size, default_clip_range=self.args.clip_range)
    #     real_delta_norm.mean = dy_model_state['delta_mean']
    #     real_delta_norm.std = dy_model_state['delta_std']
    #     # setup imagined env. Force gripper closed, as in push env
    #     self.imag_env = FetchPushEnsembleDelta(ensemble_models, self.real_env, real_o_norm, real_delta_norm, self.args.clip_range)

    # do the evaluation
    def _eval_agent(self, real=False, render=False):
        if self.args.distinguish:
            # Tell agent whether in real or imag
            env_is_real = real * 1
        else:
            env_is_real = 0
        if real:
            env = self.real_env
        else:
            env = self.imag_env
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    # Tell agent whether real or imag
                    input_tensor = torch.cat((input_tensor, torch.tensor(env_is_real, dtype=torch.float32).reshape(1,1)), dim=1)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, r, _, _ = env.step(actions)
                if render:
                    env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(r+1) # info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
