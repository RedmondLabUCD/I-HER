import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer, BasicReplayBuffer
from rl_modules.models import actor, critic

from mpi_utils.normalizer import normalizer

from her_modules.her import her_sampler

import torch.nn as nn

from dynamics.simplefeedforward.dynamics_model import DynamicsModel
from worldmodels.fetchpush_ensemble import FetchPushEnsemble
# from worldmodels.fetchreach_ensemble import FetchReachEnsemble
import copy
import sys

"""
ddpg with HER (MPI-version)
In this we tell agent whether its in real env or imagination

Collect real data continously alongside imaginary data

"""
class ddpg_differ_real_imag2:
    def __init__(self, args, env, env_params):
        self.args = args
        self.real_env = env
        self.env_params = env_params
        
        obs = self.real_env.reset()
        self.obs_size = obs['observation'].shape[0] # + obs['desired_goal'].shape[0]
        self.act_size = self.real_env.action_space.shape[0]
        # setup dynamics model
        self.ensemble_models, self.ensemble_optimizers = self.create_ensemble()
        self.mseloss = nn.MSELoss()
        # setup real replay buffer and obs rms
        self.real_o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.buffer_real_dynamics = BasicReplayBuffer(self.obs_size, self.act_size, self.args.buffer_size)
        # setup imagined env
        self.imag_env = FetchPushEnsemble(self.ensemble_models, self.real_env, self.real_o_norm, self.args.clip_range)
        self.old_actors = []
        self.include_intrinsic = True
        self.ri = 0
        self.proportion_real = 0.1
        self.real_imag_sequence = self.get_real_imag_sequence()
        
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.real_env.compute_reward)
        # Real buffer for ddpg use
        self.buffer_real_ddpg = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # Add an input to tell agent if it is in real or imagination (env_is_real)
        agent_env_params = env_params.copy()
        agent_env_params['obs'] += 1
        
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
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network
        
        """
        print('beginning differ_real_imag2 learning')
        self.update_dynamics_every = 5
        self.collect_real_exp(first=True)
        self.update_dynamics(0)
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            self.ri = 0
            sys.stdout.flush()
            sys.stderr.flush()
            if epoch % self.update_dynamics_every == 0 and epoch > 1:
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
            for i in range(self.args.n_cycles):
                # Choose whether to do a real or imag rollout
                # env_is_real = np.random.choice(2, p=[1-self.proportion_real, self.proportion_real])
                env_is_real = self.real_imag_sequence[i]
                if env_is_real:
                    self.collect_real_exp(rollouts=self.args.num_rollouts_per_mpi)
                else:
                    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                    for _ in range(self.args.num_rollouts_per_mpi):
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
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions]) # Note: real exps being normalised by this also in collect_real_exp()
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network(epoch)
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            imag_success_rate = self._eval_agent(real=False)
            real_success_rate = self._eval_agent(real=True)
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.ri = self.ri / (self.args.n_batches * self.args.num_rollouts_per_mpi * self.args.n_cycles)
                print('[{}] epoch: {}, imag rate: {:.2f} , real rate: {:.2f} , ri: {:.4f}'.format(datetime.now(), epoch, imag_success_rate, real_success_rate, self.ri))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model24dynamics{}.pt'.format(epoch))

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
        # TODO: is this a mistake? Should use all rolouts for normalization? i.e. mb_actions.shape[0]*mb_actions.shape[1]
        # Or do per episode rather than batch of episodes
        # Could be issue for: refilling replay buffer and initial real collection
        # Good for goals though
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
    def _update_network(self, epoch):
        batch_size_real = int(self.proportion_real * self.args.batch_size) # TODO: make these batch sizes more accurate when proportion real and cyles don't perfectly divide
        batch_size_imag = int((1 - self.proportion_real) * self.args.batch_size)
        # sample the episodes
        transitions_imag = self.buffer_imag.sample(batch_size_imag)
        transitions_real = self.buffer_real_ddpg.sample(batch_size_real)
        # encode whether transition is real 
        transitions = {}
        for key in transitions_imag.keys():
            transitions[key] = np.concatenate((transitions_imag[key], transitions_real[key]), axis=0)
        if self.include_intrinsic:
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
        # Inform whether real or imaginary
        # Remember extra input is 'env_is_real = 0 or 1'
        imag = torch.zeros(batch_size_imag,1)
        real = torch.ones(batch_size_real,1)
        env_is_real = torch.cat((imag,real), dim=0)
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

    def get_intrinsic_reward(self, obs, action, scale=2, clip=0.8):
        obs = self.real_o_norm.normalize(obs)
        predictions = []
        for model in self.ensemble_models:
            predictions += [model(torch.tensor(obs), torch.tensor(action)).detach().numpy()]
        predictions = np.clip(np.array(predictions), -self.args.clip_range, self.args.clip_range)
        mean_prediction = np.mean(predictions, axis=0)
        dist_from_mean = np.sum(np.square(predictions - mean_prediction), axis=-1)
        r_intrinsic = np.expand_dims(np.mean(dist_from_mean, axis=0), axis=1)
        return np.clip(r_intrinsic/scale, 0, clip)
        
    def preproc_inputs_old(self, obs, g, o_mean, o_std, g_mean, g_std):
        o_clip = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
        g_clip = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        o_norm = np.clip((o_clip - o_mean) / (o_std), -self.args.clip_range, self.args.clip_range)
        g_norm = np.clip((g_clip - g_mean) / (g_std), -self.args.clip_range, self.args.clip_range)
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def refill_imagined_replay_buffer(self, epoch, max_refill_steps=1e6):
        # Tell agent it's in imagination
        env_is_real = 0
        # TODO: don't go over replay limit
        rollouts_per_epoch = int(self.args.n_cycles * self.args.num_rollouts_per_mpi * (1 - self.proportion_real))
        rollouts_so_far = epoch * rollouts_per_epoch
        refill_rollouts = np.min((rollouts_so_far, max_refill_steps//self.env_params['max_timesteps'], self.buffer_imag.size))
        refill_actors = int(refill_rollouts / rollouts_per_epoch)
        actor_idx = len(self.old_actors) - refill_actors
        while actor_idx < len(self.old_actors):
            old_actor = self.old_actors[actor_idx]
            actor_idx += 1
            actor = old_actor['actor']
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            for _ in range(rollouts_per_epoch):
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
                        input_tensor = self.preproc_inputs_old(obs, g, old_actor['o_mean'], old_actor['o_std'], old_actor['g_mean'], old_actor['g_std'])
                        # Tell agent it's in imagination
                        input_tensor = torch.cat((input_tensor, torch.tensor(env_is_real, dtype=torch.float32).reshape(1,1)), dim=1)
                        pi = actor(input_tensor)
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
            # TODO: clear normaliser and re-initialise?
            if self.buffer_imag.current_size >= (max_refill_steps // self.env_params['max_timesteps']):
                return

    def create_ensemble(self, ensemble_size=5):
        ensemble_models = []
        ensemble_optimizers = []
        for i in range(ensemble_size):
            model = DynamicsModel(self.obs_size, self.act_size)
            sync_networks(model) #TODO: verify these sync properly!!!!!!!
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            ensemble_models += [model]
            ensemble_optimizers += [optimizer]
        return ensemble_models, ensemble_optimizers          

    def update_dynamics(self, epoch, batch_size=512, steps=2e4, scale_initial=5):
        steps = int(steps/MPI.COMM_WORLD.Get_size())
        if epoch == 0:
            steps = steps * scale_initial
        e_loss = [0,0,0,0,0] # TODO: remove
        for j in range(len(self.ensemble_models)):
            cum_loss = 0
            for i in range(steps):
                batch = self.buffer_real_dynamics.sample_batch_biasrecent(batch_size) # TODO: revert!!
                obs, a, obs2 = batch['obs'], batch['act'], batch['obs2']
                obs = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
                obs2 = np.clip(obs2, -self.args.clip_obs, self.args.clip_obs)
                obs_norm = torch.tensor(self.real_o_norm.normalize(obs.detach().numpy())).float()
                obs2_norm = torch.tensor(self.real_o_norm.normalize(obs2.detach().numpy())).float()
                
                self.ensemble_optimizers[j].zero_grad()
                predictions = self.ensemble_models[j](obs_norm, a)
                loss = self.mseloss(predictions.float(), obs2_norm.float())
                loss.backward()
                sync_grads(self.ensemble_models[j])
                self.ensemble_optimizers[j].step()
                
                cum_loss += loss.item()
            e_loss[j] += cum_loss
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('Prediction loss: {}'.format(e_loss[0]/steps))
            print('buffer_real_ddpg.current_size: {}'.format(self.buffer_real_ddpg.current_size))
            model_state = {
                "means": self.real_o_norm.mean,
                "stds": self.real_o_norm.std
                }
            for i in range(len(self.ensemble_models)):
                model_state['state_dict{}'.format(i)] = self.ensemble_models[i].state_dict()
                model_state['optimizer{}'.format(i)] = self.ensemble_optimizers[i].state_dict()
            torch.save(model_state, '24EnsembleOf{}_epoch{}.tar'.format(len(self.ensemble_models), epoch))

    def collect_real_exp(self, first=False, rollouts=512):
        # TODO: Just use a single buffer
        # Tell agent its in real env
        env_is_real = 1
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        if first:
            rollouts = int(rollouts/MPI.COMM_WORLD.Get_size())
        for _ in range(rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.real_env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                if first:
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
                self.buffer_real_dynamics.store(obs, action, obs_new)
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
        self.buffer_real_ddpg.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        # Update agents normalizer
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        # Update dynamics model normalizer
        mb_obs = np.clip(mb_obs, -self.args.clip_obs, self.args.clip_obs)
        self.real_o_norm.update(mb_obs)
        self.real_o_norm.recompute_stats()
    
    def get_real_imag_sequence(self):
        num_real = self.proportion_real * self.args.n_cycles
        step = int(np.round(self.args.n_cycles / num_real))
        sequence = np.zeros(self.args.n_cycles)
        i = self.args.n_cycles - 1
        while i > 0:
            sequence[i] = 1
            i -= step
        return sequence
        
    # def test_dynamics(self):
    #     for _ in range(5):
    #         observation = self.realenv.reset()
    #         self.env.reset()
    #         obs = observation['observation']
    #         g = observation['desired_goal']
    #         for _ in range(self.env_params['max_timesteps']):
    #             with torch.no_grad():
    #                 input_tensor = self._preproc_inputs(obs, g)
    #                 pi = self.actor_network(input_tensor)
    #                 action = self._select_actions(pi)
    #             observation, _, _, info = self.realenv.step(action)
    #             obs = observation['observation']
    #             self.env.step(action)
    #             self.env.render()
    #             self.realenv.render()

    # do the evaluation
    def _eval_agent(self, real=False):
        # Tell agent whether in real or imag
        env_is_real = real * 1
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
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(r+1) # info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
