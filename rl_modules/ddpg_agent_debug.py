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

from worldmodels.mdrnn import MDRNNCell, MDRNN, gmm_loss
# from worldmodels.fetchreach_rnn import FetchReachRNN
from worldmodels.fetchpush_rnn import FetchPushRNN
import torch.nn.functional as f
from worldmodels.misc import save_checkpoint
from pprint import pprint

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.realenv = env
        
        # setup mdrnn
        obs = self.realenv.reset()
        self.obs_size = obs['observation'].shape[0] # + obs['desired_goal'].shape[0]
        self.act_size = self.realenv.action_space.shape[0]
        device = torch.device('cpu')
        RSIZE = 256
        rnn_file = 'worldmodels/exp_dir_push/mdrnn256_push_goal_obsonly_noreward/pretrained.tar'
        rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})
        print("Loading MDRNN at epoch {} with test loss {}".format(rnn_state["epoch"], rnn_state["precision"]))
        self.mdrnn = MDRNN(self.obs_size, self.act_size, RSIZE, 5).float().to(device)
        self.mdrnn_cell = MDRNNCell(self.obs_size, self.act_size, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(rnn_state['state_dict'])
        self.mdrnn_cell.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        self.mdrnn_optimizer = torch.optim.RMSprop(self.mdrnn.parameters(), lr=1e-3, alpha=.9)
        self.mdrnn_optimizer.load_state_dict(rnn_state["optimizer"])
        
        # setup imagined env
        self.env = FetchPushRNN(self.mdrnn_cell, self.realenv)
        
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
        # create the replay buffers
        self.buffer_imag = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        self.buffer_real = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
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
        # self.test_mdrnn()
        self.collect_real_exp()
        self.update_mdrnn()
        # self.test_mdrnn()
        return
        
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            if epoch % 10 == 0:
                self.collect_real_exp()
                self.update_mdrnn()
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
                            self.model_path + '/modelimagine2.pt')

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

    def update_mdrnn(self):
        batch_size = 10 # TODO: make args
        steps = 1000
        cum_loss = 0
        verbose = False
        for i in range(steps):
            print('\nStep {}'.format(i))
            nanweights = np.sum(np.isnan(self.mdrnn.rnn.weight_ih_l0.detach().numpy())) + np.sum(np.isnan(self.mdrnn.rnn.weight_hh_l0.detach().numpy())) + np.sum(np.isnan(self.mdrnn.rnn.bias_ih_l0.detach().numpy())) + np.sum(np.isnan(self.mdrnn.rnn.bias_hh_l0.detach().numpy()))
            print('RNN weight nans:{}'.format(nanweights))
            gmmweights = np.sum(np.isnan(self.mdrnn.gmm_linear.weight.detach().numpy())) + np.sum(np.isnan(self.mdrnn.gmm_linear.bias.detach().numpy()))
            print('Gmm weight nans: {}'.format(gmmweights))
            
            infweights = np.sum(np.isinf(self.mdrnn.rnn.weight_ih_l0.detach().numpy())) + np.sum(np.isinf(self.mdrnn.rnn.weight_hh_l0.detach().numpy())) + np.sum(np.isinf(self.mdrnn.rnn.bias_ih_l0.detach().numpy())) + np.sum(np.isinf(self.mdrnn.rnn.bias_hh_l0.detach().numpy()))
            print('RNN weight infs:{}'.format(infweights))
            infgmmweights = np.sum(np.isinf(self.mdrnn.gmm_linear.weight.detach().numpy())) + np.sum(np.isinf(self.mdrnn.gmm_linear.bias.detach().numpy()))
            print('Gmm weight infs: {}'.format(infgmmweights))
            # weightsmean = np.mean(self.mdrnn.rnn.weight_ih_l0.squeeze().detach().numpy()) + np.mean(self.mdrnn.rnn.weight_hh_l0.squeeze().detach().numpy()) + np.mean(self.mdrnn.rnn.bias_ih_l0.squeeze().detach().numpy()) + np.mean(self.mdrnn.rnn.bias_hh_l0.squeeze().detach().numpy())
            # print('\nRNN weights mean:{}'.format(weightsmean))
            # gmmweights = np.mean(self.mdrnn.gmm_linear.weight.squeeze().detach().numpy()) + np.mean(self.mdrnn.gmm_linear.bias.squeeze().detach().numpy())
            # print('Gmm weights contains nan: {}'.format(gmmweights))
            # if i > 1:
            #     print('Mean grads: {}'.format(np.mean(self.mdrnn.gmm_linear.weight.grad.numpy())))
            #     print('Min grads: {}'.format(np.amin(self.mdrnn.gmm_linear.weight.grad.numpy())))
            #     print('Max grads: {}'.format(np.amax(self.mdrnn.gmm_linear.weight.grad.numpy())))
            #     print('Min grads: {}'.format(np.amin(np.abs(self.mdrnn.gmm_linear.weight.grad.numpy()))))
            batch = self.buffer_real.mdrnn_sample(batch_size)
            obs, action, next_obs = batch['obs'], batch['actions'], batch['obs_next']
            reward = torch.zeros(obs.shape[0],obs.shape[1])
            terminal = torch.zeros(obs.shape[0],obs.shape[1])
            losses = self.get_mdrnn_loss(torch.tensor(obs), torch.tensor(action), reward, terminal, torch.tensor(next_obs), include_reward=False, verbose=verbose)
            print('step {} loss: {}'.format(i, losses['loss']))
            self.mdrnn_optimizer.zero_grad()
            losses['loss'].backward()
            
            rnngradsnan = np.sum(np.isnan(self.mdrnn.rnn.weight_ih_l0.grad.numpy())) + np.sum(np.isnan(self.mdrnn.rnn.weight_hh_l0.grad.numpy())) + np.sum(np.isnan(self.mdrnn.rnn.bias_ih_l0.grad.numpy())) + np.sum(np.isnan(self.mdrnn.rnn.bias_hh_l0.grad.numpy()))
            gmmgradnan = np.sum(np.isnan(self.mdrnn.gmm_linear.weight.detach().numpy())) + np.sum(np.isnan(self.mdrnn.gmm_linear.bias.detach().numpy()))
            print('RNN grad nans:{}'.format(rnngradsnan))
            print('Gmm grad nans:{}'.format(gmmgradnan))
            
            rnngradsnan = np.sum(np.isinf(self.mdrnn.rnn.weight_ih_l0.grad.numpy())) + np.sum(np.isinf(self.mdrnn.rnn.weight_hh_l0.grad.numpy())) + np.sum(np.isinf(self.mdrnn.rnn.bias_ih_l0.grad.numpy())) + np.sum(np.isinf(self.mdrnn.rnn.bias_hh_l0.grad.numpy()))
            gmmgradnan = np.sum(np.isinf(self.mdrnn.gmm_linear.weight.detach().numpy())) + np.sum(np.isinf(self.mdrnn.gmm_linear.bias.detach().numpy()))
            print('RNN grad infs:{}'.format(rnngradsnan))
            print('Gmm grad infs:{}'.format(gmmgradnan))
            
            self.mdrnn_optimizer.step()
            cum_loss += losses['loss']
            
            if np.isnan(losses['loss'].detach().numpy()):
                if verbose == True:
                    i = 10000
                    break
                verbose = True
                
        print('MDRNN loss: {}'.format(cum_loss/steps))
        self.env.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in self.mdrnn.state_dict().items()})
        checkpoint_fname = self.model_path + '/mdrnn_checkpoint.tar'
        save_checkpoint({
            "state_dict": self.mdrnn.state_dict(),
            "optimizer": self.mdrnn_optimizer.state_dict(),
            "epoch": 0}, 0, checkpoint_fname,
                        self.model_path + 'mdrnn_best.tar')

    def get_mdrnn_loss(self, obs, action, reward, terminal,
             next_obs, include_reward: bool, verbose=False):
      
        obs, action,\
            reward, terminal,\
            next_obs = [arr.transpose(1, 0)
                               for arr in [obs, action,
                                           reward, terminal,
                                           next_obs]]
        mus, sigmas, logpi, rs, ds = self.mdrnn(action, obs, verbose=verbose)
        # if verbose or np.sum(np.isnan(mus.detach().numpy())) != 0 or True:
        #     print('\nget_mdrnn_loss:')
        #     print('mus: contains nan {}'.format(np.sum(np.isnan(mus.detach().numpy()))))
        #     print('sigmas: contains nan {}'.format(np.sum(np.isnan(sigmas.detach().numpy()))))
        #     print('logpi: contains nan {}'.format(np.sum(np.isnan(logpi.detach().numpy()))))
        #     print('rs: contains nan {}'.format(np.sum(np.isnan(rs.detach().numpy()))))
        #     print('ds: contains nan {}'.format(np.sum(np.isnan(ds.detach().numpy()))))
        if verbose or np.sum(np.isnan(mus.detach().numpy())) != 0 or True:
            print('get_mdrnn_loss:')
            print('mus: contains inf: {}'.format(np.sum(np.isinf(mus.detach().numpy()))))
            print('sigmas: contains inf: {}/{}'.format(np.sum(np.isinf(sigmas.detach().numpy())), sigmas.shape[0]*sigmas.shape[1]*sigmas.shape[2]*sigmas.shape[3]))
            print('logpi: contains inf: {}'.format(np.sum(np.isinf(logpi.detach().numpy()))))
            print('obs max: {}, action max: {}'.format(torch.amax(obs), torch.amax(action)))
        for i in range(10):
            print('rollout {} sigmas contains inf: {}/{}'.format(i, np.sum(np.isinf(sigmas[:,i,:,:].detach().numpy())), sigmas[:,i,:,:].shape[0]*sigmas[:,i,:,:].shape[1]*sigmas[:,i,:,:].shape[2]))
            if i == 6:
                for j in range(50):
                    print('frame {} sigmas contains inf: {}/{}'.format(j, np.sum(np.isinf(sigmas[j,i,:,:].detach().numpy())), sigmas[j,i,:,:].shape[0]*sigmas[j,i,:,:].shape[1]))
            
        gmm = gmm_loss(next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        if verbose or np.sum(np.isnan(gmm.detach().numpy())) != 0 or True:
            print('gmm_loss nan: {}'.format(1*np.isnan(gmm.detach().numpy())))
            print('bce_loss nan: {}'.format(1*np.isnan(bce.detach().numpy())))
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = self.obs_size + 2
        else:
            mse = 0
            scale = self.obs_size + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

    def collect_real_exp(self):
        rollouts = 100 # TODO: make arg
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for i in range(rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.realenv.reset()
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
                observation_new, _, _, info = self.realenv.step(action)
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
        print('obs nans before storage: {}'.format(np.sum(np.isnan(mb_obs))))
        # store the episodes
        self.buffer_real.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        
    def test_mdrnn(self):
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
