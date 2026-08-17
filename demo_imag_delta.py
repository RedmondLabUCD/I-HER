
import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
# from worldmodels.fetchreach_rnn import FetchReachRNN
# from worldmodels.mdrnn import MDRNNCell

from dynamics.simplefeedforward.dynamics_model import DynamicsModelDelta

# from worldmodels.fetchpush_dynamics import FetchPushDynamics
from mpi_utils.normalizer import normalizer
from worldmodels.fetchpush_ensembledelta import FetchPushEnsembleDelta
import time
from her_modules.her import goal_distance_reward
import random

# from mpi_utils.normalizer import RunningMeanStd

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args, encode=False, env_is_real=1):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm], axis=-1)
    inputs = torch.tensor(inputs, dtype=torch.float32).view(-1,inputs.shape[-1])
    if encode:
        inputs = torch.cat((inputs, torch.ones(inputs.shape[0], 1, dtype=torch.float32) * env_is_real), dim=-1)
    return inputs

def get_intrinsic_reward(ensemble, obs, action, real_o_norm, close_gripper=False):
    if close_gripper:
        action[..., -1] = -1
    obs = real_o_norm.normalize(obs)
    predictions = []
    for model in ensemble:
        predictions += [model(torch.tensor(obs), torch.tensor(action)).detach().numpy()]
    predictions = np.clip(np.array(predictions), -5, 5)
    # if np.amax(np.abs(predictions)) == 5: print('clipped')
    mean_prediction = np.mean(predictions, axis=0)
    dist_from_mean = np.sum(np.square(predictions - mean_prediction), axis=-1)
    r_intrinsic = np.mean(dist_from_mean)
    return np.clip(r_intrinsic * 0.5, 0, 100)

def get_pred_error(ensemble, obs, action, obs_next, o_norm, delta_normalizer):
    delta = obs_next - obs
    obs_norm = torch.tensor(o_norm.normalize(obs))
    delta_norm = delta_normalizer.normalize(delta)
    delta_pred = ensemble[1](obs_norm, torch.tensor(action, dtype=torch.float32))
    error = np.mean(np.square(delta_pred.detach().numpy() - delta_norm), axis=-1)
    return error
    
if __name__ == '__main__':
    print('demo begin')
    close_gripper = False
    encode = True
    model_path = 'dynamics/final/IHER/Pick/saved_models/ac_epoch80.pt'
    # model_path = 'dynamics/push/transfer3/controllers/model3transfer70.pt'
    dy_model_state = torch.load('dynamics/final/IHER/Pick/saved_models/Eof5_epoch80.tar')
    real_env = False
    env_is_real = False
    above_table_only = True
    
    
    
    args = get_args()
    if args.simulator == 'mujoco':
        env = gym.make(args.env_name)
    elif args.simulator == 'bullet':
        import pybullet_multigoal_gym as pmg
        env = pmg.make_env(task=args.env_name, render=False)
        env.compute_reward = goal_distance_reward
    else:
        assert False, 'Simulator must be Mujoco or PyBullet'
        
    # # Set seeds
    # env.seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    
    # load the model param
    # model_path = 'dynamics/push/val_patience50/ac_epoch9.pt'
    # model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # create the environment
    obs = env.reset()
    obs_size = obs['observation'].shape[0] # + obs['desired_goal'].shape[0]
    act_size = env.action_space.shape[0]
    intrinsic = False
    
    # Setup ensemble dynamics imagination env
    # dy_model_state = torch.load('dynamics/push/old_dy/ensembles/Eof5pushold_dy0.tar')
    # print(dy_model_state.keys())
    ensemble_models = []
    for i in range(5):
        dy_model = DynamicsModelDelta(obs_size, act_size)
        # print('Not loading in ensemble!!!')
        dy_model.load_state_dict(dy_model_state['state_dict{}'.format(i)])
        ensemble_models += [dy_model]
    # setup normalisers
    real_o_norm = normalizer(size=obs_size, default_clip_range=args.clip_range)
    real_o_norm.mean = dy_model_state['obs_mean']
    real_o_norm.std = dy_model_state['obs_std']
    real_delta_norm = normalizer(size=obs_size, default_clip_range=args.clip_range)
    real_delta_norm.mean = dy_model_state['delta_mean']
    real_delta_norm.std = dy_model_state['delta_std']
    if not real_env:
        # setup imagined env
        env = FetchPushEnsembleDelta(ensemble_models, env, real_o_norm, real_delta_norm, args.clip_range, close_gripper=close_gripper, render=True)
    intrinsic = True
    print('models loaded')
    
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    if encode:
        env_params['obs'] += 1
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    total_successes = 0
    args.demo_length = 10
    # t0 = time.time()
    pred_error, ris = [], []
    print('ac loaded')
    for i in range(args.demo_length):
        if i % 4 == 0:
            env_is_real = False
        else:
            env_is_real = True
        
        
        good_goal = False
        while not good_goal:
            print('Finding new goal...')
            observation = env.reset()
            # start to do the demo
            obs = observation['observation']
            g = observation['desired_goal']
            if above_table_only and g[-1] < 0.5:
                good_goal = False
            else:
                good_goal = True
        for t in range(env._max_episode_steps-10):
            env.render()
            if i == 0 and t == 0:
                input()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args, encode=encode, env_is_real=env_is_real)
            with torch.no_grad():
                pi = actor_network(inputs)
                action = pi.detach().numpy().squeeze()
                # action = env.action_space.sample()
                # put actions into the environment
                observation_new, reward, _, _ = env.step(action)
            if intrinsic:
                ri = get_intrinsic_reward(ensemble_models, obs, action, real_o_norm, close_gripper=close_gripper)
                # print('r_intrinsic: {}'.format(ri))
                error = get_pred_error(ensemble_models, obs, action, observation_new['observation'], real_o_norm, real_delta_norm)
                # print('pred_error: {}'.format(error))
                pred_error.append(error)
                ris.append(ri)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, reward==0))
        total_successes += reward + 1
    print('Success rate: {}%'.format((total_successes/args.demo_length)*100))
    print('Mean prediction error: {}'.format(np.mean(pred_error)))
    print('Mean prediction r_intrinsic: {}'.format(np.mean(ris)))
    # t1 = time.time()
    # print('Time taken for {} transitions: {} seconds'.format(args.demo_length*env._max_episode_steps, int(t1-t0)))
    env.close()