print('line 1')
import numpy as np
import gym
import os, sys
from arguments import get_args
print('line 6')
from mpi4py import MPI
print('line 8')
# from rl_modules.ddpg_agent import ddpg_agent
# from rl_modules.ddpg_agent_simpledynamics import ddpg_agent_simpledynamics
# from rl_modules.ddpg_agent_iter_dynamics import ddpg_agent_iter_dynamics
# from rl_modules.ddpg_agent_explore_exploit import ddpg_agent_explore_exploit
# from rl_modules.ddpg_agent_ensemble import ddpg_agent_ensemble
# from rl_modules.ddpg_imag_transfer import ddpg_imag_transfer
# from rl_modules.ddpg_agent_real_imag import ddpg_agent_real_imag
# from rl_modules.ddpg_differ_real_imag import ddpg_differ_real_imag
# from rl_modules.ddpg_differ_real_imag2 import ddpg_differ_real_imag2
from rl_modules.ddpg_differ_real_delta import ddpg_differ_real_delta
# from rl_modules.ddpg_real_transfer import ddpg_real_transfer

import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    args.env_name = 'FetchReach-v1' # TODO: UNDO!!!
    # create the ddpg_agent
    if args.simulator == 'mujoco':
        print('Creating {}...'.format(args.env_name))
        env = gym.make(args.env_name)
        print('Created {}'.format(args.env_name))
    elif args.simulator == 'bullet':
        import pybullet_multigoal_gym as pmg
        env = pmg.make_env(task=args.env_name)
    else:
        assert False, 'Simulator must be Mujoco or PyBullet'
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('############ ARGUMENTS:\n{}\n'.format(args))
        
    # seed = args.seed * 10000 * MPI.COMM_WORLD.Get_rank()
    seed = (args.seed * 100) + MPI.COMM_WORLD.Get_rank()
    # set random seeds for reproduce
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        
    # get the environment parameters
    env_params = get_env_params(env)
    
    # create the ddpg agent to interact with the environment 
    # ddpg_trainer = ddpg_agent(args, env, env_params)
    # ddpg_trainer = ddpg_imag_transfer(args, env, env_params)
    ddpg_trainer = ddpg_differ_real_delta(args, env, env_params)
    # ddpg_trainer = ddpg_agent_simpledynamics(args, env, env_params)
    
    ddpg_trainer.learn()

if __name__ == '__main__':
    print('begin...')
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    print('get args...')
    # get the params
    args = get_args()
    print('launch...')
    launch(args)
