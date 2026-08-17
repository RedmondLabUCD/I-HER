import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np

import pybullet_multigoal_gym as pmg

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    # model_path = args.save_dir + args.env_name + '/model1.pt'
    model_path = 'saved_models/bullet/pick/baselines/default_seed/PickBaselineBullet_ac.pt'
    o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
    if args.simulator == 'mujoco':
        # create the environment
        env = gym.make(args.env_name)
    else:
        camera_setup = [
            {
                'cameraEyePosition': [-1.0, 0.25, 0.6],
                'cameraTargetPosition': [-0.6, 0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 128,
                'render_height': 128
            },
            {
                'cameraEyePosition': [-1.0, -0.25, 0.6],
                'cameraTargetPosition': [-0.6, -0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 128,
                'render_height': 128
            }
        ]
        env = pmg.make_env(
            # task args ['reach', 'push', 'slide', 'pick_and_place', 
            #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
            task=args.env_name,
            gripper='parallel_jaw',
            render=True,
            binary_reward=True,
            # image observation args
            image_observation=False,
            depth_image=False,
            goal_image=True,
            visualize_target=True,
            camera_setup=camera_setup,
            observation_cam_id=0,
            goal_cam_id=1)
        
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, reward==0))
