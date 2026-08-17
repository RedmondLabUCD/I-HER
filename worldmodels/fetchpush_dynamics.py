"""
RNN simulated environment.
"""
from os.path import join, exists
import torch
import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
from dynamics.simplefeedforward.dynamics_model import DynamicsModel

class FetchPushDynamics(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.
    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.
    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    def __init__(self, dy_model, env, obs_rms, clip_range, render=False):

        self.dy_model = dy_model
        self.realenv = env
        self.obs_rms = obs_rms
        self.clip_range = clip_range

        # spaces
        # TODO: verify these are valid
        self.action_space = env.action_space # spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = env.observation_space # spaces.Box(low=0, high=255, shape=(RED_SIZE, RED_SIZE, 3), dtype=np.uint8)
        self.compute_reward = self.realenv.compute_reward
        self._max_episode_steps = self.realenv._max_episode_steps

        # obs
        self._obs = self.realenv.reset()
        self._stepcount = 0
        self._done = 0

        if render:
            # rendering
            self.fig, self.ax = plt.subplots()
            self.ax = plt.axes(projection='3d')

    def reset(self):
        """ Resetting """
        self._obs = self.realenv.reset()
        return self._obs

    def step(self, action):
        """ One step forward """
        with torch.no_grad():
            action = torch.Tensor(action)#.unsqueeze(0)
            # Normalise obs before and de-normalise after inputting to dynamics_model
            obs = np.clip((self._obs['observation'] - self.obs_rms.cur_mean) / self.obs_rms.cur_std, -self.clip_range, self.clip_range)
            prediction = self.dy_model(torch.tensor(obs), action)
            prediction = (prediction.detach().numpy().squeeze() * self.obs_rms.cur_std) + self.obs_rms.cur_mean

            self._obs['observation'] = prediction # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self._obs['achieved_goal'] = self._obs['observation'][3:6]
            
            r = self.compute_reward(self._obs['achieved_goal'], self._obs['desired_goal'], None)
            
            self._stepcount += 1
            if self._stepcount >= self._max_episode_steps:
                self._done = 1

            return self._obs, r.item(), self._done > 0, None

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        self.ax.cla()
        # Plot table
        front_x, back_x = 1.6, 1
        left_y, right_y = 0.4, 1.1
        top_z, bottom_z = 0.4, 0
        # Top
        tablex = np.array([front_x,back_x,back_x,front_x,front_x])
        tabley = np.array([left_y,left_y,right_y,right_y,left_y])
        tablez = np.array([top_z,top_z,top_z,top_z,top_z])
        self.ax.plot(tablex, tabley, tablez, 'k')
        # Bottom
        tablex = np.array([back_x,front_x,front_x])
        tabley = np.array([left_y,left_y,right_y])
        tablez = np.array([bottom_z,bottom_z,bottom_z])
        self.ax.plot(tablex, tabley, tablez, 'k')
        # Sides
        tablex = np.array([back_x,back_x,front_x,front_x,front_x,front_x])
        tabley = np.array([left_y,left_y,left_y,left_y,right_y,right_y])
        tablez = np.array([bottom_z,top_z,top_z,bottom_z,bottom_z,top_z])
        self.ax.plot(tablex, tabley, tablez, 'k')
        
        # Plot gripper
        gripper = self._obs['observation'][0:3]
        handx = np.array([gripper[0], gripper[0]-0.2])
        handy = np.array([gripper[1], gripper[1]])
        handz = np.array([gripper[2],gripper[2]+0.2])
        self.ax.plot(handx, handy, handz, 'k')
        self.ax.plot(gripper[0], gripper[1], gripper[2], 'ko')
        
        # Plot goal + cube
        self.ax.plot(self._obs['achieved_goal'][0], self._obs['achieved_goal'][1], self._obs['achieved_goal'][2], 'bo')
        self.ax.plot(self._obs['desired_goal'][0], self._obs['desired_goal'][1], self._obs['desired_goal'][2], 'ro')
        
        self.ax.set_xlim([0.5, 2])
        self.ax.set_ylim([0, 1.5])
        self.ax.set_zlim([0, 1])
        # self.ax.set_title('Imagined')
        self.ax.axis('off')
        # self.ax.view_init(30, -30)
        self.ax.view_init(20, -40)
        plt.pause(.01)

    def close(self):
        self.realenv.close()
