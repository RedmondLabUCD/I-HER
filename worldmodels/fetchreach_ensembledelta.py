"""
Ensemble of dynamics models simulated environment.
"""
from os.path import join, exists
import torch
import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
from dynamics.simplefeedforward.dynamics_model import DynamicsModel

class FetchReachEnsembleDelta(gym.Env):
    def __init__(self, ensemble, env, obs_norm, delta_norm, clip_range, close_gripper=False):

        self.ensemble = ensemble
        self.realenv = env
        self.obs_norm = obs_norm
        self.delta_norm = delta_norm
        self.clip_range = clip_range
        self.close_gripper = close_gripper

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

        # rendering
        self.fig, self.ax = plt.subplots()
        self.ax = plt.axes(projection='3d')

    def reset(self):
        """ Resetting """
        self._obs = self.realenv.reset()
        return self._obs

    def step(self, action):
        """ One step forward """
        if self.close_gripper:
            action = action.copy()
            action[-1] = -1
        with torch.no_grad():
            # Normalise obs
            obs = np.clip((self._obs['observation'] - self.obs_norm.mean) / self.obs_norm.std, -self.clip_range, self.clip_range)
            i = np.random.randint(len(self.ensemble))
            delta = self.ensemble[i](torch.tensor(obs).float(), torch.tensor(action).float())
            # Denormalise delta prediction
            delta = (delta.detach().numpy().squeeze() * self.delta_norm.std) + self.delta_norm.mean
            # Add delta to curent obs to get next obs
            prediction = self._obs['observation'] + delta

            self._obs['observation'] = prediction
            self._obs['achieved_goal'] = self._obs['observation'][0:3]
            
            r = self.compute_reward(self._obs['achieved_goal'], self._obs['desired_goal'], None)
            
            self._stepcount += 1
            if self._stepcount >= self._max_episode_steps:
                self._done = 1

            return self._obs, r.item(), self._done > 0, None

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        self.ax.cla()
        # Plot table-top
        tablex = np.array([1.6,1,1,1.6,1.6])
        tabley = np.array([0.4,0.4,1.1,1.1,0.4])
        tablez = np.array([0.4,0.4,0.4,0.4,0.4])
        self.ax.plot(tablex, tabley, tablez, 'k')
        
        # Plot gripper
        gripper = self._obs['observation'][0:3]
        handx = np.array([gripper[0], gripper[0]-0.2])
        handy = np.array([gripper[1], gripper[1]])
        handz = np.array([gripper[2],gripper[2]+0.2])
        self.ax.plot(handx, handy, handz, 'k')
        self.ax.plot(gripper[0], gripper[1], gripper[2], 'ko')
        
        # Plot goal + cube
        self.ax.plot(self._obs['desired_goal'][0], self._obs['desired_goal'][1], self._obs['desired_goal'][2], 'ro')
        
        self.ax.set_xlim([0.5, 2])
        self.ax.set_ylim([0, 1.5])
        self.ax.set_zlim([0, 1])
        self.ax.set_title('Imagined')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.view_init(30, -30)
        plt.pause(.01)

    def close(self):
        self.realenv.close()
