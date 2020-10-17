"""
Experiment demonstrating task composition
"""
import torch
from gym.wrappers import Monitor
import os
import deepdish as dd
import numpy as np

from dqn import ComposedDQN, FloatTensor
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame, MaxLength
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

if __name__ == '__main__':        
    def exp(name='or', save_trajectories=True, max_episodes = 4, max_trajectory = 20):    
        
        env = CollectEnv()
        dqn_purple = load('./models/purple/model.dqn', env)
        dqn_blue = load('./models/blue/model.dqn', env)
        dqn_crate = load('./models/crate/model.dqn', env)
        if torch.cuda.is_available():
            dqn_purple.cuda()
            dqn_blue.cuda()
            dqn_crate.cuda()
    
        dqn_max = ComposedDQN([dqn_purple,dqn_blue,dqn_crate], compose="or")
        dqn_not = ComposedDQN([dqn_blue], dqn_max=dqn_max, compose="not")
        dqn_or = ComposedDQN([dqn_blue,dqn_crate], compose="or")
        dqn_and = ComposedDQN([dqn_blue,dqn_crate], compose="and")
        dqn_not_and = ComposedDQN([dqn_and], dqn_max=dqn_max, compose="not")
        dqn_xor = ComposedDQN([dqn_or,dqn_not_and], compose="and")
        
        goals = []
        if os.path.exists('./goals.h5'):
            goals = dd.io.load('goals.h5')
            
        if name == 'blue':
            dqn = dqn_blue
            goal_condition=lambda x: x.colour == 'blue'
        elif name == 'purple':
            dqn = dqn_purple
            goal_condition=lambda x: x.colour == 'purple'
        elif name == 'square':
            dqn = dqn_crate
            goal_condition=lambda x: x.shape == 'square'
        if name == 'not':
            dqn = dqn_not
            goal_condition=lambda x: not x.colour == 'blue'
        elif name == 'or':
            dqn = dqn_or
            goal_condition=lambda x: x.colour == 'blue' or x.shape == 'square'
        elif name == 'and':
            dqn = dqn_and
            goal_condition=lambda x: x.colour == 'blue' and x.shape == 'square'
        elif name == 'xor':
            dqn = dqn_xor
            goal_condition=lambda x: (x.colour == 'blue' or x.shape == 'square') and not (x.colour == 'blue' and x.shape == 'square')
        else:
            print("Invalid name")
            return
        
        env = MaxLength(WarpFrame(CollectEnv(goal_condition=goal_condition)), max_trajectory)
        
        trajectories = []
        with torch.no_grad():
            episode = 0
            while episode < max_episodes:    
                obs = env.reset()                  
                trajectory = []          
                for _ in range(max_trajectory):
                    trajectory.append(Image.fromarray(np.uint8(env.render(mode='rgb_img'))))
                    
                    obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
                    values = []
                    for goal in goals:
                        goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
                        x = torch.cat((obs,goal),dim=3)
                        values.append(dqn(x).squeeze(0))
                    values = torch.stack(values,1).t()
                    action = values.data.max(0)[0].max(0)[1].item()
                    obs, reward, done, _ = env.step(action)
                    if done:
                        episode += 1
                        trajectories += trajectory[:-1]
                        break
        
        if save_trajectories:
            trajectories[0].save('./trajectories/'+name+'.gif',
               save_all=True, append_images=trajectories[1:], optimize=False, duration=250, loop=0)
    
    save_trajectories=False
    exp(name='blue', save_trajectories=save_trajectories)
    exp(name='purple', save_trajectories=save_trajectories)
    exp(name='square', save_trajectories=save_trajectories)
    exp(name='or', save_trajectories=save_trajectories)
    exp(name='and', save_trajectories=save_trajectories)
    exp(name='xor', save_trajectories=save_trajectories)
    exp(name='not', save_trajectories=True)
            
