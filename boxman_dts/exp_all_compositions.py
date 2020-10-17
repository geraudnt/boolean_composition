"""
Experiment demonstrating task composition
"""
import torch
from gym.wrappers import Monitor
import os
import deepdish as dd
import numpy as np
import itertools

from dqn import *
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame, MaxLength
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


if __name__ == '__main__':
    env = CollectEnv()
    
    all_goals = np.array([('beige','circle'),('beige','square'),('blue','circle'),
                         ('blue','square'),('purple','circle'),('purple','square')])
    base_tasks = np.array([[0,0,0,0,1,1],[0,0,1,1,0,0],[0,1,0,1,0,1]])

    dqn_purple = load('./models/purple/model.dqn', env)
    dqn_blue = load('./models/blue/model.dqn', env)
    dqn_crate = load('./models/crate/model.dqn', env)
    if torch.cuda.is_available():
        dqn_purple.cuda()
        dqn_blue.cuda()
        dqn_crate.cuda()
    models = {'P': dqn_purple, 'B': dqn_blue, 'S': dqn_crate}   
    mgoals = []
    if os.path.exists('./goals.h5'):
        mgoals = dd.io.load('goals.h5')

    
    def experiment(task=None, save_trajectories=True, max_episodes = 4, max_trajectory = 20):
        
        exp = task_exp(base_tasks,task,len(all_goals), list(models.keys()))
        dqn = exp_EVF(exp, models)

        goals = all_goals[task==1]
        goal_condition = lambda x: ((x.colour,x.shape) in  goals)  
        env = WarpFrame(CollectEnv(goal_condition=goal_condition))
        
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
                    for goal in mgoals:
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
            exp = str(exp.simplify()).replace('~','NOT ')
            exp = exp.replace('|',' OR ')
            exp = exp.replace('&',' AND ')
            trajectories[0].save('./trajectories/all_/'+exp+'.gif',
               save_all=True, append_images=trajectories[1:], optimize=False, duration=250, loop=0)
    
    for g in range(1,2**len(all_goals)):
        print(g)
        task = [int(i) for i in bin(g)[2:]]
        task = np.array([0]*(len(all_goals)-len(task))+task)
        experiment(task=task)
            