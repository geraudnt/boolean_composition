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

if __name__ == '__main__':

    max_episodes = 50000
    max_trajectory = 20

    start_positions = {'crate_beige': (7, 1),
                       'player': (6, 3),
                       'circle_purple': (7, 7),
                       'circle_beige': (1, 7),
                       'crate_blue': (1, 1),
                       'crate_purple': (8, 1),
                       'circle_blue': (1, 8)}

    B = lambda x: x.colour=="blue"
    S = lambda x: x.shape=="square"
    env = MaxLength(WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=lambda x: (B(x) or S(x)) and not (B(x) and S(x)))),
                     max_trajectory)
    
    goals = []
    if os.path.exists('./goals.h5'):
        goals = dd.io.load('goals.h5')
    
    dqn_blue = load('./models/blue/model.dqn', env)
    dqn_crate = load('./models/crate/model.dqn', env)
    if torch.cuda.is_available():
        dqn_blue.cuda()
        dqn_crate.cuda()

    dqn_or = ComposedDQN([dqn_blue,dqn_crate], compose="or")
    dqn_and = ComposedDQN([dqn_blue,dqn_crate], compose="and")
    dqn_not_and = ComposedDQN([dqn_and], compose="not")
    dqn_xor = ComposedDQN([dqn_or,dqn_not_and], compose="and")
    
    dqn = dqn_and
    with torch.no_grad():
        for episode in range(max_episodes):
            if episode % 1000 == 0:
                print(episode)  
            obs = env.reset()
            for _ in range(max_trajectory):
                env.render()
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
                    break
