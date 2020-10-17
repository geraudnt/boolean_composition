import torch
from gym.wrappers import Monitor
import os
import deepdish as dd
import numpy as np
import random

from dqn import ComposedDQN, FloatTensor
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame, MaxLength

from shortest import shortest



start_positions = {'crate_beige': (3, 4),
                   'player': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_blue': (1, 1),
                   'crate_purple': (8, 1),
                   'circle_blue': (1, 8)}
all_goals = ['BC','BS','bS','PS','bC','PC']
all_goals_P = [(1,8),(8,1),(1,1),(6,3),(1,7),(7,7)]

# Tasks = ["B.S", "B.-S", "S.-B", "-(B+S)", "B", "-B", "S", "-S", "B+S", "B+-S", "S+-B", "-(B.S)", "-BxorS", "BxorS"]
# Tasks_N = [1, 1, 2, 2, 2, 4, 3, 3, 4, 4, 5, 5, 3, 3]
Tasks = ["B", "S", "B+S", "B.S", "BxorS"]
Tasks_P = [[(1,8),(8,1)], [(8,1),(1,1),(6,3)], [(1,8),(8,1),(1,1),(6,3)], [(8,1)], [(1,8),(1,1),(6,3)]]

goals = []
if os.path.exists('./goals.h5'):
    goals = dd.io.load('goals.h5')

env = CollectEnv()
dqn_blue = load('./models/blue/model.dqn', env)
dqn_square = load('./models/crate/model.dqn', env)
if torch.cuda.is_available():
    dqn_blue.cuda()
    dqn_square.cuda()
dqn_not_blue = ComposedDQN([dqn_blue], compose="not")
dqn_not_square = ComposedDQN([dqn_square], compose="not")
dqn_or = ComposedDQN([dqn_blue,dqn_square], compose="or")
dqn_not_or = ComposedDQN([dqn_or], compose="not")
dqn_and = ComposedDQN([dqn_blue,dqn_square], compose="and")
dqn_not_and = ComposedDQN([dqn_and], compose="not")
dqn_xor = ComposedDQN([dqn_or,dqn_not_and], compose="and")
    
def evaluate(name='or', max_trajectory = 20):    
            
    if name == 'B.S':
        dqn = dqn_and
        goal_condition=lambda x: x.colour == 'blue' and x.shape == 'square'
    elif name == 'B.-S':
        dqn = ComposedDQN([dqn_blue,dqn_not_square], compose="and")
        goal_condition=lambda x: x.colour == 'blue' and not x.shape == 'square'
    elif name == 'S.-B':
        dqn = ComposedDQN([dqn_square,dqn_not_blue], compose="and")
        goal_condition=lambda x: x.shape == 'square' and not x.colour == 'blue'
    elif name == '-(B+S)':
        dqn = dqn_not_or
        goal_condition=lambda x: not (x.colour == 'blue' or x.shape == 'square')
    elif name == 'B':
        dqn = dqn_blue
        goal_condition=lambda x: x.colour == 'blue'
    elif name == '-B':
        dqn = dqn_not_blue
        goal_condition=lambda x: not x.colour == 'blue'
    elif name == 'S':
        dqn = dqn_square
        goal_condition=lambda x: x.shape == 'square'
    elif name == '-S':
        dqn = dqn_not_square
        goal_condition=lambda x: not x.shape == 'square'
    elif name == 'B+S':
        dqn = dqn_or
        goal_condition=lambda x: x.colour == 'blue' or x.shape == 'square'
    elif name == 'B+-S':
        dqn = ComposedDQN([dqn_blue,dqn_not_square], compose="or")
        goal_condition=lambda x: x.colour == 'blue' or not x.shape == 'square'
    elif name == 'S+-B':
        dqn = ComposedDQN([dqn_square,dqn_not_blue], compose="or")
        goal_condition=lambda x: x.shape == 'square' or not x.colour == 'blue'
    elif name == '-(B.S)':
        dqn = dqn_not_and
        goal_condition=lambda x: not (x.colour == 'blue' and x.shape == 'square')
    elif name == '-BxorS':
        dqn = ComposedDQN([dqn_xor], compose="not")
        goal_condition=lambda x: not((x.colour == 'blue' or x.shape == 'square') and not (x.colour == 'blue' and x.shape == 'square'))
    elif name == 'BxorS':
        dqn = dqn_xor
        goal_condition=lambda x: (x.colour == 'blue' or x.shape == 'square') and not (x.colour == 'blue' and x.shape == 'square')
    else:
        print("Invalid name")
        return
    
    env = MaxLength(WarpFrame(CollectEnv(start_positions=start_positions,goal_condition=goal_condition)), max_trajectory)
    
    G = 0
    with torch.no_grad():
        obs = env.reset()                
        for _ in range(max_trajectory):
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
            values = []
            for goal in goals:
                goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
                x = torch.cat((obs,goal),dim=3)
                values.append(dqn(x).squeeze(0))
            values = torch.stack(values,1).t()
            action = values.data.max(0)[0].max(0)[1].item()
            obs, reward, done, _ = env.step(action)        
            G += reward

            if done:
                break
    return G

def optimal(goals,free_spaces,dist):
    s = random.choice(free_spaces)
    ds = []
    for g in goals:
        g = (g[1],g[0])
        d = dist[(s,g)]+1
        ds.append(d)
    G = 2 + -0.1 * min(ds)            
    return G

free_spaces,dist = shortest()

num_runs = 1000
data0 = np.zeros((num_runs,len(Tasks))) 
data1 = np.zeros((num_runs,len(Tasks))) 
for i in range(num_runs):
    print('run: ',i)
    for j in range(len(Tasks)):
        data0[i,j] = optimal(Tasks_P[j],free_spaces,dist)
        data1[i,j] = evaluate(Tasks[j])
    dd.io.save('data/exp_returns_0.h5', data0)
    dd.io.save('data/exp_returns_1.h5', data1)

