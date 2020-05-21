import torch
from torch.autograd import Variable

from gym_repoman.envs.collect_env import CollectEnv
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import deepdish as dd
import os

from dqn import DQN, ComposedDQN, FloatTensor
from wrappers import WarpFrame

def load(path, env):
    dqn = DQN(env.action_space.n)
    dqn.load_state_dict(torch.load(path))
    return dqn

def remove(positions, pos):
    found = None
    for key, val in positions.items():
        if key != 'player' and pos == val:
            found = key
            break

    if found is not None:
        positions[found] = None
    return positions


WEIGHTS = [0, 1]

start_positions = {'crate_beige': (3, 4),
                   'player': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_blue': (1, 1),
                   'crate_purple': (8, 1),
                   'circle_blue': (1, 8)}

goal_condition = lambda x: False
env = (WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition)))

goals = []
if os.path.exists('./goals.h5'):
    goals = dd.io.load('goals.h5')

dqn_blue = load('./models/blue/model.dqn', env)
dqn_purple = load('./models/purple/model.dqn', env)
dqn_crate = load('./models/crate/model.dqn', env)
if torch.cuda.is_available():
    dqn_blue.cuda()
    dqn_purple.cuda()
    dqn_crate.cuda()

dqn_or = ComposedDQN([dqn_blue,dqn_purple], compose="or")
dqn_and = ComposedDQN([dqn_blue,dqn_crate], compose="and")
dqn_not_blue = ComposedDQN([dqn_blue], compose="not")
dqn_not_crate = ComposedDQN([dqn_crate], compose="not")
dqn_blue_not_crate = ComposedDQN([dqn_blue,dqn_not_crate], compose="and")
dqn_crate_not_blue = ComposedDQN([dqn_not_blue,dqn_crate], compose="and")
dqn_xor = ComposedDQN([dqn_blue_not_crate,dqn_crate_not_blue], compose="or")

dqn = dqn_and
name = 'and'

if name == 'blue':
    goal_condition=lambda x: x.colour == 'blue'
elif name == 'purple':
    goal_condition=lambda x: x.colour == 'purple'
elif name == 'crate':
    goal_condition=lambda x: x.shape == 'square'
elif name == 'or':
    goal_condition=lambda x: x.colour == 'blue' or x.colour == 'purple'
elif name == 'and':
    goal_condition=lambda x: x.colour == 'blue' and x.shape == 'square'
elif name == 'xor':
    goal_condition=lambda x: (x.colour == 'blue' or x.shape == 'square') and not (x.colour == 'blue' and x.shape == 'square')

values = np.zeros_like(env.board, dtype=float)
for pos in env.free_spaces:
    positions = copy.deepcopy(start_positions)
    positions['crate_beige'] = pos
    env = (WarpFrame(CollectEnv(start_positions=positions, goal_condition=goal_condition)))
    
    obs = env.reset()
    obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
    with torch.no_grad():
        nv = []
        for goal in goals:
            #plt.imshow(goal)
            #plt.show()
            goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
            x = torch.cat((obs,goal),dim=3)
            nv.append(dqn(x))
        nv = torch.stack(nv,1).t()
        #print(nv)
        v = nv.data.max(0)[0].max(0)[0]#np.random.randint(0,4)#
        #v = dqn.get_value(obs, tau=1)
    values[pos] = v

idxs = values == 0 # values<=0 #
update = list(map(tuple, np.argwhere(idxs)))

for _ in range(1000):
    for i in range(len(values)):
        for j in range(len(values[i])):
            if (i, j) in update:

                s = values[i, j]
                count = 1
                for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    m = i + offset[0]
                    n = j + offset[1]
                    if m < 0 or m >= len(values) or n < 0 or n >= len(values[i]):
                        continue
                    count += 1
                    s += values[m, n]
                values[i, j] = s / count

print(values)

np.savetxt('data/values_'+name+'.txt', values, fmt='%.4f')

for ii in range(6):
    values = np.zeros_like(env.board, dtype=float)
    for pos in env.free_spaces:
        positions = copy.deepcopy(start_positions)
        positions['crate_beige'] = pos
        env = (WarpFrame(CollectEnv(start_positions=positions, goal_condition=goal_condition)))
        
        obs = env.reset()
        obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
        with torch.no_grad():
            nv = []
            for goal in goals:
                #plt.imshow(goal)
                #plt.show()
                goal = torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
                x = torch.cat((obs,goal),dim=3)
                nv.append(dqn(x))
            nv = torch.stack(nv,1).t()
            #print(nv)
            v = nv.data[ii].max(0)[0]#np.random.randint(0,4)#
            #v = dqn.get_value(obs, tau=1)
        values[pos] = v

    idxs = values == 0 # values<=0 #
    update = list(map(tuple, np.argwhere(idxs)))

    for _ in range(1000):
        for i in range(len(values)):
            for j in range(len(values[i])):
                if (i, j) in update:

                    s = values[i, j]
                    count = 1
                    for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        m = i + offset[0]
                        n = j + offset[1]
                        if m < 0 or m >= len(values) or n < 0 or n >= len(values[i]):
                            continue
                        count += 1
                        s += values[m, n]
                    values[i, j] = s / count

    print(values)

    np.savetxt('data/values_'+name+'_'+str(ii)+'.txt', values, fmt='%.4f')
