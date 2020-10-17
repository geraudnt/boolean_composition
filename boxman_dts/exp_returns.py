import torch
from gym.wrappers import Monitor
import os
import deepdish as dd
import numpy as np

from dqn import ComposedDQN, FloatTensor
from trainer import load
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame, MaxLength

from shortest import shortest


# Tasks = ["B.S", "B.-S", "S.-B", "-(B+S)", "B", "-B", "S", "-S", "B+S", "B+-S", "S+-B", "-(B.S)", "-BxorS", "BxorS"]
# Tasks_N = [1, 1, 2, 2, 2, 4, 3, 3, 4, 4, 5, 5, 3, 3]
Tasks = ["P", "B", "S", "P+B", "B.S", "BxorS"]
Tasks_N = [2, 2, 3, 4, 1, 3]

goals = []
if os.path.exists('./goals.h5'):
    goals = dd.io.load('goals.h5')

env = CollectEnv()
dqn_purple = load('./models/purple/model.dqn', env)
dqn_blue = load('./models/blue/model.dqn', env)
dqn_square = load('./models/crate/model.dqn', env)
if torch.cuda.is_available():
    dqn_purple.cuda()
    dqn_blue.cuda()
    dqn_square.cuda()

max_evf = ComposedDQN([dqn_purple,dqn_blue,dqn_square])
dqn_not_blue = ComposedDQN([dqn_blue], dqn_max = max_evf, compose="not")
dqn_not_square = ComposedDQN([dqn_square], dqn_max = max_evf, compose="not")
dqn_or_purple = ComposedDQN([dqn_purple,dqn_blue], compose="or")
dqn_or = ComposedDQN([dqn_blue,dqn_square], compose="or")
dqn_not_or = ComposedDQN([dqn_or], dqn_max = max_evf, compose="not")
dqn_and = ComposedDQN([dqn_blue,dqn_square], compose="and")
dqn_not_and = ComposedDQN([dqn_and], dqn_max = max_evf, compose="not")
dqn_xor = ComposedDQN([dqn_or,dqn_not_and], compose="and")
    
def evaluate(name='or', save_trajectories=True, max_trajectory = 20):    
            
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
    elif name == 'P':
        dqn = dqn_purple
        goal_condition=lambda x: x.colour == 'purple'
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
    elif name == 'P+B':
        dqn = dqn_or_purple
        goal_condition=lambda x: x.colour == 'purple' or x.colour == 'blue'
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
        dqn = ComposedDQN([dqn_xor], dqn_max = max_evf, compose="not")
        goal_condition=lambda x: not((x.colour == 'blue' or x.shape == 'square') and not (x.colour == 'blue' and x.shape == 'square'))
    elif name == 'BxorS':
        dqn = dqn_xor
        goal_condition=lambda x: (x.colour == 'blue' or x.shape == 'square') and not (x.colour == 'blue' and x.shape == 'square')
    else:
        print("Invalid name")
        return
    
    env = MaxLength(WarpFrame(CollectEnv(goal_condition=goal_condition)), max_trajectory)
    
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

# Obtained from shortest.py. -0.1 on means to account for pickup action
G=[(1.5036297640653358, 0.2943079914627588),
(1.633708192896033, 0.2444848035735638),
(1.7078841303886678, 0.2095051257786029),
(1.75778133945103, 0.1837586078497989),
(1.7940084062999377, 0.16393116684268055)
]

num_runs = 1000
data0 = np.zeros((num_runs,len(Tasks))) 
data1 = np.zeros((num_runs,len(Tasks))) 
for i in range(num_runs):
    print('run: ',i)
    for j in range(len(Tasks)):
        mean, std = G[Tasks_N[j]-1] #shortest(Tasks_N[j]) 
        data0[i,j] = min(1.9,np.random.normal(loc=mean, scale=std))
        # data1[i,j] = evaluate(Tasks[j])
    dd.io.save('data/exp_returns_0.h5', data0)
    # dd.io.save('data/exp_returns_1.h5', data1)

# for t in range(len(types)):
#     print("type: ",t)
    
#     # Learning universal bounds (min and max tasks)
#     env = GridWorld(goals=T_states, dense_rewards = not types[t][0])
#     EQ_max,_ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
    
#     env = GridWorld(goals=T_states, goal_reward=-0.1, dense_rewards = not types[t][0])
#     EQ_min,_ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
    
#     # Learning base tasks and doing composed tasks
#     goals=Bases[0]
#     goals = [[pos,pos] for pos in goals]
#     env = GridWorld(goals=goals, dense_rewards = not types[t][0], T_states=T_states if types[t][1] else goals)
#     A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, T_states=None if types[t][1] else T_states)
    
#     goals=Bases[1]
#     goals = [[pos,pos] for pos in goals]
#     env = GridWorld(goals=goals, dense_rewards = not types[t][0], T_states=T_states if types[t][1] else goals)
#     B,stats2 = Goal_Oriented_Q_learning(env, maxiter=maxiter, T_states=None if types[t][1] else T_states)
    
#     NEG = lambda x: NOT(x,EQ_max=EQ_max,EQ_min=EQ_min)
#     XOR = lambda EQ1, EQ2: OR(AND(EQ1,NEG(EQ2)),AND(EQ2,NEG(EQ1)))
#     composed = [EQ_min,EQ_max,AND(A,B),AND(A,NEG(B)),AND(B,NEG(A)),NEG(OR(A,B)),A,NEG(A),B,NEG(B),OR(A,B),OR(A,NEG(B)),OR(B,NEG(A)),NEG(AND(A,B)),NEG(XOR(A,B)),XOR(A,B)]
    
#     #for EQ in composed:
#     #    env.render( P=EQ_P(EQ), V = EQ_V(EQ))   
    
#     num_runs = num_runs
#     data = np.zeros((num_runs,len(Tasks))) 
#     for i in range(num_runs):
#         for j in range(len(Tasks)):
#             goals = [[pos,pos] for pos in Tasks[j]]
#             data[i,j] = evaluate(goals,composed[j])    
#     data1 = dd.io.save('exps_data/exp3_returns_'+str(t)+'.h5', data)
    

