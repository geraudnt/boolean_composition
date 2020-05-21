import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from library import *



env = GridWorld()
T_states=[(3,3),(3,9),(9,3),(9,9),
          (1,1),(1,2),(1,3),(1,4),(1,5),(1,7),(1,8),(1,9),(1,10),(1,11),
          (11,1),(11,2),(11,3),(11,4),(11,5),(11,7),(11,8),(11,9),(11,10),
          (2,1),(3,1),(4,1),(5,1),(7,1),(8,1),(9,1),(10,1),
          (2,11),(3,11),(4,11),(5,11),(6,11),(8,11),(9,11),(10,11),(11,11)]

###################################### Qs
BTasksQ = [[t] for t in T_states]
###################################### EQs
Bases = []
n=int(np.ceil(np.log2(len(T_states))))
m=(2**n)/2
for i in range(n):
    Bases.append([])
    b=False
    for j in range(0,2**n):
        if j>=len(T_states):
            break
        if b:
            Bases[i].append(1) #1=True=rmax
        else:
            Bases[i].append(0) #0=False=rmin
        if (j+1)%m==0:
            if b:
                b=False
            else:
                b=True
    m=m/2

BTasksEQ=[]
for i in range(len(Bases)):
    BTasksEQ.append([])
    for j in range(len(Bases[i])):
        if Bases[i][j]==1:
            BTasksEQ[i].append(T_states[j])
######################################

T_states = [[pos,pos] for pos in T_states]


Qs = dd.io.load('exps_data/40Goals_Optimal_Qs.h5')
Qs = [{s:v for (s,v) in Q} for Q in Qs]
EQs = dd.io.load('exps_data/40Goals_Optimal_EQs.h5')
EQs = [{s:{s__:v__ for (s__,v__) in v} for (s,v) in EQ} for EQ in EQs]

num_runs = 1
dataQ = np.zeros((num_runs,len(BTasksQ))) 
dataEQ = np.zeros((num_runs,len(BTasksEQ))) 
idxs=np.arange(len(BTasksQ))
for i in range(num_runs):
    print("run: ",i)
    np.random.shuffle(idxs)
    for j in idxs:
        print("goals: ",j)
        goals = [[pos,pos] for pos in BTasksQ[j]]
        env = GridWorld(goals=goals, T_states=T_states, rmax=1, rmin=-0.01, goal_reward=1, step_reward=-0.01)
        _,stats = Q_learning(env, Q_optimal=Qs[j])
        dataQ[i,j] = stats["T"]
idxs=np.arange(len(BTasksEQ))
for i in range(num_runs):
    print("run: ",i)
    np.random.shuffle(idxs)
    for j in idxs:
        print("goals: ",j)
        goals = [[pos,pos] for pos in BTasksEQ[j]]
        env = GridWorld(goals=goals, T_states=T_states, rmax=1, rmin=-0.01, goal_reward=1, step_reward=-0.01)
        _,stats = Goal_Oriented_Q_learning(env, Q_optimal=EQs[j])
        dataEQ[i,j] = stats["T"]

data1 = dd.io.save('exps_data/exp2_samples_Qs.h5', dataQ )
data2 = dd.io.save('exps_data/exp2_samples_EQs.h5', dataEQ)



