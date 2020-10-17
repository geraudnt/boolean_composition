import numpy as np
from matplotlib import pyplot as plt
import deepdish as dd
from GridWorld import GridWorld
from library import *


T_states=[(3,3),(3,9),(9,3),(9,9)]
T_states = [[pos,pos] for pos in T_states]

Bases = [[(3,3),(3,9)], [(3,3),(9,3)]]
Tasks = [[],[(3,3),(3,9),(9,3),(9,9)],[(3,3)],[(3,9)],[(9,3)],[(9,9)],[(3,3),(3,9)],[(9,3),(9,9)],[(3,3),(9,3)],[(3,9),(9,9)],[(3,3),(3,9),(9,3)],[(3,3),(3,9),(9,9)],[(3,3),(9,3),(9,9)],[(3,9),(9,3),(9,9)],[(3,3),(9,9)],[(3,9),(9,3)]]

#Sparse rewards, Same terminal states
types = [(True,True),(True,False),(False,True),(False,False)] 
t=3

#Slip probabilities
slip_probs = [0,0.1,0.2,0.3] 

epsilon=0.5
alpha=0.1
maxiter=50000
num_runs = 100000


def evaluate(goals,EQ,slip_prob=0):
    env = GridWorld(goals=goals, T_states=T_states,slip_prob=slip_prob)
    policy =  EQ_P(EQ)
    state = env.reset()
    done = False
    t=0
    G = 0
    while not done and t<100:
        action = policy[state]        
        state_, reward, done, _ = env.step(action)       
        state = state_            
        G += reward
        t += 1
    return G

for sp in range(len(slip_probs)):
    slip_prob=slip_probs[sp]
    print("slip_prob: ",slip_prob)
    
    # Learning universal bounds (min and max tasks)
    env = GridWorld(goals=T_states, slip_prob=slip_prob, dense_rewards = not types[t][0])
    EQ_max,_ = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
    
    env = GridWorld(goals=T_states, goal_reward=-0.1, slip_prob=slip_prob, dense_rewards = not types[t][0])
    EQ_min,_ = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
    
    # Learning base tasks and doing composed tasks
    goals=Bases[0]
    goals = [[pos,pos] for pos in goals]
    env = GridWorld(goals=goals, slip_prob=slip_prob, dense_rewards = not types[t][0], T_states=T_states if types[t][1] else goals)
    A,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, T_states=None if types[t][1] else T_states)
    
    goals=Bases[1]
    goals = [[pos,pos] for pos in goals]
    env = GridWorld(goals=goals, slip_prob=slip_prob, dense_rewards = not types[t][0], T_states=T_states if types[t][1] else goals)
    B,stats2 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, T_states=None if types[t][1] else T_states)
    
    NEG = lambda x: NOT(x,EQ_max=EQ_max,EQ_min=EQ_min)
    XOR = lambda EQ1, EQ2: OR(AND(EQ1,NEG(EQ2)),AND(EQ2,NEG(EQ1)))
    composed = [EQ_min,EQ_max,AND(A,B),AND(A,NEG(B)),AND(B,NEG(A)),NEG(OR(A,B)),A,NEG(A),B,NEG(B),OR(A,B),OR(A,NEG(B)),OR(B,NEG(A)),NEG(AND(A,B)),NEG(XOR(A,B)),XOR(A,B)]
    #for EQ in composed:
    #    env.render( P=EQ_P(EQ), V = EQ_V(EQ)) 
    
    data = np.zeros((num_runs,len(Tasks))) 
    for i in range(num_runs):
        for j in range(len(Tasks)):
            goals = [[pos,pos] for pos in Tasks[j]]
            data[i,j] = evaluate(goals,composed[j],slip_prob=slip_prob)    
    data1 = dd.io.save('exps_data/exp4_returns_'+str(sp)+'.h5', data)
    

