import numpy as np
from matplotlib import pyplot as plt
from GridWorld import GridWorld
from library import *



env = GridWorld()
maxiter=500
T_states = [(3,3),(9,3),(3,9),(9,9)]
T_states = [[pos,pos] for pos in T_states]

### Learning universal bounds
env = GridWorld(goals=T_states, T_states=T_states)
EQ_max,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)

env = GridWorld(goals=T_states, goal_reward = -0.1, T_states=T_states)
EQ_min,stats2 = Goal_Oriented_Q_learning(env, maxiter=maxiter)

### Learning base tasks
goals=[(3,3),(3,9)]
goals = [[pos,pos] for pos in goals]
env = GridWorld(goals=goals, T_states=T_states)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
env.render(R=env_R(env, A.keys()))

goals=[(3,3),(9,3)]
goals = [[pos,pos] for pos in goals]
env = GridWorld(goals=goals, T_states=T_states)
B,stats2 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
env.render(R=env_R(env, B.keys()))


### Zero-shot composition
NEG = lambda x: NOT(x,EQ_max=EQ_max,EQ_min=EQ_min)
XOR = lambda EQ1, EQ2: OR(AND(EQ1,NEG(EQ2)),AND(EQ2,NEG(EQ1)))
print("______________________________________________________________________")
env.render( P=EQ_P(A), V = EQ_V(A))
env.render( P=EQ_P(B), V = EQ_V(B))
print("______________________________________________________________________")
env.render( P=EQ_P(OR(A,B)), V = EQ_V(OR(A,B)))
env.render( P=EQ_P(AND(A,B)), V = EQ_V(AND(A,B)))
env.render( P=EQ_P(XOR(A,B)), V = EQ_V(XOR(A,B)))
env.render( P=EQ_P(NEG(OR(A,B))), V = EQ_V(NEG(OR(A,B))))
print("______________________________________________________________________")


