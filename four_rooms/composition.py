from GridWorld import GridWorld
from library import *


env = GridWorld()
T_states = [(3,3),(9,3),(3,9),(9,9)]
T_states = [[pos,pos] for pos in T_states]

maxiter=500
### Learning base tasks
goals=[(3,3),(3,9)]
goals = [[pos,pos] for pos in goals]
env = GridWorld(goals=goals, T_states=T_states)
env.render(R=env.env_R())
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)

goals=[(3,3),(9,3)]
goals = [[pos,pos] for pos in goals]
env = GridWorld(goals=goals, T_states=T_states)
env.render(R=env.env_R())
B,stats2 = Goal_Oriented_Q_learning(env, maxiter=maxiter)


### Zero-shot composition
XOR = lambda EQ1, EQ2: OR(AND(EQ1,NOT(EQ2)),AND(EQ2,NOT(EQ1)))

env.render( P=EQ_P(A), V = EQ_V(A))
env.render( P=EQ_P(B), V = EQ_V(B))

env.render( P=EQ_P(OR(A,B)), V = EQ_V(OR(A,B)))
env.render( P=EQ_P(AND(A,B)), V = EQ_V(AND(A,B)))
env.render( P=EQ_P(XOR(A,B)), V = EQ_V(XOR(A,B)))
env.render( P=EQ_P(NOT(OR(A,B))), V = EQ_V(NOT(OR(A,B))))



