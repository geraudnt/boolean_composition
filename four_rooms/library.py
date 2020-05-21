import numpy as np

from collections import defaultdict
from copy import deepcopy
from ast import literal_eval


#########################################################################################
def Q_equal(Q1,Q2,epsilon=1e-5):    
    for state in Q1:
        for action in range(len(Q1[state])): 
            if abs(Q1[state][action]-Q2[state][action])>epsilon:
                return False
    return True

def EQ_equal(EQ1,EQ2,epsilon=1e-5):    
    for state in EQ1:
        for goal in EQ1[state]:
            for action in range(len(EQ1[state][goal])): 
                if abs(EQ1[state][goal][action]-EQ2[state][goal][action])>epsilon:
                    return False
    return True
#########################################################################################
def epsilon_greedy_policy_improvement(env, Q, epsilon=1):
    """
    Implements policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy
    epsilon -- probability

    Returns:
    policy_improved -- Improved policy
    """
        
    def policy_improved(state, epsilon = epsilon):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        best_action = np.random.choice(np.flatnonzero(Q[state] == Q[state].max())) #np.argmax(Q[state]) #
        probs[best_action] += 1.0 - epsilon
        return probs

    return policy_improved

def epsilon_greedy_generalised_policy_improvement(env, Q, epsilon = 1):
    """
    Implements generalised policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy

    Returns:
    policy_improved -- Improved policy
    """
    
    def policy_improved(state, epsilon = epsilon):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        values = [Q[state][goal] for goal in Q[state].keys()]
        best_action = np.random.randint(env.action_space.n) if len(values)==0 else np.argmax(np.max(values,axis=0))
        probs[best_action] += 1.0 - epsilon
        return probs

    return policy_improved

#########################################################################################
def Q_learning(env, Q_optimal=None, gamma=1, epsilon=1, alpha=1, maxiter=100, maxstep=100):
    """
    Implements Q_learning

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q, epsilon = epsilon)
    
    stop_cond = lambda k: k < maxiter
    if Q_optimal:
        stop_cond = lambda _: not Q_equal(Q_optimal,Q)
        
    stats = {"R":[], "T":0}
    k=0
    T=0
    state = env.reset()
    stats["R"].append(0)
    while stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)
        
        stats["R"][k] += reward
        
        G = 0 if done else np.max(Q[state_])
        TD_target = reward + gamma*G
        TD_error = TD_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha*TD_error
        
        state = state_
        T+=1
        if done:            
            state = env.reset()
            stats["R"].append(0)
            k+=1
    stats["T"] = T
    
    return Q, stats

def Goal_Oriented_Q_learning(env, T_states=None, Q_optimal=None, gamma=1, epsilon=1, alpha=1, maxiter=100, maxstep=100):
    """
    Implements Goal Oriented Q_learning

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    N = min(env.rmin, (env.rmin-env.rmax)*env.diameter)
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q, epsilon = epsilon)
    
    sMem={} # Goals memory
    if T_states:
        for state in T_states:
            sMem[str(state)]=0
    
    stop_cond = lambda k: k < maxiter
    if Q_optimal:
        stop_cond = lambda _: not EQ_equal(Q_optimal,Q)
                
    stats = {"R":[], "T":0}
    k=0
    T=0
    state = env.reset()
    stats["R"].append(0)
    while stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)
        
        stats["R"][k] += reward
        
        if done:
            sMem[state] = 0
        
        for goal in sMem.keys():
            if state != goal and done:  
                reward_ = N
            else:
                reward_ = reward
            
            G = 0 if done else np.max(Q[state_][goal])
            TD_target = reward_ + gamma*G
            TD_error = TD_target - Q[state][goal][action]
            Q[state][goal][action] = Q[state][goal][action] + alpha*TD_error
        
        state = state_
        T+=1
        if done:
            state = env.reset()
            stats["R"].append(0)
            k+=1
    stats["T"] = T

    return Q, stats

#########################################################################################
def EQ_NP(EQ):
    P = defaultdict(lambda: defaultdict(lambda: 0))
    for state in EQ:
        for goal in EQ[state]:
                P[state][goal] = np.argmax(EQ[state][goal])
    return P
def EQ_P(EQ, goal=None):
    P = defaultdict(lambda: 0)
    for state in EQ:
        if goal:
            P[state] = np.argmax(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            P[state] = np.argmax(np.max(Vs,axis=0))
    return P
def Q_P(Q):
    P = defaultdict(lambda: 0)
    for state in Q:
        P[state] = np.argmax(Q[state])
    return P

def EQ_NV(EQ):
    V = defaultdict(lambda: defaultdict(lambda: 0))
    for state in EQ:
        for goal in EQ[state]:
                V[state][goal] = np.max(EQ[state][goal])
    return V
def EQ_V(EQ, goal=None):
    V = defaultdict(lambda: 0)
    for state in EQ:
        if goal:
            V[state] = np.max(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            V[state] = np.max(np.max(Vs,axis=0))
    return V
def NV_V(NV, goal=None):
    V = defaultdict(lambda: 0)
    for state in NV:
        if goal:
            V[state] = NV[state][goal]
        else:
            Vs = [NV[state][goal] for goal in NV[state].keys()]
            V[state] = np.max(Vs)
    return V
def Q_V(Q):
    V = defaultdict(lambda: 0)
    for state in Q:
        V[state] = np.max(Q[state])
    return V

def EQ_Q(EQ, goal=None):
    Q = defaultdict(lambda: np.zeros(5))
    for state in EQ:
        if goal:
            Q[state] = EQ[state][goal]
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            Q[state] = np.max(Vs,axis=0)
    return Q

def env_R(env, states):
    R = defaultdict(lambda: np.zeros(env.action_space.n))
    for state in states: 
        for action in range(env.action_space.n):
            R[state][action] = env._get_reward(literal_eval(state),action)
                
    return R

#########################################################################################
def MAX(Q1, Q2):
    Q = defaultdict(lambda: 0)
    for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
        Q[s] = np.max([Q1[s],Q2[s]], axis=0)
    return Q

def AVG(Q1, Q2):
    Q = defaultdict(lambda: 0)
    for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
        Q[s] = (Q1[s]+Q2[s])/2
    return Q

#########################################################################################
def EQMAX(EQ,rmax=2): #Estimating EQ_max
    rmax = rmax
    EQ_max = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmax-max(EQ[g][g])
            if s==g:
                EQ_max[s][g] = EQ[s][g]*0 + rmax
            else:      
                EQ_max[s][g] = EQ[s][g] + c   
    return EQ_max

def EQMIN(EQ,rmin=-0.1): #Estimating EQ_min
    rmin = rmin
    EQ_min = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmin-max(EQ[g][g])
            if s==g:
                EQ_min[s][g] = EQ[s][g]*0 + rmin
            else:      
                EQ_min[s][g] = EQ[s][g] + c  
    return EQ_min

def NOT(EQ, EQ_max=None, EQ_min=None):
    EQ_max = EQ_max if EQ_max else EQMAX(EQ)
    EQ_min = EQ_min if EQ_min else EQMIN(EQ)
    EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            EQ_not[s][g] = (EQ_max[s][g]+EQ_min[s][g]) - EQ[s][g]    
    return EQ_not

def OR(EQ1, EQ2):
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ1.keys()):
        for g in list(EQ1[s].keys()):
            EQ[s][g] = np.max([EQ1[s][g],EQ2[s][g]],axis=0)
    return EQ

def AND(EQ1, EQ2):
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for s in list(EQ1.keys()):
        for g in list(EQ1[s].keys()):
            EQ[s][g] = np.min([EQ1[s][g],EQ2[s][g]],axis=0)
    return EQ

#########################################################################################
