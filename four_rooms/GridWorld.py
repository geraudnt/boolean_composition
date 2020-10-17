from ast import literal_eval
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as colors
from collections import defaultdict

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

# Define colors
COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0, 0.5, 0], 10: [0, 0, 1], 20:[1, 1, 0.0], 21:[0.8, 0.8, 0.8]}

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    MAP = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 1 1 1 0 1 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 1 1 1 1 1 1 1 1 1 1 1 1"

    def __init__(self, MAP=MAP, dense_rewards = False, goal_reward=2, step_reward=-0.1, goals=None, T_states=None, start_position=None, slip_prob=0):

        self.n = None
        self.m = None

        self.grid = None
        self.hallwayStates = None
        self.possiblePositions = []
        self.walls = []
        
        self.MAP = MAP
        self._map_init()
        self.diameter = (self.n+self.m)-4

        self.done = False
        
        self.slip_prob = slip_prob
        
        self.start_position = start_position
        self.position = self.start_position if start_position else (1, 1)
        self.state = [None, self.position]
        
        if goals:
            self.goals = goals
        else:
            self.goals = [(11, 11), (11, 11)]
            
        if T_states:
            self.T_states = T_states
        else:
            self.T_states = self.goals

        # Rewards
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.rmax = 2
        self.rmin = -0.1
        
        self.dense_rewards = dense_rewards
        if self.dense_rewards:
            self.rmin = self.rmin*10

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possiblePositions))
        self.action_space = spaces.Discrete(5)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]
    
    def pertube_action(self,action):      
        if action != STAY:
            a = 1-self.slip_prob
            b = self.slip_prob/(self.action_space.n-2)
            if action == UP:
                probs = [a,b,b,b,0]
            elif action == DOWN:
                probs = [b,b,a,b,0]
            elif action == RIGHT:
                probs = [b,a,b,b,0]
            elif action == LEFT:
                probs = [b,b,b,a,0]
            # else:
            #     probs = [b,b,b,b,a]
            action = np.random.choice(np.arange(len(probs)), p=probs)       
        return action

    def step(self, action):
        assert self.action_space.contains(action)
        
        action = self.pertube_action(action)
        
        g = [None,None] # #virtual state 
        if self.state in self.T_states:
            return str(g), self._get_reward(self.state, action), True, None
        elif action == STAY and ([self.position,self.position] in self.T_states) and not self.state[0]:
            new_state = [self.position, self.position]
        else:
            x, y = self.state[1]
            if action == UP:
                x = x - 1
            elif action == DOWN:
                x = x + 1
            elif action == RIGHT:
                y = y + 1
            elif action == LEFT:
                y = y - 1
            self.position = (x, y)
            new_state = [None, self.position]
        
        reward = self._get_reward(self.state, action)
        
        if self._get_grid_value(new_state) == 1:  # new_state in walls list
            # stay at old state if new coord is wall
            self.position = self.state[1]
        else:
            self.state = new_state
        
        return str(self.state), reward, self.done, None

    def _get_dense_reward(self, state, action):
        g = np.array([g[1] for g in self.goals])
        s = np.array([state[1]]*len(g))
        reward = 0.1*np.mean(np.exp(-0.25*np.linalg.norm(s-g, axis=1)**2))
        return reward

    def _get_reward(self, state, action):      
        reward = 0
        if self.dense_rewards:
            reward += self._get_dense_reward(state,action)

        if state in self.goals:
            reward += self.goal_reward
        else:
            reward += self.step_reward
        
        return reward        
    
    def env_R(self):
        R = defaultdict(lambda: np.zeros(self.action_space.n))
        for position in self.possiblePositions: 
            state = [None,position]
            for action in range(self.action_space.n):
                R[str(state)][action] = self._get_reward(state,action)
            state = [position,position]
            if state in self.T_states:
                for action in range(self.action_space.n):
                    R[str(state)][action] = self._get_reward(state,action) 
        return R

    def reset(self):
        self.done = False
        if not self.start_position:
            idx = np.random.randint(len(self.possiblePositions))
            self.position = self.possiblePositions[idx]  # self.start_state_coord
        else:
            self.position = self.start_position
        self.state = [None, self.position]
        return str(self.state)

    def render(self, fig=None, goal=None, mode='human', P=None, V = None, Q = None, R = None, T = None, Ta = None, title=None, grid=False, cmap='YlOrRd'):

        img = self._gridmap_to_img(goal=goal)        
        if not fig:
            fig = plt.figure(1, figsize=(20, 15), dpi=60, facecolor='w', edgecolor='k')
        
        params = {'font.size': 40}
        plt.rcParams.update(params)
        plt.clf()
        plt.xticks(np.arange(0, 2*self.n, 1))
        plt.yticks(np.arange(0, 2*self.m, 1))
        plt.grid(grid)
        if title:
            plt.title(title, fontsize=20)

        plt.imshow(img, origin="upper", extent=[0, self.n, self.m, 0])
        fig.canvas.draw()
        
        if Q: # For showing rewards
            ax = fig.gca()
            cmap_ = cm.get_cmap(cmap)
            v_max = float("-inf")
            v_min = float("inf")
            for state, q in Q.items():
                if literal_eval(state)[0] or not literal_eval(state)[1]:
                    continue
                for action in range(self.action_space.n):
                    if q[action] > v_max:
                        v_max = q[action]
                    if q[action] < v_min:
                        if action == self.action_space.n-1:
                            v_min += self.step_reward 
                        else:
                            v_min = q[action]
            norm = colors.Normalize(v_min,v_max)
            for state, q in Q.items():
                if literal_eval(state)[0] or not literal_eval(state)[1]:
                    continue
                y, x = literal_eval(state)[1]
                for action in range(self.action_space.n):
                    v = (q[action]-v_min)/(v_max-v_min)
                    self._draw_reward(ax, x, y, action, v, cmap_)
            m = cm.ScalarMappable(norm=norm, cmap=cmap_)
            m.set_array(ax.get_images()[0])
            fig.colorbar(m, ax=ax)
                    
        if V: # For showing optimal values
            ax = fig.gca()
            v = np.zeros((self.m,self.n))+float("-inf")
            for state, val in V.items():
                if literal_eval(state)[0] or not literal_eval(state)[1]:
                    continue
                y, x = literal_eval(state)[1]
                v[y,x] = val  
            c = plt.imshow(v, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            fig.colorbar(c, ax=ax)
                
        if P:  # For drawing arrows of optimal policy
            ax = fig.gca()
            for state, action in P.items():
                if literal_eval(state)[0] or not literal_eval(state)[1]:
                    continue
                y, x = literal_eval(state)[1]
                self._draw_action(ax, x, y, action)
        
        cmap = 'YlOrRd'
        if R: # For showing rewards
            ax = fig.gca()
            cmap_ = cm.get_cmap(cmap)
            norm = colors.Normalize(vmin=self.rmin, vmax=self.rmax)
            for state, reward in R.items():
                y, x = literal_eval(state)[1]
                for action in range(self.action_space.n):
                    if literal_eval(state)[0] and action!=4:
                        continue
                    r = (reward[action]-self.rmin)/(self.rmax-self.rmin)
                    r = r**0.25
                    self._draw_reward(ax, x, y, action, r, cmap_)
            m = cm.ScalarMappable(norm=norm, cmap=cmap_)
            m.set_array(ax.get_images()[0])
            fig.colorbar(m, ax=ax)
        
        if T:  # For showing transition probabilities of single action
            ax = fig.gca()
            vprob = np.zeros((self.m,self.n))+float("-inf")
            for state, prob in T.items():
                if literal_eval(state)[0]:
                    continue
                y, x = literal_eval(state)[1]
                vprob[y,x] = prob  
            c = plt.imshow(vprob, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            fig.colorbar(c, ax=ax)
            
        if Ta:  # For showing transition probabilities of all actions
            ax = fig.gca()
            vprob = np.zeros((self.m,self.n))+float("-inf")
            for state, probs in Ta.items():
                if literal_eval(state)[0]:
                    continue
                y, x = literal_eval(state)[1]
                vprob[y,x] = 0
                for action in range(self.action_space.n):
                    if probs[action]:
                        self._draw_action(ax, x, y, action)
            
        plt.pause(0.00001)  # 0.01
        return None #fig

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1":
                    self.walls.append((i, j))
                # possible states
                else:
                    self.possiblePositions.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possiblePositions:
            if ((self.grid[x - 1][y] == 1) and (self.grid[x + 1][y] == 1)) or \
                    ((self.grid[x][y - 1] == 1) and (self.grid[x][y + 1] == 1)):
                self.hallwayStates.append((x, y))

    def _get_grid_value(self, state):
        return self.grid[state[1][0]][state[1][1]]

    # specific for self.MAP
    def _getRoomNumber(self, state=None):
        if state == None:
            state = self.state
        # if state isn't at hall way point
        xCount = self._greaterThanCounter(state[1], 0)
        yCount = self._greaterThanCounter(state[1], 1)
        room = 0
        if yCount >= 2:
            if xCount >= 2:
                room = 2
            else:
                room = 1
        else:
            if xCount >= 2:
                room = 3
            else:
                room = 0

        return room

    def _greaterThanCounter(self, state, index):
        count = 0
        for h in self.hallwayStates:
            if state[index] > h[index]:
                count = count + 1
        return count

    def _draw_action(self, ax, x, y, action):
        if action == UP:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if action == DOWN:
            x += 0.5
            dx = 0
            dy = 0.4
        if action == RIGHT:
            y += 0.5
            dx = 0.4
            dy = 0
        if action == LEFT:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0
        if action == STAY:
            x += 0.5
            y += 0.5
            dx = 0
            dy = 0
            
            ax.add_patch(plt.Circle((x, y), radius=0.25, fc='k'))
            return

        ax.add_patch(plt.arrow(x,  # x1
                      y,  # y1
                      dx,  # x2 - x1
                      dy,  # y2 - y1
                      facecolor='k',
                      edgecolor='k',
                      width=0.005,
                      head_width=0.4,
                      )
                    )

    def _draw_reward(self, ax, x, y, action, reward, cmap):
        x += 0.5
        y += 0.5
        triangle = np.zeros((3,2))
        triangle[0] = [x,y]
        
        if action == UP:
            triangle[1] = [x-0.5,y-0.5]
            triangle[2] = [x+0.5,y-0.5]
        if action == DOWN:
            triangle[1] = [x-0.5,y+0.5]
            triangle[2] = [x+0.5,y+0.5]
        if action == RIGHT:
            triangle[1] = [x+0.5,y-0.5]
            triangle[2] = [x+0.5,y+0.5]
        if action == LEFT:
            triangle[1] = [x-0.5,y-0.5]
            triangle[2] = [x-0.5,y+0.5]
        if action == STAY:            
            ax.add_patch(plt.Circle((x, y), radius=0.25, color=cmap(reward)))
            return

        ax.add_patch(plt.Polygon(triangle, color=cmap(reward)))


    def _gridmap_to_img(self, goal=None):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    if (i, j) == self.position:#start_position:
                        this_value = COLOURS[10][k]
                    elif goal and (i, j) == literal_eval(goal)[1]:
                        this_value = COLOURS[20][k]
                    elif [(i, j),(i, j)] in self.goals:
                        this_value = COLOURS[3][k]
                    else:
                        colour_number = int(self.grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img