import numpy as np
import torch
from gym.wrappers import Monitor

from dqn import Agent, DQN, FloatTensor
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame


def video_callable(episode_id):
    return episode_id > 1 and episode_id % 500 == 0


def train(path, env):
    #env = Monitor(env, path, video_callable=video_callable, force=True)
    agent = Agent(env,path=path)
    agent.train()
    return agent


def save(path, agent):
    torch.save(agent.q_func.state_dict(), path)


def load(path, env):
    dqn = DQN(env.action_space.n)
    dqn.load_state_dict(torch.load(path))
    return dqn

start_positions = {'crate_beige': (3, 4),
                   'player': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_blue': (1, 1),
                   'crate_purple': (8, 1),
                   'circle_blue': (1, 8)}

def learn(colour, shape, condition):
    name = colour + shape
    base_path = './models/{}/'.format(name)
    env = WarpFrame(CollectEnv(start_positions=start_positions,goal_condition=condition))
    agent = train(base_path, env)
    save(base_path + 'model.dqn', agent)

if __name__ == '__main__':

    learn('blue', '', lambda x: x.colour == 'blue')
