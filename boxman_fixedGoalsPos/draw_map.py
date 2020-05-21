from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame
import scipy.misc


if __name__ == '__main__':


    start_positions = {'crate_beige': (3, 4),
                       'player': (6, 3),
                       'circle_purple': (7, 7),
                       'circle_beige': (1, 7),
                       'crate_blue': (1, 1),
                       'crate_purple': (8, 1),
                       'circle_blue': (1, 8)}
    env = WarpFrame(CollectEnv(start_positions=start_positions,
                               task_condition=lambda x: x.colour == 'purple' or x.colour == 'blue'))

    env.reset()
    image = env.render()

    scipy.misc.imsave('map.png', image)

