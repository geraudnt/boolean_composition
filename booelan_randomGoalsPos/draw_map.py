from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame
import scipy.misc


if __name__ == '__main__':

    start_positions = {'player': (3, 4),
                       'crate_purple': (6, 3),
                       'circle_purple': (7, 7),
                       'circle_beige': (1, 7),
                       'crate_beige': (2, 2),
                       'crate_blue': (8, 1),
                       'circle_blue': (2, 8)}
    env = WarpFrame(CollectEnv(start_positions=start_positions,
                               goal_condition=lambda x: x.colour == 'purple' or x.colour == 'blue'))

    env.reset()
    image = env.render()

    scipy.misc.imsave('map.png', image)

