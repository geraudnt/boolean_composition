import pygame

from gym_repoman.envs.collect_env import CollectEnv


class MultiCollectEnv(CollectEnv):

    def __init__(self, termination_condition, reward_condition, start_positions=None):
        super().__init__(goal_condition=termination_condition, start_positions=start_positions)
        self.reward_condition = reward_condition

    def step(self, action):
        direction = self._ACTIONS[action]
        prev_pos = self.player.position
        next_pos = (direction[0] + prev_pos[0], direction[1] + prev_pos[1])
        if self.board[next_pos] != '#':
            self.player.step(direction)

        collected = pygame.sprite.spritecollide(self.player, self.collectibles, True)
        self.collected.add(collected)
        self.render_group.remove(collected)

        reward = -0.1
        if len(collected) > 0:
            if self.reward_condition(collected[0]):
                reward = 1

        done = False
        if self.goal_condition(self.collected):
            done, reward = True, 1.0

        obs = self._draw_screen(self._surface)
        return obs, reward, done, {'collected': self.collected}
