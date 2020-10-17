import collections
import os
import random

import gym
import numpy as np
import pygame
from gym.spaces import Box, Discrete
from pygame.compat import geterror
from pygame.locals import QUIT

main_dir = os.path.split(os.path.abspath(__file__))[0]
assets_dir = os.path.join(main_dir, 'assets')


def _load_image(name):
    fullname = os.path.join(assets_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert_alpha()
    return image


def _calculate_topleft_position(position, sprite_size):
    return sprite_size * position[1], sprite_size * position[0]


class _Collectible(pygame.sprite.Sprite):
    _COLLECTIBLE_IMAGES = {
        ('square', 'purple'): 'purple_square.png',
        ('circle', 'purple'): 'purple_circle.png',
        ('square', 'beige'): 'beige_square.png',
        ('circle', 'beige'): 'beige_circle.png',
        ('square', 'blue'): 'blue_square.png',
        ('circle', 'blue'): 'blue_circle.png'
    }

    def __init__(self, sprite_size, shape, colour):
        self.name = shape + '_' + colour
        self._sprite_size = sprite_size
        self.shape = shape
        self.colour = colour
        pygame.sprite.Sprite.__init__(self)
        image = _load_image(self._COLLECTIBLE_IMAGES[(self.shape, self.colour)])
        self.high_res = image
        self.low_res = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.image = self.low_res
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)


class _Player(pygame.sprite.Sprite):
    def __init__(self, sprite_size):
        self.name = 'player'
        self._sprite_size = sprite_size
        pygame.sprite.Sprite.__init__(self)
        image = _load_image('character.png')
        self.high_res = image
        self.low_res = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.image = self.low_res
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    def step(self, move):
        self.position = (self.position[0] + move[0], self.position[1] + move[1])
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)


class CollectEnv(gym.Env):
    """
    This environment consists of an agent attempting to collect a number of objects. The agents has four actions
    to move him up, down, left and right, but may be impeded by walls.
    There are two types of objects in the environment: fridges and TVs, each of which take one of three colours
    (white, blue and purple) for a total of six objects.

    The objects the agent must collect can be specified by passing a goal condition lambda to the environment.

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    _BOARDS = {
        'original': ['##########',
                     '#        #',
                     '#        #',
                     '#    #   #',
                     '#   ##   #',
                     '#  ##    #',
                     '#   #    #',
                     '#        #',
                     '#        #',
                     '##########'],
    }

    _AVAILABLE_COLLECTIBLES = [
        ('square', 'purple'),
        ('circle', 'purple'),
        ('square', 'beige'),
        ('circle', 'beige'),
        ('square', 'blue'),
        ('circle', 'blue')
    ]

    _WALL_IMAGE = 'wall.png'
    _GROUND_IMAGE = 'ground.png'

    _ACTIONS = {
        0: (-1, 0),  # North
        1: (0, 1),  # East
        2: (1, 0),  # South
        3: (0, -1),  # West
        4: (0, 0)  # Stay
    }

    _SCREEN_SIZE = (400, 400)
    _SPRITE_SIZE = 40

    def __init__(self, board='original', available_collectibles=None, start_positions=None,
                 goal_condition=lambda _: True):
        """
        Create a new instance of the RepoMan environment. The observation space is a single RGB image of size 400x400,
        and the action space are four discrete actions corresponding to North, East, South and West.


        :param board: the state of the walls and free space
        :param available_collectibles: the items that will be placed in the task
        :param start_positions: an option parameter to specify the starting position of the objects and player
        :param goal_condition: a lambda that determines when to end the episode based on teh object just collected.
        By default, the episode ends when any object is collected.
        """
        self.goal = None
        self.viewer = None
        self.start_positions = start_positions
        self.goal_condition = goal_condition
        self.action_space = Discrete(4)
        self.rmin = -0.1
        self.rmax = 2

        self.available_collectibles = available_collectibles \
            if available_collectibles is not None else self._AVAILABLE_COLLECTIBLES

        self.board = np.array([list(row) for row in self._BOARDS[board]])
        self.diameter = sum(self.board.shape)

        self.observation_space = Box(0, 255, [self._SCREEN_SIZE[0], self._SCREEN_SIZE[1], 3], dtype=int)
        pygame.init()
        pygame.display.init()
        pygame.display.set_mode((1, 1))

        self._bestdepth = pygame.display.mode_ok(self._SCREEN_SIZE, 0, 32)
        self._surface = pygame.Surface(self._SCREEN_SIZE, 0, self._bestdepth)
        self._background = pygame.Surface(self._SCREEN_SIZE)
        self._clock = pygame.time.Clock()

        self.free_spaces = list(map(tuple, np.argwhere(self.board != '#')))
        self._build_board()

        self.initial_positions = None
        self.collectibles = pygame.sprite.Group()
        self.collected = pygame.sprite.Group()
        self.render_group = pygame.sprite.RenderPlain()
        self.player = _Player(self._SPRITE_SIZE)
        self.render_group.add(self.player)

        for shape, colour in self.available_collectibles:
            self.collectibles.add(_Collectible(self._SPRITE_SIZE, shape, colour))

    def _build_board(self):
        for col in range(self.board.shape[1]):
            for row in range(self.board.shape[0]):
                position = _calculate_topleft_position((row, col), self._SPRITE_SIZE)
                image = self._WALL_IMAGE if self.board[row, col] == '#' else self._GROUND_IMAGE
                image = _load_image(image)
                image = pygame.transform.scale(image, (self._SPRITE_SIZE, self._SPRITE_SIZE))
                self._background.blit(image, position)

    def _draw_screen(self, surface,draw_background=True):
        if draw_background:
            surface.blit(self._background, (0, 0))
        else:
            self._background.fill((0,0,0))
        self.render_group.draw(surface)
        surface_array = pygame.surfarray.array3d(surface)
        observation = np.copy(surface_array).swapaxes(0, 1)
        del surface_array
        return observation

    def reset(self):
        self.goal = None
        collected = self.collected.sprites()
        self.collectibles.add(collected)
        self.collected.empty()
        self._build_board()
        self.render_group.empty()
        self.render_group.add(self.collectibles.sprites())
        self.render_group.add(self.player)
        
        for sprite in self.collectibles:            
            sprite.image = sprite.low_res
            self.player.image = self.player.low_res

        render_group = sorted(self.render_group, key=lambda x: x.name)
        if self.start_positions is None:
            positions = random.sample(self.free_spaces, k=len(render_group))
        else:
            start_positions = collections.OrderedDict(sorted(self.start_positions.items()))
            positions = start_positions.values()

        self.initial_positions = collections.OrderedDict()
        
        terminal = False
        for position, sprite in zip(positions, render_group):
            self.initial_positions[sprite] = position
            sprite.reset(position)
        
        collected = pygame.sprite.spritecollide(self.player, self.collectibles, False)
        self.collected.add(collected)
        self.render_group.remove(collected)
        if len(collected)>0:
            if self.goal_condition(collected[0]):
                self.goal = True
                
                image = _load_image(self._GROUND_IMAGE)
                image = pygame.transform.scale(image, self._SCREEN_SIZE)
                self._background.blit(image, (0,0))

                self.render_group.empty()
                collected[0].reset((0,0))
                self.player.reset((0,0))             
                collected[0].image = pygame.transform.scale(collected[0].high_res, self._SCREEN_SIZE)
                self.player.image = pygame.transform.scale(self.player.high_res, self._SCREEN_SIZE)
                self.render_group.add(collected)
                self.render_group.add(self.player)
            
        return self._draw_screen(self._surface)

    def step(self, action):    
        direction = self._ACTIONS[action]
        prev_pos = self.player.position
        next_pos = (direction[0] + prev_pos[0], direction[1] + prev_pos[1])
        if self.board[next_pos] != '#':
            self.player.step(direction)
        
        collected = pygame.sprite.spritecollide(self.player, self.collectibles, False)
        self.collected.add(collected)
        self.render_group.remove(collected)
         
        done, reward = False, self.rmin        
        if self.goal:
            done, reward = True, self.rmax
            return self._draw_screen(self._surface), reward, done, {'collected': self.collected}
        
        if len(collected) > 0:#self.action_space.n-1:
            if self.goal_condition(collected[0]):
                self.goal = True
            
                image = _load_image(self._GROUND_IMAGE)
                image = pygame.transform.scale(image, self._SCREEN_SIZE)
                self._background.blit(image, (0,0))
                
                self.render_group.empty()
                collected[0].reset((0,0))
                self.player.reset((0,0))             
                collected[0].image = pygame.transform.scale(collected[0].high_res, self._SCREEN_SIZE)
                self.player.image = pygame.transform.scale(self.player.high_res, self._SCREEN_SIZE)
                self.render_group.add(collected)
                self.render_group.add(self.player)
                return self._draw_screen(self._surface), reward, done, {'collected': self.collected}
        else:
            collected = []
             
        return self._draw_screen(self._surface), reward, done, {'collected': self.collected}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                pygame.quit()
                self.viewer = None
            return
        
        if self.viewer is None:
            self.viewer = pygame.display.set_mode(self._SCREEN_SIZE, 0, self._bestdepth)

        self._clock.tick(10 if mode != 'human' else 2)
        arr = self._draw_screen(self.viewer)
        pygame.display.flip()
        return arr
