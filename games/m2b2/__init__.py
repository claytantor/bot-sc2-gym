import os
import sys
import numpy as np
import pygame

from pygame.constants import K_UP, K_DOWN, K_RIGHT, K_LEFT
from envs.base import PyGameWrapper

MOVE_SIZE = 8
BEACON_MOVE_SIZE = 0
BEACON_UPDATE = 10

GRID_SIZE = 16

WHITE = (255, 255, 255)
GREEN = (11, 252, 3)
RED = (252, 3, 3)
YELLOW = (248, 252, 3)
BLACK = (0, 0, 0)

SURFACE_ID = 1
START_ID = 3
GOAL_ID = 2

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class Player():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, map_info):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        # always 16x16 (one unit)
        self.rect = (0,0,GRID_SIZE,GRID_SIZE)

        self.width = self.rect[2]
        self.height = self.rect[3]

        self.init(map_info)


    def move(self, a):

        col = self.col
        row = self.row
        
        if a == LEFT:
            col = max(self.col - 1, 0)
            # print("left", self.col, col)
            self.pos_x = col*GRID_SIZE

        elif a == DOWN:
            row = min(self.row + 1, self.map_info.shape[1] - 1)
            # print("down",self.row, row)
            self.pos_y = row*GRID_SIZE

        elif a == RIGHT:
            col = min(self.col + 1, self.map_info.shape[0] - 1)
            # print("right",self.col, col)
            self.pos_x = col*GRID_SIZE

        elif a == UP:
            row = max(self.row - 1, 0)
            # print("up",self.row, row)
            self.pos_y = row*GRID_SIZE

        # # save state
        self.col = col
        self.row = row

        # print("move", self.pos_x, self.pos_y)
            

        
    def init(self, map_info):
        self.map_info = map_info

        rows_y, cols_x = np.where(self.map_info == START_ID)

        self.col = cols_x[0]
        self.row = rows_y[0]

        self.pos_y = rows_y[0]*GRID_SIZE
        self.pos_x = cols_x[0]*GRID_SIZE



    def update(self):   
        # self.rect.center = (self.pos_x, self.pos_y)  
        pass 

    def draw(self, screen):
        GAME_FONT = pygame.font.SysFont('droidsansmonoforpowerline', 10)

        pos_txt_img = GAME_FONT.render('{},{}'.format(self.pos_x, self.pos_y), True, WHITE)
        screen.blit(pos_txt_img, (3, 3))

        # action_txt_img = GAME_FONT.render('{}'.format(self.action_direction), True, self.action_color)
        # screen.blit(action_txt_img, (3, 13))

        # print("draw player", self.pos_x, self.pos_y, self.col, self.row)
        pygame.draw.rect(screen,YELLOW,(self.pos_x, self.pos_y, self.width, self.height))


class Beacon():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, map_info):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        # always 16x16 (one unit)
        self.rect = (0,0,GRID_SIZE,GRID_SIZE)

        self.width = self.rect[2]
        self.height = self.rect[3]

        self.init(map_info)

    def init(self, map_info):
        self.map_info = map_info

        rows_y, cols_x = np.where(self.map_info == GOAL_ID)


        self.col = cols_x[0]
        self.row = rows_y[0]

        self.pos_x = cols_x[0]*GRID_SIZE
        self.pos_y = rows_y[0]*GRID_SIZE

        # print("GOAL_ID",self.map_info, cols_x, rows_y, self.pos_x, self.pos_y )

        
    def update(self):
        # self.rect.center = (self.pos_x, self.pos_y)  
        pass 

    def draw(self, screen):
        #print("draw beacon", self.pos_x, self.pos_y, self.col, self.row)
        pygame.draw.rect(screen, GREEN, (self.pos_x, self.pos_y, self.width, self.height))

class Background():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

    def draw(self, screen):
        pygame.draw.rect(screen,BLACK,(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT))


class MoveToBeacon(PyGameWrapper):

    def __init__(self, map_info):

    
        self.actions = {
            "up": K_UP, 
            "down": K_DOWN,
            "right": K_RIGHT,
            "left": K_LEFT
        }

        self.backdrop = None
        self.player = None
        self.beacon = None

        self.score = 0.0
        self.event_key = ''

        self.set_map(map_info)
        
        PyGameWrapper.__init__(self, self.width, self.height, actions=self.actions)
        PyGameWrapper._setup(self)

        self.init()

    def set_map(self, map_info):
        # make the playing area a 
        # grid with 16 pixel squares
        self.map_info = map_info
        map_shape = map_info.shape
        self.width = map_shape[0]*16
        self.height = map_shape[1]*16


    def init(self):
    
        self.score = 0.0
        self.lives = 1
        self.game_tick = 0

        self.reset_game()


    def reset_game(self):

        if self.backdrop is None:
            self.backdrop = Background(
                self.width,
                self.height
            )


        if self.player is None:
            self.player = Player(
                self.width,
                self.height,
                self.map_info)
        else:
            self.player.init(self.map_info)


        if self.beacon is None:
            self.beacon = Beacon(
                self.width,
                self.height,
                self.map_info)
        else:
            self.beacon.init(self.map_info)
    
        self.backdrop.draw(self.screen)
        self.beacon.draw(self.screen)
        self.player.draw(self.screen)

            
    def game_over(self):
        return self.game_tick > 100

    @property
    def tick_count(self):
        return self.game_tick

    def step(self, action):

        lookup = {
            0:"LEFT",
            1:"DOWN",
            2:"RIGHT",
            3:"UP"
        }
        
        self.player.move(action)
        self.player.update()
        self.beacon.update()
         
        self.backdrop.draw(self.screen)

        GAME_FONT = pygame.font.SysFont('droidsansmonoforpowerline', 10)

        action_txt = GAME_FONT.render(lookup[action], True, WHITE)
        self.screen.blit(action_txt, (3, 13))

        step_txt = GAME_FONT.render(str(self.game_tick), True, WHITE)
        self.screen.blit(step_txt, (3, 23))

        self.beacon.draw(self.screen)
        self.player.draw(self.screen)
        
        self.game_tick += 1

        pygame.display.update()


