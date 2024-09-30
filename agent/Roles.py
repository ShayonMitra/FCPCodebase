import math
import numpy as np
import itertools
from world.World import World
# behaviors
from behaviors.Behavior import Behavior

class Role:
    def __init__(self, unum: int, world) -> None:
        self.world = world

    def calculate_player_positions(self, current_role: int):
        '''
        Calculate the positions of all players/roles based on the current ball position.
        Returns: numpy array of 9 positions (x,y) for 9 players.
        '''
        w = self.world
        ball_2d = w.ball_abs_pos[:2]

        # Initialise all 9 positions to (0,0) 
        player_positions = np.array([(0.0, 0.0) for _ in range(9)])
        # x and y coordinates of the ball
        bx = ball_2d[0]
        by = ball_2d[1]

        # Basic 4-3-3 formation with roles Left Wing, Right Wing, Center Mid... and so on

        # Coordinate for Left Wing
        player_positions[8] = (bx, (10.0 + by) / 2.0)
        
        # Coordinate for Right Wing
        player_positions[7] = (bx, (by - 10.0) / 2.0)
        
        # Coordinate for Center Mid
        player_positions[6] = ((-15 + 2 * bx) / 3.0, (2 * by + 10.0) / 3.0)
        
        # Coordinate for Left Center Mid
        player_positions[5] = ((-15 + 2 * bx) / 3.0, by)
        
        # Coordinate for Right Center Mid
        player_positions[4] = ((-15 + 2 * bx) / 3.0, (2 * by - 10.0) / 3.0)
        
        # Coordinate for Right Back
        player_positions[3] = ((-30 + bx) / 3.0, (3 * by - 20.0) / 5.0)
        
        # Coordinate for Right Center Back
        player_positions[2] = ((-30 + bx) / 3.0, (4 * by - 10.0) / 5.0)
        
        # Coordinate for Left Center Back
        player_positions[1] = ((-30 + bx) / 3.0, (4 * by + 10.0) / 5.0)
        
        # Coordinate for Left Back
        player_positions[0] = ((-30 + bx) / 3.0, (3 * by + 20.0) / 5.0)

        # Return the position of the current role
        return player_positions[current_role]

class Striker(Role):
    def calcpos(self):
        return self.calculate_player_positions(6)

    def move2point(self):
        self.move(self.calcpos())

class LeftWing(Role):
    def calcpos(self):
        return self.calculate_player_positions(8)

    def move2point(self):
        self.move(self.calcpos())

class RightWing(Role):
    def calcpos(self):
        return self.calculate_player_positions(7)

    def move2point(self):
        self.move(self.calcpos())

class LeftBack(Role):
    def calcpos(self):
        return self.calculate_player_positions(0)

    def move2point(self):
        self.move(self.calcpos())

class RightBack(Role):
    def calcpos(self):
        return self.calculate_player_positions(3)

    def move2point(self):
        self.move(self.calcpos())

class GoalKeeper(Role):
    pass

class MidForward(Role):
    def calcpos(self):
        return self.calculate_player_positions(5)

    def move2point(self):
        self.move(self.calcpos())

class MidBack(Role):
    def calcpos(self):
        return self.calculate_player_positions(4)

    def move2point(self):
        self.move(self.calcpos())
