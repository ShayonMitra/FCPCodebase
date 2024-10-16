import math
import numpy as np
import itertools
from world.World import World
# behaviors
from behaviors.Behavior import Behavior

class Role:
    def __init__(self, idealx, idealy, world) -> None:
        self.world = world
        self.idealPos = (idealx, idealy)
    def move2point(self):
        self.move(self.idealPos())

class Striker(Role):
    pass

class LeftWing(Role):
    pass

class RightWing(Role):
    pass

class LeftBack(Role):
    pass

class RightBack(Role):
    pass

class GoalKeeper(Role):
    pass

class CenterAttackingMid(Role):
    pass

class RightDefensiveMid(Role):
    pass

class LeftDefensiveMid(Role):
    pass

class LeftCenterBack(Role):
    pass

class RightCenterBack(Role):
    pass