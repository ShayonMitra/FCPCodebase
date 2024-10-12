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

class Striker(Role):
    def move2point(self):
        self.move(self.idealPos())

class LeftWing(Role):
    def move2point(self):
        self.move(self.idealPos())

class RightWing(Role):
    def move2point(self):
        self.move(self.idealPos())

class LeftBack(Role):
    def move2point(self):
        self.move(self.idealPos())

class RightBack(Role):
    def move2point(self):
        self.move(self.idealPos())

class GoalKeeper(Role):
    def move2point(self):
        self.move(self.idealPos())

class CenterAttackingMid(Role):
    def move2point(self):
        self.move(self.idealPos())

class RightDefensiveMid(Role):
    def move2point(self):
        self.move(self.idealPos())

class LeftDefensiveMid(Role):
    def move2point(self):
        self.move(self.idealPos())

class LeftCenterBack(Role):
    def move2point(self):
        self.move(self.idealPos())


class RightCenterBack(Role):
    def move2point(self):
        self.move(self.idealPos())