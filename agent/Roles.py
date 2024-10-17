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
    def __repr__(self) -> str:
        return "Striker"

class LeftWing(Role):
    def __repr__(self) -> str:
        return "LeftWing"

class RightWing(Role):
    def __repr__(self) -> str:
        return "RightWing"

class LeftBack(Role):
    def __repr__(self) -> str:
        return "LeftBack"

class RightBack(Role):
    def __repr__(self) -> str:
        return "RightBack"

class GoalKeeper(Role):
    def __repr__(self) -> str:
        return "GoalKeeper"

class CenterAttackingMid(Role):
    def __repr__(self) -> str:
        return "CenterAttackingMid"

class RightDefensiveMid(Role):
    def __repr__(self) -> str:
        return "RightDefensiveMid"

class LeftDefensiveMid(Role):
    def __repr__(self) -> str:
        return "LeftDefensiveMid"

class LeftCenterBack(Role):
    def __repr__(self) -> str:
        return "LeftCenterBack"

class RightCenterBack(Role):
    def __repr__(self) -> str:
        return "RightCenterBack"