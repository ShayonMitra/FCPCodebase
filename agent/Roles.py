import math
import numpy as np
import itertools
from world.World import World

class Role:
    def __init__(self, idealx, idealy, world) -> None:
        self.world = world
        self.idealPos = (idealx, idealy)
        self.behavior = world.behavior
        self.path_manager = world.path_manager
        self.scom = world.scom
        self.kick_direction = 0
        self.kick_distance = 0
        
        self.preprocessing()

    def preprocessing(self):
        w = self.world
        slow_ball_pos = w.get_predicted_ball_pos(0.5) # predicted future 2D ball position when ball speed <= 0.5 m/s

        # list of squared distances between teammates (including self) and slow ball (sq distance is set to 1000 in some conditions)
        teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000 # force large distance if teammate does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in w.teammates ]

        # list of squared distances between opponents and slow ball (sq distance is set to 1000 in some conditions)
        opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and w.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000 # force large distance if opponent does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in w.opponents ]

        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(min_teammate_ball_sq_dist)   # distance between ball and closest teammate
        self.min_opponent_ball_dist = math.sqrt(min(opponents_ball_sq_dist)) # distance between ball and closest opponent

    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool    
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance

    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control

    def default_action(self):
        self.move(self.idealPos, orientation=None)

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