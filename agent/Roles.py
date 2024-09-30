from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import itertools

class Role(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        # robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]
        robot_type = (0,2,1,2,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.bestRoleMap = {}
        self.old_pos = (0,0)
        self.cycles = 0
        self.previous_positions = {}
        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy
        self.role=None
        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def calculate_player_positions(self):
        '''
        Calculate the positions of all players/roles based on the current ball position.
        returns: numpy array of 9 positions (x,y) for 9 players
        '''
        w = self.world
        ball_2d = w.ball_abs_pos[:2]

        # Initialise all 9 positions to (0,0) 
        player_positions = np.array([(0.0, 0.0) for _ in range(9)])
        # x and y coordinates of the ball
        bx = ball_2d[0]
        by = ball_2d[1]

        # Basic 4-3-3 formation with roles Left Wing, Right Wing, Center Mid... and so on
        # self.role.calcul
        # Coordinate for Left Wing
        player_positions[8] = (bx, (10.0 + by)/2.0)
        
        # Coordinate for Right Wing
        player_positions[7] = (bx, (by + (-10.0))/2.0)
        
        # Coordinate for Center Mid
        player_positions[6] = ((-15*1.0 + bx*2.0)/3.0, (by*2.0 + 10.0)/3.0)
        
        # Coordinate for Left Center Mid
        player_positions[5] = ((-15*1.0 + bx*2.0)/3.0, by)
        
        # Coordinate for Right Center Mid
        player_positions[4] = ((-15*1.0 + bx*2.0)/3.0, (by*2.0 + (-10.0))/3.0)
        
        # Coordinate for Right Back
        player_positions[3] = ((-15*2.0 + bx*1.0)/3.0, (by*3.0 + (-10.0)*2.0)/5.0)
        
        # Coordinate for Right Center Back
        player_positions[2] = ((-15*2.0 + bx*1.0)/3.0, (by*4.0 + (-10.0)*1.0)/5.0)
        
        # Coordinate for Left Center Back
        player_positions[1] = ((-15*2.0 + bx*1.0)/3.0, (by*4.0 + (10.0)*1.0)/5.0)
        
        # Coordinate for Left Back
        player_positions[0] = ((-15*2.0 + bx*1.0)/3.0, (by*3.0 + (10.0)*2.0)/5.0)

        # Return a numpy array consisting of coordinates of all players
        return player_positions

class Striker(Role):
    pass

class  LeftWing(Role):
    pass

class RightWing(Role):
    pass

class LeftBack(Role):
    pass

class RightBack(Role):
    pass

class GoalKeeper(Role):
    pass

class MidForward(Role):
    pass

class MidBack(Role):
    pass
