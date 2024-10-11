from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import itertools


class Agent(Base_Agent):
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


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


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

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

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

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()




    def think_and_send(self):
        w = self.world
        r = self.world.robot  
        my_head_pos_2d = r.loc_head_position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_sq_dist = ball_dist * ball_dist # for faster comparisons
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
        behavior = self.behavior
        goal_dir = M.target_abs_angle(ball_2d,(15.05,0))
        path_draw_options = self.path_manager.draw_options
        PM = w.play_mode
        PM_GROUP = w.play_mode_group

        #--------------------------------------- 1. Preprocessing

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

        active_player_unum = teammates_ball_sq_dist.index(min_teammate_ball_sq_dist) + 1


        #--------------------------------------- 2. Decide action



        if PM == w.M_GAME_OVER:
            pass
        elif PM_GROUP == w.MG_ACTIVE_BEAM:
            self.beam()
        elif PM_GROUP == w.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1 # return to normal state if get up behavior has finished
        elif PM == w.M_OUR_KICKOFF:
            if r.unum == 9:
                self.kick(120,3) # no need to change the state when PM is not Play On
            else:
                self.move(self.init_pos, orientation=ball_dir) # walk in place
        elif PM == w.M_THEIR_KICKOFF:
            self.move(self.init_pos, orientation=ball_dir) # walk in place
        
        
        elif active_player_unum != r.unum: # I am not the active player
            if r.unum == 1: # I am the goalkeeper
                self.move(self.init_pos, orientation=ball_dir) # walk in place 
            else:

                # dynamic role assignment to calculate the new position of the inactive players
                if self.cycles == 0:
                    role_mapping = self.dynamic_role_assignment(active_player_unum)
                    try:
                        # print("Assigning roles to inactive players dynamically! State = ", self.state)
                        new_x, new_y = role_mapping[r.unum]
                        self.old_pos = (new_x, new_y)
                        # print("AGENT", r.unum, "GO TO : ", new_x, new_y)
                    except:
                        # print("Failed to assign role, State = ", self.state)
                        new_x, new_y = ball_2d[0], ball_2d[1]
                    self.cycles = 100
                else:
                    self.cycles -= 1
                    new_x, new_y = self.old_pos

                # Now call the new EWMA function to smooth the transition
                smoothed_x, smoothed_y = self.apply_ewma_to_positions((new_x, new_y), r.unum)

                # Move the agent to the smoothed position
                # self.move((smoothed_x, smoothed_y), orientation=ball_dir, priority_unums=[active_player_unum])

                # compute basic formation position based on ball position
                # new_x = max(0.5,(ball_2d[0]+15)/15) * (self.init_pos[0]+15) - 15
            
                # if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                #     new_x = min(new_x + 3.5, 13) # advance if team has possession
                # self.move((new_x,self.init_pos[1]), orientation=ball_dir, priority_unums=[active_player_unum])
                if self.enable_draw: 
                    d = w.draw
                    d.point((new_x,new_y), 5, d.Color.blue, "target", False) # last ball prediction
                    d.annotation((new_x,new_y), f"{r.unum}" , d.Color.yellow, "target")

                self.move((new_x,new_y), orientation=None, priority_unums=[active_player_unum])

        else: # I am the active player
            # print("Active player", r.unum)
            path_draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True) # enable path drawings for active player (ignored if self.enable_draw is False)
            enable_pass_command = (PM == w.M_PLAY_ON and ball_2d[0]<6)

            if r.unum == 1 and PM_GROUP == w.MG_THEIR_KICK: # goalkeeper during their kick
                self.move(self.init_pos, orientation=ball_dir) # walk in place 
            if PM == w.M_OUR_CORNER_KICK:
                self.kick( -np.sign(ball_2d[1])*95, 5.5) # kick the ball into the space in front of the opponent's goal
                # no need to change the state when PM is not Play On
            elif self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist: # defend if opponent is considerably closer to the ball
                if self.state == 2: # commit to kick while aborting
                    self.state = 0 if self.kick(abort=True) else 2
                else: # move towards ball, but position myself between ball and our goal
                    self.move(slow_ball_pos + M.normalize_vec((-16,0) - slow_ball_pos) * 0.2, is_aggressive=True)
            else:
                self.state = 0 if self.kick(goal_dir,9,False,enable_pass_command) else 2

            path_draw_options(enable_obstacles=False, enable_path=False) # disable path drawings

        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( r.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""

        #---------------------- annotations for debugging
        if self.enable_draw: 
            d = w.draw
            # d.point((new_x,new_y), 5, d.Color.blue, "status", False) # last ball prediction

            if active_player_unum == r.unum:
                d.point(slow_ball_pos, 3, d.Color.pink, "status", False) # predicted future 2D ball position when ball speed <= 0.5 m/s
                d.point(w.ball_2d_pred_pos[-1], 5, d.Color.pink, "status", False) # last ball prediction
                d.annotation((*my_head_pos_2d, 0.6), "I've got it!" , d.Color.yellow, "status")
            else:
                d.clear("status")




    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")

#----------------------------------- DYNAMIC ROLE ASSIGNMENT 

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


    def dynamic_role_assignment(self, active_player_unum):
        '''
        Assign roles to the inactive players based on the current ball position.
        return : dictionary mapping agent_unum to their assigned positions
        '''
        w = self.world  # Access the world object

        # If the goalkeeper is the active player, randomly assign active player in dynamic role assignment
        if active_player_unum == 1 :
            active_player_unum = np.random.randint(2, 12)

        agents_unums = [_ for _ in range(2, 12)]
        try: 
            agents_unums.remove(active_player_unum)
        except:
            print(active_player_unum)

        # Calculate ideal positions for each player
        player_positions = self.calculate_player_positions()

        # Create a dictionary mapping agent_unum to their current positions
        agent_current_positions = {
            p.unum: p.state_abs_pos[:2] if p.state_last_update != 0 and 
            (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and 
            not p.state_fallen 
            else np.array([1000, 1000])  # Large distance if the agent is invalid, not updated, or has fallen
            for p in w.teammates if p.unum in agents_unums
        }

        # Initialize bestRoleMap to store optimal assignments for subsets
        n = len(agents_unums)
        self.bestRoleMap = {frozenset(): (None, 0)}  # (Mapping, Cost) for empty set

        # Iterate through each role (position) from 1 to n
        for k in range(1, n + 1):
            pk = player_positions[k - 1]  # Current position to assign (0-based index)

            # Iterate over each agent to assign the current role
            for a in agents_unums:
                remaining_agents = [agent for agent in agents_unums if agent != a]
                subsets = list(itertools.combinations(remaining_agents, k - 1))

                # Iterate through each subset of agents
                for subset in subsets:
                    subset_set = frozenset(subset)
                    m0, m0_cost = self.bestRoleMap.get(subset_set, (None, float('inf')))

                    # Create new mapping by adding current agent and role
                    new_mapping = {a: pk}
                    if m0:
                        new_mapping.update(m0)

                    # Calculate the cost for this new assignment
                    cost = self.calculate_cost(new_mapping, agent_current_positions)
                    if (subset_set | frozenset({a})) not in self.bestRoleMap or cost < self.bestRoleMap[subset_set | frozenset({a})][1]:
                        self.bestRoleMap[subset_set | frozenset({a})] = (new_mapping, cost)

        return self.bestRoleMap[frozenset(agents_unums)][0]  # Return the optimal assignment

    def calculate_cost(self, mapping, current_positions):
        """
        Calculate the lexicographically sorted tuple of distances for the given role assignment mapping.

        :param mapping: Dictionary mapping agents to positions.
                        Each key in the mapping is an agent uniform number, and each value is the assigned position.
        :param current_positions: Dictionary or list that stores the current positions of agents, indexed by their uniform numbers.
                                For example, {2: (x2, y2), 3: (x3, y3), ..., 11: (x11, y11)}
        :return: A lexicographically (reverse) sorted tuple of distances.
        """
        # List to store the individual distances
        distances = []

        # Calculate the distance for each agent to its assigned role position
        for agent_unum, target_position in mapping.items():
            # Retrieve the current position of the agent
            agent_current_position = current_positions[agent_unum]  # This assumes current_positions is a dictionary with agent_unum as keys

            # Calculate the Euclidean distance between the agent's current position and the target position
            distance = np.linalg.norm(np.array(agent_current_position) - np.array(target_position))

            # Add the distance to the list
            distances.append(distance)

        # Sort the distances in decreasing order to form the lexicographical tuple
        sorted_distances = tuple(sorted(distances, reverse=True))

        return sorted_distances

    def apply_ewma_to_positions(self, new_position, unum, alpha=0.1):
        """
        Apply Exponentially Weighted Moving Average (EWMA) to smooth the transition of positions.

        :param current_position: tuple of current (smoothed) position (x, y)
        :param new_position: tuple of new assigned position (x, y)
        :param unum: agent's uniform number
        :param alpha: smoothing factor, controls how much weight is given to the new position (0 < alpha < 1)
        :return: smoothed (x, y) position
        """
        
        # If this agent has a previous position stored, apply EWMA
        if unum in self.previous_positions:
            prev_x, prev_y = self.previous_positions[unum]
            
            # Apply the EWMA formula
            smoothed_x = alpha * new_position[0] + (1 - alpha) * prev_x
            smoothed_y = alpha * new_position[1] + (1 - alpha) * prev_y
        else:
            # If no previous position, just use the new position
            smoothed_x, smoothed_y = new_position

        # Store the smoothed position for future cycles
        self.previous_positions[unum] = (smoothed_x, smoothed_y)

        return smoothed_x, smoothed_y
