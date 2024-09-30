from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import itertools
import agent.Roles

class Tactics():

    def __init__(self, enable_log, enable_draw, world) -> None:
        
        self.world = world
        self.bestRoleMap = {}
        self.old_pos = (0,0)
        self.cycles = 0
        self.previous_positions = {}
        self.enable_draw = enable_draw
        self.role=None
        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation
        self.player_formation =None


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
        player_positions = self.player_formation # we will fill this in think and send by calling Tactics class 

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
