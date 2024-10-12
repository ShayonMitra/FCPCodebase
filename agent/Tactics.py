from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import itertools
from agent.Roles import *

class Tactics():

    def __init__(self, enable_log, enable_draw, world, unum) -> None:
        self.world = world
        self.bestRoleMap = {}
        self.old_pos = (0, 0)
        self.cycles = 0
        self.previous_positions = {}
        self.enable_draw = enable_draw
        self.unum = unum  # Agent uniform number (player ID)
        self.init_pos = ([-14, 0], [-9, -5], [-9, 0], [-9, 5], [-5, -5], [-5, 0], [-5, 5], [-1, -6], [-1, -2.5], [-1, 2.5], [-1, 6])[unum-1]  # Initial formation
        self.player_formation = None
        #self.roles_dict = self.create_roles()  # Dictionary for roles

    # def create_roles(self):
    #     """
    #     Create a dictionary that maps agent uniform numbers to their respective roles.
    #     This uses the roles from the `Roles.py` module.
    #     """
    #     return {
    #         1: GoalKeeper(self.unum, self.world),
    #         2: LeftBack(self.unum, self.world),
    #         3: RightBack(self.unum, self.world),
    #         4: LeftCenterBack(self.unum, self.world),
    #         5: CenterAttackingMid(self.unum, self.world),
    #         6: Striker(self.unum, self.world),
    #         7: RightWing(self.unum, self.world),
    #         8: LeftWing(self.unum, self.world),
    #         9: RightCenterBack(self.unum, self.world)
    #     }
#----------------------------------- DYNAMIC ROLE ASSIGNMENT 

    def getPlayerRole(self):
        # Returns Role Object for 'unum' player
        return self.dynamic_role_assignment()[self.unum][2]

    def calculate_role_positions(self):
        '''
        Calculate the positions of all players/roles based on the current ball position.
        returns: numpy array of 11 positions (x,y) for 11 players
        '''
        w = self.world
        ball_2d = w.ball_abs_pos[:2]
        bx = ball_2d[0]
        by = ball_2d[1]

        # Initialise all 11 positions to (0,0,DummyRoleObject)
        role_positions = np.array([(0.0, 0.0, Striker(bx, by, w)) for _ in range(11)])

        # Set goalkeeper position (index 0) to (-14, 0)
        role_positions[0] = (-14.0, 0.0, GoalKeeper(-14.0, 0.0, w))

        # Set active player position (index 1) to the ball's position
        role_positions[1] = (bx, by, Striker(bx, by, w))

        # Basic 4-3-3 formation with Left Wing, Right Wing, Center Mid, etc.
        role_positions[10] = (bx, (10.0 + by)/2.0, 
        LeftWing(
            bx, # x co-ordinate
            (10.0 + by)/2.0, # y co-ordinate
            w) # world model
        )  # Left Wing

        role_positions[9] = (bx, (by + (-10.0))/2.0, RightWing(bx, (by + (-10.0))/2.0, w))  # Right Wing

        role_positions[8] = ((-15*1.0 + bx*2.0)/3.0, (by*2.0 + 10.0)/3.0, CenterAttackingMid((-15*1.0 + bx*2.0)/3.0, (by*2.0 + 10.0)/3.0, w))  # Center Mid

        role_positions[7] = ((-15*1.0 + bx*2.0)/3.0, by, LeftDefensiveMid((-15*1.0 + bx*2.0)/3.0, by, w))  # Left Center Mid

        role_positions[6] = ((-15*1.0 + bx*2.0)/3.0, (by*2.0 + (-10.0))/3.0, RightDefensiveMid((-15*1.0 + bx*2.0)/3.0, (by*2.0 + (-10.0))/3.0, w))  # Right Center Mid

        role_positions[5] = ((-15*2.0 + bx*1.0)/3.0, (by*3.0 + (-10.0)*2.0)/5.0, RightBack((-15*2.0 + bx*1.0)/3.0, (by*3.0 + (-10.0)*2.0)/5.0, w))  # Right Back

        role_positions[4] = ((-15*2.0 + bx*1.0)/3.0, (by*4.0 + (-10.0)*1.0)/5.0, RightCenterBack((-15*2.0 + bx*1.0)/3.0, (by*4.0 + (-10.0)*1.0)/5.0, w))  # Right Center Back

        role_positions[3] = ((-15*2.0 + bx*1.0)/3.0, (by*4.0 + (10.0)*1.0)/5.0, LeftCenterBack((-15*2.0 + bx*1.0)/3.0, (by*4.0 + (10.0)*1.0)/5.0, w))  # Left Center Back

        role_positions[2] = ((-15*2.0 + bx*1.0)/3.0, (by*3.0 + (10.0)*2.0)/5.0, LeftBack((-15*2.0 + bx*1.0)/3.0, (by*3.0 + (10.0)*2.0)/5.0, w))  # Left Back

        # Return a numpy array consisting of coordinates of all 11 players
        return role_positions

    def dynamic_role_assignment(self):
        '''
        Assign roles to the inactive players based on the current ball position.
        return : dictionary mapping agent_unum to their assigned positions
        '''
        w = self.world  # Access the world object

        # Get the uniform numbers of the spawned players
        agents_unums = [p.unum for p in w.teammates]
        agents_unums = sorted(agents_unums)

        # Calculate ideal positions for each player
        role_positions = self.calculate_role_positions()

        # Create a dictionary mapping agent_unum to their current positions
        agent_current_positions = {
            p.unum: p.state_abs_pos[:2] if p.state_last_update != 0 and 
            (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and 
            not p.state_fallen 
            else np.array([1000, 1000])  # Large distance if the agent is invalid, not updated, or has fallen
            for p in w.teammates
        }

        # Initialize bestRoleMap to store optimal assignments for subsets
        n = len(agents_unums)

        self.bestRoleMap = {frozenset({1}): ({1:role_positions[0]}, 0)}  # (Mapping, Cost) for empty set
        
        if n == 1:
            return {agents_unums[0]: role_positions[0]}



        for k in range(2, n + 1):
            pk = role_positions[k - 1]

            for a in agents_unums:
                remaining_agents = [agent for agent in agents_unums if agent != a]
                subsets = list(itertools.combinations(remaining_agents, k - 1))

                for subset in subsets:
                    subset_set = frozenset(subset)
                    m0, m0_cost = self.bestRoleMap.get(subset_set, (None, float('inf')))

                    new_mapping = {a: pk}
                    if m0:
                        new_mapping.update(m0)

                    cost = self.calculate_cost(new_mapping, agent_current_positions)
                    if (subset_set | frozenset({a})) not in self.bestRoleMap or cost < self.bestRoleMap[subset_set | frozenset({a})][1]:
                        self.bestRoleMap[subset_set | frozenset({a})] = (new_mapping, cost)

        return self.bestRoleMap[frozenset(agents_unums)][0]

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
            target_position = target_position[:2:]
            agent_current_position = current_positions[agent_unum]  # This assumes current_positions is a dictionary with agent_unum as keys

            # Calculate the Euclidean distance between the agent's current position and the target position
            distance = np.linalg.norm(np.array(agent_current_position) - np.array(target_position))

            # Add the distance to the list
            distances.append(distance)

        # Sort the distances in decreasing order to form the lexicographical tuple
        sorted_distances = tuple(sorted(distances, reverse=True))

        return sorted_distances