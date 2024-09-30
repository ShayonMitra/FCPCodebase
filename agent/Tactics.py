from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import itertools

class Tactics(Base_Agent):

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
