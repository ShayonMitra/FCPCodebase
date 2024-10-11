from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np
import itertools
import agent.Roles

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
        self.roles_dict = self.create_roles()  # Dictionary for roles

    def create_roles(self):
        """
        Create a dictionary that maps agent uniform numbers to their respective roles.
        This uses the roles from the `Roles.py` module.
        """
        return {
            1: agent.Roles.GoalKeeper(self.unum, self.world),
            2: agent.Roles.LeftBack(self.unum, self.world),
            3: agent.Roles.RightBack(self.unum, self.world),
            4: agent.Roles.MidBack(self.unum, self.world),
            5: agent.Roles.MidForward(self.unum, self.world),
            6: agent.Roles.Striker(self.unum, self.world),
            7: agent.Roles.RightWing(self.unum, self.world),
            8: agent.Roles.LeftWing(self.unum, self.world),
            9: agent.Roles.MidForward(self.unum, self.world)
        }

    def calculate_player_positions(self):
        '''
        Calculate the positions of all players/roles based on the current ball position.
        Returns: numpy array of 9 positions (x, y) for 9 players
        '''
        w = self.world
        ball_2d = w.ball_abs_pos[:2]

        # Initialize all 9 positions to (0,0) 
        player_positions = np.array([(0.0, 0.0) for _ in range(9)])

        # x and y coordinates of the ball
        bx = ball_2d[0]
        by = ball_2d[1]

        # Basic 4-3-3 formation with roles Left Wing, Right Wing, Center Mid... and so on
        player_positions[8] = (bx, (10.0 + by) / 2.0)  # Left Wing
        player_positions[7] = (bx, (by - 10.0) / 2.0)  # Right Wing
        player_positions[6] = ((-15 + 2 * bx) / 3.0, (2 * by + 10.0) / 3.0)  # Center Mid
        player_positions[5] = ((-15 + 2 * bx) / 3.0, by)  # Left Center Mid
        player_positions[4] = ((-15 + 2 * bx) / 3.0, (2 * by - 10.0) / 3.0)  # Right Center Mid
        player_positions[3] = ((-30 + bx) / 3.0, (3 * by - 20.0) / 5.0)  # Right Back
        player_positions[2] = ((-30 + bx) / 3.0, (4 * by - 10.0) / 5.0)  # Right Center Back
        player_positions[1] = ((-30 + bx) / 3.0, (4 * by + 10.0) / 5.0)  # Left Center Back
        player_positions[0] = ((-30 + bx) / 3.0, (3 * by + 20.0) / 5.0)  # Left Back

        return player_positions

    def dynamic_role_assignment(self, active_player_unum):
        '''
        Assign roles to the inactive players based on the current ball position.
        Return: dictionary mapping agent_unum to their assigned positions
        '''
        w = self.world  # Access the world object

        if active_player_unum == 1:
            active_player_unum = np.random.randint(2, 12)

        agents_unums = [_ for _ in range(2, 12)]
        try:
            agents_unums.remove(active_player_unum)
        except:
            print(active_player_unum)

        player_positions = self.player_formation  # Player formation stored in class

        agent_current_positions = {
            p.unum: p.state_abs_pos[:2] if p.state_last_update != 0 and 
            (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and 
            not p.state_fallen 
            else np.array([1000, 1000])
            for p in w.teammates if p.unum in agents_unums
        }

        # Initialize bestRoleMap for role assignment optimization
        n = len(agents_unums)
        self.bestRoleMap = {frozenset(): (None, 0)}  # (Mapping, Cost) for empty set

        for k in range(1, n + 1):
            pk = player_positions[k - 1]

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
        :return: A lexicographically (reverse) sorted tuple of distances.
        """
        distances = []

        for agent_unum, target_position in mapping.items():
            agent_current_position = current_positions[agent_unum]

            distance = np.linalg.norm(np.array(agent_current_position) - np.array(target_position))
            distances.append(distance)

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
        if unum in self.previous_positions:
            prev_x, prev_y = self.previous_positions[unum]

            smoothed_x = alpha * new_position[0] + (1 - alpha) * prev_x
            smoothed_y = alpha * new_position[1] + (1 - alpha) * prev_y
        else:
            smoothed_x, smoothed_y = new_position

        self.previous_positions[unum] = (smoothed_x, smoothed_y)

        return smoothed_x, smoothed_y
