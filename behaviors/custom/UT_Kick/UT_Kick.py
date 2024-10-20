from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.Math_Ops import Math_Ops as M


class UT_Kick():

    def __init__(self, base_agent : Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "Walk to ball and perform a basic kick"
        self.auto_head = True

        r_type = self.world.robot.type
        self.bias_dir = [22,29,26,29,22][self.world.robot.type]
        self.ball_x_limits = ((0.19,0.215), (0.2,0.22), (0.19,0.22), (0.2,0.215), (0.2,0.215))[r_type]
        self.ball_y_limits = ((-0.115,-0.1), (-0.125,-0.095), (-0.12,-0.1), (-0.13,-0.105), (-0.09,-0.06))[r_type]
        self.ball_x_center = (self.ball_x_limits[0] + self.ball_x_limits[1])/2
        self.ball_y_center = (self.ball_y_limits[0] + self.ball_y_limits[1])/2
      
    def execute(self,reset, direction, abort=False) -> bool: # You can add more arguments 
        '''
        Parameters
        ----------
        direction : float
            kick direction relative to field, in degrees
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        '''

        # if reset:
        #     self.phase = 0
        #     self.reset_time = self.world.time_local_ms

        # if(self.world.robot.slot%10 == 0):
        #     print(f"Coords at slot {self.world.robot.slot}: ", self.world.ball_rel_torso_cart_pos)
        
        # if self.phase == 0:
        #     self.phase += 1
        #     print("First Kick Motion: ", self.world.robot.slot)
        #     print("Initial Coords: ", self.world.ball_rel_torso_cart_pos)
        #     return self.behavior.execute_sub_behavior("Kick_Motion", True)
        # else:
        #     if(self.world.robot.slot == 169):
        #         print("Final Coords: ", self.world.ball_rel_torso_cart_pos)
        #     return self.behavior.execute_sub_behavior("Kick_Motion", False)

        XOFFSET = -0.1563029484732077
        YOFFSET = +0.03557468546273009

        w = self.world
        r = self.world.robot
        b = w.ball_rel_torso_cart_pos
        ball_abs = w.ball_abs_pos
        t = w.time_local_ms
        # gait : Step_Generator = self.behavior.get_custom_behavior_object("Walk").env.step_generator

        if reset:
            self.phase = 0
            self.reset_time = t

        if self.phase == 0: 
            biased_dir = M.normalize_deg(direction + self.bias_dir) # add bias to rectify direction
            # set b to offsetted position. b is x,y,z relative to torso. multiply offset with sin or cos as needed
            # angle is between -180 and 180

            target = ball_abs[:2]
            target[0] += XOFFSET * M.deg_cos(biased_dir) - YOFFSET * M.deg_sin(biased_dir)
            target[1] += XOFFSET * M.deg_sin(biased_dir) + YOFFSET * M.deg_cos(biased_dir)

            ang_diff = abs(M.normalize_deg( biased_dir - r.loc_torso_orientation )) # the reset was learned with loc, not IMU

            # next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
            #     x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)


            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_target(target, torso_ori=biased_dir)
            if(self.world.robot.slot%10 == 1):
                print("Slot: ", self.world.robot.slot, "Target: ", target, "Next Pos: ", next_pos, "Next Ori: ", next_ori, "Dist: ", dist_to_final_target)

            if (dist_to_final_target < 0.03 ):

                self.phase += 1

                print("First Kick Motion: ", self.world.robot.type)
                return self.behavior.execute_sub_behavior("Kick_Motion_15", True)

            
            # if (w.ball_last_seen > t - w.VISUALSTEP_MS and ang_diff < 5 and       # ball is visible & aligned
            #     self.ball_x_limits[0] < b[0] < self.ball_x_limits[1] and          # ball is in kick area (x)
            #     self.ball_y_limits[0] < b[1] < self.ball_y_limits[1] and          # ball is in kick area (y)
            #     t - w.ball_abs_pos_last_update < 100 and                          # ball absolute location is recent
            #     dist_to_final_target < 0.03 and                                   # if absolute ball position is updated
            #     not gait.state_is_left_active and gait.state_current_ts == 2 and  # walk gait phase is adequate
            #     t - self.reset_time > 500): # to avoid kicking immediately without preparation & stability

            #     self.phase += 1

            #     print("First Kick Motion: ", self.world.robot.type)
            #     return self.behavior.execute_sub_behavior("Kick_Motion", True)
            else:
                dist = max(0.07, dist_to_final_target)
                reset_walk = reset and self.behavior.previous_behavior != "Walk" # reset walk if it wasn't the previous behavior
                self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist) # target, is_target_abs, ori, is_ori_abs, distance
                return abort # abort only if self.phase == 0

        else: # define kick parameters and execute 
            return self.behavior.execute_sub_behavior("Kick_Motion_15", False)

      
    def is_ready(self) -> any: # You can add more arguments 
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True
