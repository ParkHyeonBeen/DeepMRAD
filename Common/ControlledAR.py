from Common.Utils import *
from gym.envs.mujoco.quad_rate import *

args = hyperparameters()

class ControlledAR():
    def __init__(self, env_sim, env_real, args_test):

        # setting
        self.args_test = args_test

        if self.args_test.render is True:
            self.error_plot = False
        else:
            self.error_plot = True

        self.error_plot = False
        self.do_disturb = True

        # environment
        self.env_sim = env_sim
        self.env_real = env_real
        self.timestep = self.env_real.timestep

        self.frame_skip = args.frame_skip
        self.inner_skip = args.inner_skip

    def state_converter(self, ob, dim=9):

        state = np.zeros(dim)

        state[0:3] = ob[0:3]
        state[3:6] = quat2rpy(ob[3:7])
        state[6:9] = ob[7:10]

        return state

    def step(self, input, skip_pid = 5):

        self.env_sim.local_step += 1
        self.env_real.local_step += 1

        if args.goal_make == True:
            self.env_real.goal[0:3] = self.env_real.make_trajectory(args.goal_shape)
            self.env_sim.goal[0:3] = self.env_sim.make_trajectory(args.goal_shape)

        if args.mode_test == True:
            action = self.env_real.PID(self.env_real.goal[0:3])
        else:
            if args.mode_cmd == 'velocity':
                action = denomalize(input, self.env_real.act_max, self.env_real.act_min)
                action = np.clip(action, a_min=self.env_real.act_min, a_max=self.env_real.act_max)
            elif args.mode_cmd == 'position':
                action = self.env_real.PID(0.5 * input[:3])
            else:
                raise Exception(" check your test mode ")

        action_init = denomalize(input, self.env_real.act_max, self.env_real.act_min)

        action = action_init
        if self.error_plot is True:
            print("start")

        ob_real = self.env_real._get_obs()
        qpos_real = ob_real[:7]
        qvel_real = ob_real[7:]
        self.env_sim.set_state(qpos_real, qvel_real)

        print("start")
        for i in range(self.frame_skip//skip_pid):
            if self.do_disturb is True and i == 0:
                action, T_dist, R_dist = self.env_real.add_disturbance(action, showme=True)
                print("mag_dist : ", T_dist, R_dist)
            self.env_real.do_simulation(action, skip_pid)
            self.env_sim.do_simulation(action_init, skip_pid)

            ob_ref = self.env_sim._get_obs()
            next_state = self.state_converter(ob_ref)
            # next_state = self.make_nextstate_ref(ob_ref, action_init)
            action_ctrl = self.PID_car(next_state)
            print(action_ctrl)
            action_next = action_init + action_ctrl
            # action_next = np.clip(action_next, a_min=self.env_real.act_min, a_max=self.env_real.act_max)
            action = action_next

        ob = self.env_real._get_obs()
        put_path(ob)

        if args.convert_relative == True:
            ob[0:3] = ob[0:3] - self.env_real.goal[:3]

            pos = ob[0:3]
            ang = quat2rpy(ob[3:7])
            self.env_real.error_pos = linalg.norm(pos)
            self.env_real.error_ang = linalg.norm(self.env_real.goal[3:] - ang)
        else:
            pos = ob[0:3]
            ang = quat2rpy(ob[3:7])
            self.env_real.error_pos = linalg.norm(self.env_real.goal[:3] - pos)
            self.env_real.error_ang = linalg.norm(self.env_real.goal[3:] - ang)

        ob = np.append(ob, [self.env_real.error_pos, self.env_real.error_ang])

        reward, done, info = self.env_real._get_reward(ob, action)

        if self.env_real.local_step == 8000:
            done = True
        if done:
            self.env_real.local_step = 0

        return ob, reward, done, info

    def PID_car(self, ref):

        K_x = np.array([-10., -10., 10.])
        K_y = np.array([10., 10., 10.])
        K_z = np.array([1e3, 1e3])

        ob = self.env_real._get_obs()
        state = self.state_converter(ob)

        error = state - ref
        put_data(error)
        if self.error_plot is True:
            print("error :", error)
            plot_data(error, label=['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'x_rate', 'y_ra te', 'z_rate'])

        error_x = np.array([error[0],
                            error[4],
                            error[6]],
                           dtype="object").T
        error_y = np.array([error[1],
                            error[3],
                            error[7]],
                           dtype="object").T
        error_z = np.array([error[2],
                            error[8]],
                           dtype="object").T
        error_yaw = np.array(error[5], dtype="object")

        thrust = -K_z @ error_z
        roll_rate = -K_x @ error_y
        pitch_rate = -K_y @ error_x
        yaw_rate = -1.0 * error_yaw

        action = np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype="object")
        # act_min=np.array([-10., -(1/12)*math.pi,-(1/12)*math.pi, -0.03])
        # act_max=np.array([10.,(1/12)*math.pi,(1/12)*math.pi, 0.03])
        # action = np.clip(action, a_min=act_min, a_max=act_max)

        if self.error_plot is True:
            print("action :", action)

        return action

    def make_nextstate_ref(self, state, action):

        timestep = self.inner_skip * self.timestep

        pos = state[0:3]
        ang = state[3:6]
        pos_vel = state[6:9]
        ang_vel = action[1:4]
        thrust = np.array([0, 0, action[0]], dtype="object").T
        gravity = np.array([0., 0., 9.81]).T

        R_roll = np.array([[1, 0, 0],
                           [0, math.cos(ang[0]), -math.sin(ang[0])],
                           [0, math.sin(ang[0]), math.cos(ang[0])]])

        R_pitch = np.array([[math.cos(ang[1]), 0, math.sin(ang[1])],
                            [0, 1, 0],
                            [-math.sin(ang[1]), 0, math.cos(ang[1])]])

        R_yaw = np.array([[math.cos(ang[2]), -math.sin(ang[2]), 0.], \
                          [math.sin(ang[2]), math.cos(ang[2]), 0.], \
                          [0., 0., 1.]])

        R = R_yaw @ R_pitch @ R_roll

        # pos_acc = ((pos_vel - prev_pose_vel)/timestep ).T
        pos_acc = (np.linalg.inv(R) @ thrust) / (self.env_real.get_mass()[0]) - gravity
        pos_acc = pos_acc.T

        next_pos_vel = pos_vel + timestep * pos_acc
        next_pos = pos + timestep * pos_vel
        next_ang = ang + timestep * ang_vel

        next_state_ref = np.hstack((next_pos, next_ang, next_pos_vel))

        return next_state_ref