""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import time
from enum import Enum, auto
from typing import Dict, NamedTuple

import numpy as np
import torch
import torchcontrol as toco
from polymetis import RobotInterface


class HybridJointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
    """

    def __init__(
        self,
        joint_pos_current,
        kq,
        kqd,
        kx,
        kxd,
        robot_model: torch.nn.Module,
        ki=None,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(kq, kqd, kx, kxd)
        self.ki = ki

        self.kp = torch.nn.Parameter(torch.tensor(0.0))
        self.kd = torch.nn.Parameter(torch.tensor(0.0))
        # Reference pose
        self.q_desired = torch.nn.Parameter(torch.tensor(joint_pos_current))
        self.qd_desired = torch.nn.Parameter(torch.zeros_like(joint_pos_current))

        self.integral_error = torch.zeros_like(joint_pos_current)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        # print(state_dict)
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        if self.ki is not None:
            error = self.q_desired - joint_pos_current
            error_integral = (
                self.integral_error + error
            )  # Accumulate error for integral term
            self.integral_error = error_integral  # Update integral error

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.q_desired,
            self.qd_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward
        if self.ki is not None:
            torque_out += self.ki @ error_integral

        return {"joint_torques": torque_out}


class JointPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, joint_pos_current, kp, kd, **kwargs):
        """
        Args:
            desired_joint_pos (int):    Number of steps policy should execute
            hz (double):                Frequency of controller
            kp, kd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.kp = torch.nn.Parameter(kp)
        self.kd = torch.nn.Parameter(kd)
        self.q_desired = torch.nn.Parameter(joint_pos_current)
        self.qd_desired = torch.nn.Parameter(torch.zeros_like(joint_pos_current))

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(self.kp, self.kd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]
        self.feedback.Kp = torch.diag(self.kp)
        self.feedback.Kd = torch.diag(self.kd)

        # Execute PD control
        output = self.feedback(q_current, qd_current, self.q_desired, self.qd_desired)
        return {"joint_torques": output}


class HumanControl(toco.PolicyModule):
    # stolen from SimulationFramework
    def __init__(self, robot: RobotInterface, regularize=True):
        super().__init__()

        # get joint limits for regularization
        limits = robot.robot_model.get_joint_angle_limits()
        self.joint_pos_min = limits[0]
        self.joint_pos_max = limits[1]

        # define gain
        self.gain = torch.Tensor([0.3, 0.12, 0.40, 1.11, 1.10, 0.6, 0.85])

        if regularize:
            self.reg_gain = torch.Tensor([5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0])
        else:
            self.reg_gain = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ext = state_dict["motor_torques_external"]

        human_torque = -self.gain * ext
        # reg_torgue = torch.zeros_like(ext)

        joint_pos_current = state_dict["joint_positions"]

        left_boundary = 1 / torch.clamp(
            torch.abs(self.joint_pos_min - joint_pos_current), 1e-8, 100000
        )
        right_boundary = 1 / torch.clamp(
            torch.abs(self.joint_pos_max - joint_pos_current), 1e-8, 100000
        )

        reg_load = left_boundary - right_boundary

        reg_torgue = self.reg_gain * reg_load

        return {"joint_torques": human_torque + reg_torgue}


class ImitationControl(HybridJointImpedanceControl):
    def __init__(self, robot: RobotInterface):
        super().__init__(
            joint_pos_current=robot.get_joint_positions(),
            kq=robot.Kq_default,
            kqd=robot.Kqd_default,
            kx=robot.Kx_default,
            kxd=robot.Kxd_default,
            robot_model=robot.robot_model,
            ignore_gravity=robot.use_grav_comp,
        )
        # super().__init__(
        #     joint_pos_current=robot.get_joint_positions(),
        #     kq=robot.Kq_default*0.85,
        #     kqd=robot.Kqd_default*0.5,
        #     kx=robot.Kx_default*0.35,
        #     kxd=robot.Kxd_default*0.25,
        #     ki=robot.Kq_default*0.001,
        #     robot_model=robot.robot_model,
        #     ignore_gravity=robot.use_grav_comp,
        # )


class ControlType(Enum):
    HUMAN_CONTROL = auto()
    IMITATION_CONTROL = auto()
    HYBRID_JOINT_IMPEDANCE_CONTROL = auto()
    CARTESIAN_IMPEDANCE_CONTROL = auto()
    DEFAULT = auto()


class ArmState(NamedTuple):
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    ee_pos: torch.Tensor
    ee_vel: torch.Tensor


class FrankaArm:
    def __init__(
        self,
        name,
        ip_address,
        control_type: ControlType,
        port=50051,
        gain_scale=1.0,
        reset_gain_scale=1.0,
        default_reset_pose=[-0.1231, -0.0156, 0.1160, -2.2044, -0.0125, 2.1907, 0.7891],
        hz=6,
        **kwargs
    ):
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.robot = None
        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale
        self.default_reset_pose = default_reset_pose
        self.hz = hz
        self.velocity_limits = np.array([[-4 * np.pi / 2, 4 * np.pi / 2]] * 7).T / 32
        self.velocity_limits_norm = np.linalg.norm(self.velocity_limits)
        self.control_type = control_type

    def set_policy(self):
        if self.control_type == ControlType.HUMAN_CONTROL:
            self.policy = HumanControl(self.robot)
        elif self.control_type == ControlType.IMITATION_CONTROL:
            self.policy = ImitationControl(self.robot)
        elif self.control_type == ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL:
            self.policy = HybridJointImpedanceControl(
                joint_pos_current=self.robot.get_joint_positions(),
                kq=self.robot.Kq_default * 0.85,
                kqd=self.robot.Kqd_default * 0.5,
                kx=self.robot.Kx_default * 0.35,
                kxd=self.robot.Kxd_default * 0.25,
                ki=self.robot.Kq_default * 0.001,
                robot_model=self.robot.robot_model,
                ignore_gravity=self.robot.use_grav_comp,
            )
        elif self.control_type == ControlType.CARTESIAN_IMPEDANCE_CONTROL:
            self.policy = toco.policies.impedance.CartesianImpedanceControl(
                joint_pos_current=self.robot.get_joint_positions(),
                Kp=self.robot.Kx_default,
                Kd=self.robot.Kxd_default,
                robot_model=self.robot.robot_model,
                ignore_gravity=self.robot.use_grav_comp,
            )

        if not self.control_type == ControlType.DEFAULT:
            self.robot.send_torch_policy(self.policy, blocking=False)

    def connect(self, home_pose=None):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = RobotInterface(
                ip_address=self.ip_address, enforce_version=False, port=self.port
            )
            print("Success")
        except Exception as e:
            self.robot = None  # declare dead
            print("Failed with exception: ", e)
            return connection

        print("Testing {} connection: ".format(self.name), end="")
        connection = self.okay()
        if connection:
            print("okay")
            if self.default_reset_pose:
                self.robot.set_home_pose(torch.Tensor(self.default_reset_pose))

            self.reset()  # reset the robot before starting operaions
            self.set_policy()
        else:
            print("Not ready. Please retry connection")

        return connection

    def okay(self):
        """Return hardware health"""
        okay = False
        if self.robot:
            try:
                state = self.robot.get_robot_state()
                delay = time.time() - (
                    state.timestamp.seconds + 1e-9 * state.timestamp.nanos
                )
                assert delay < 5, "Acquired state is stale by {} seconds".format(delay)
                okay = True
            except:
                self.robot = None  # declare dead
                okay = False
        return okay

    def close(self):
        """Close hardware connection"""
        if self.robot:
            print("Terminating PD policy: ", end="")
            try:
                self.reset()
                state_log = self.robot.terminate_current_policy()
                print("Success")
            except:
                # print("Failed. Resetting directly to home: ", end="")
                print("Resetting Failed. Exiting: ", end="")
            self.robot = None
            print("Done")
        return True

    def reconnect(self):
        print("Attempting re-connection")
        self.connect()
        while not self.okay():
            self.connect()
            time.sleep(2)
        print("Re-connection success")

    def reset(self, reset_pos=None, time_to_go=2):
        """Reset hardware"""

        if self.okay():
            # Use default controller
            print("Resetting using default controller")
            self.robot.go_home(time_to_go=time_to_go)
            self.set_policy()
        else:
            print(
                "Can't connect to the robot for reset. Attemping reconnection and trying again"
            )
            self.reconnect()
            self.reset(reset_pos, time_to_go)

    def get_state(self) -> ArmState:
        try:
            joint_pos = self.robot.get_joint_positions()
            joint_vel = self.robot.get_joint_velocities()
            ee_pos = torch.cat(self.robot.get_ee_pose())
        except:
            print("Failed to get current sensors: ", end="")
            self.reconnect()
            return self.get_state()

        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)
        ee_vel = jacobian @ joint_vel

        return ArmState(
            joint_pos=joint_pos, joint_vel=joint_vel, ee_pos=ee_pos, ee_vel=ee_vel
        )

    def apply_commands(self, q_desired=None, qd_desired=None, kp=None, kd=None):
        """Apply hardware commands"""
        udpate_pkt = {}
        if q_desired is not None:
            udpate_pkt["q_desired"] = (
                q_desired if torch.is_tensor(q_desired) else torch.tensor(q_desired)
            )
        if qd_desired is not None:
            udpate_pkt["qd_desired"] = (
                qd_desired if torch.is_tensor(qd_desired) else torch.tensor(qd_desired)
            )
        if kp is not None:
            udpate_pkt["kp"] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt["kd"] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        try:
            self.robot.update_current_policy(udpate_pkt)
        except Exception as e:
            print("1> Failed to udpate policy with exception", e)
            self.reconnect()

    def generate_waypoints_within_limits(
        self, start, goal, hz, max_vel_norm=float("inf")
    ):
        step_duration = 1 / hz
        vel = (goal - start) / step_duration
        vel_norm = np.linalg.norm(vel)

        if vel_norm > max_vel_norm:
            feasible_vel = (vel / vel_norm) * max_vel_norm
        else:
            feasible_vel = vel

        feasible_norm = np.linalg.norm(feasible_vel)

        n_steps = int(np.ceil(vel_norm / feasible_norm))

        t = torch.linspace(0, 1, n_steps + 1)[1:]
        waypoints = (1 - t[:, None]) * start + t[:, None] * goal

        return waypoints, feasible_vel

    def go_to_within_limits(self, goal):
        goal = torch.tensor(goal)
        q_initial = self.robot.get_joint_positions().detach().cpu()
        waypoints, feasible_vel = self.generate_waypoints_within_limits(
            q_initial, goal, self.hz, max_vel_norm=self.velocity_limits_norm
        )
        print(len(waypoints))
        dwaypoints = torch.diff(waypoints, dim=0)
        for i in range(len(waypoints)):
            # self.apply_commands(q_desired=waypoints[i], qd_desired=dwaypoints[i] if i < len(dwaypoints) else torch.zeros_like(feasible_vel))
            self.apply_commands(
                q_desired=waypoints[i],
                qd_desired=(
                    feasible_vel
                    if i < len(dwaypoints)
                    else torch.zeros_like(feasible_vel)
                ),
            )
            time.sleep(1 / self.hz)

    def __del__(self):
        self.close()


if __name__ == "__main__":

    # user inputs
    time_to_go = 20 * np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    franka = FrankaArm(name="Franka-Demo", ip_address="141.3.53.154")

    # connect to robot with default policy
    assert franka.connect(control_type=None), "Connection to robot failed."

    # reset using the user controller
    franka.reset()

    Q1 = torch.tensor(
        [
            -1.3821e-01,
            1.9691e-03,
            -5.3979e-02,
            -2.0517e00,
            6.1574e-02,
            1.9851e00,
            -9.0278e-01,
        ]
    )
    Q2 = torch.tensor([0.3845, 0.1215, 0.3458, -0.7015, 0.3703, 1.9355, -0.8570])

    action_hz = 1
    actions = [
        Q1,
        Q2,
        Q1,
        Q2 * 0.5 + Q1 * 0.5,
        Q1,
        Q2 * 0.25 + Q1 * 0.75,
        Q1,
        Q1 + 0.1 * torch.randn(Q1.size()),
        Q1 + 0.1 * torch.randn(Q1.size()),
        Q1 + 0.1 * torch.randn(Q1.size()),
    ]

    for action in actions:
        curr_time = time.time()
        franka.go_to_within_limits(action)
        elapsed = time.time() - curr_time
        slack = 1 / action_hz - elapsed
        print(
            "Time taken: ",
            elapsed,
            "Slack: ",
            slack,
            "Action duration: ",
            1 / action_hz,
        )
        time.sleep(max(0, slack))

    # # Update policy to execute a sine trajectory on joint 6 for 5 seconds
    # print("Starting sine motion updates...")
    # s_initial = franka.get_sensors()
    # q_initial = s_initial['joint_pos'].clone()
    # q_desired = s_initial['joint_pos'].clone()

    # for i in range(int(time_to_go * hz)):
    #     q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
    #     # q_desired[5] = q_initial[5] + 0.05*np.random.uniform(high=1, low=-1)
    #     # q_desired = q_initial + 0.01*np.random.uniform(high=1, low=-1, size=7)
    #     franka.apply_commands(q_desired = q_desired)
    #     time.sleep(1 / hz)

    # # Udpate the gains
    # kp_new = 0.1* torch.Tensor(franka.robot.metadata.default_Kq)
    # kd_new = 0.1* torch.Tensor(franka.robot.metadata.default_Kqd)
    # franka.apply_commands(kp=kp_new, kd=kd_new)

    # print("Starting sine motion updates again with updated gains.")
    # for i in range(int(time_to_go * hz)):
    #     q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
    #     franka.apply_commands(q_desired = q_desired)
    #     time.sleep(1 / hz)

    print("Closing and exiting hardware connection")
    franka.close()
