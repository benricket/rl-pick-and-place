"""
Arm model templates for the FunRobo kinematics library.

Each class provides a common interface for:
- Forward kinematics
- Inverse kinematics (analytical or numerical)
- Velocity kinematics
- Computing joint and end-effector positions for visualization

"""

import math
import numpy as np
from typing import List, Tuple
import utils as ut

class BaseRobot():
    """
    Template model for a generic robotic arm. It serves as a parent class for all subsequent
    robot arm models.

    Attributes:
        joint_values (list[float]): Current joint angles [theta1, theta2, ...] in radians.
        joint_limits (list[list[float]]): Joint angle limits [[min, max], ...] in radians.
        joint_vel_limits (list[list[float]]): Joint velocity limits (rad/s).
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (default is 1).
        points (list[np.ndarray]): Homogeneous coordinates of joint positions.
    """


    def __init__(self) -> None:
        """
        Initialize the generic robot model with default geometry and joint limits.
        """
        self.joint_values = []
        self.joint_limits = []  # Joint limits
        self.joint_vel_limits = []

        self.ee = ut.EndEffector()  # The end-effector object
        self.num_dof = 1  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points


    def calc_forward_kinematics(
        self, joint_values: List[float], radians: bool = True
    ) -> Tuple[ut.EndEffector, List[np.ndarray]]:
        """
        Compute the forward kinematics of the robot.

        Given a set of joint angles, this method computes the pose of the
        end-effector and the corresponding homogeneous transformation matrices.

        Args:
            joint_values (list[float]): Joint angles [theta1, theta2].
            radians (bool, optional): If False, joint angles are assumed to be
                in degrees and will be converted to radians. Defaults to True.

        Returns:
            tuple:
                - EndEffector: End-effector pose container (position/orientation).
                - list[np.ndarray]: List of 4x4 individual link transforms (length = num_dof).

        """
        ee = ut.EndEffector()
        Hlist = [np.eye(4,4)] * self.num_dof
        return ee, Hlist


    def calc_inverse_kinematics(
        self, ee: ut.EndEffector, joint_values: List[float], soln: int = 0
    ) -> List[float]:
        """
        Compute an analytical inverse kinematics solution.

        Given a desired end-effector pose, this method computes a set of joint
        angles that achieve the pose, if a valid solution exists.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (List[float]): Initial or previous joint angles, used
                for solution selection or continuity.
            soln (int, optional): Solution branch index (e.g., elbow-up vs
                elbow-down). Defaults to 0.

        Returns:
            list[float]: Joint angles [theta1, theta2] in radians that achieve
            the desired end-effector pose.
        """
        new_joint_values = joint_values.copy()
        return new_joint_values


    def calc_numerical_ik(
        self, ee: ut.EndEffector, joint_values: List[float], tol: float = 0.01, ilimit: int = 100
    ) -> List[float]:
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles.
            tol (float, optional): Convergence tolerance on pose/position error. Defaults to 0.01.
            ilimit (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            list[float]: Estimated joint angles in radians.
        """
        new_joint_values = joint_values.copy()
        return new_joint_values


    def calc_velocity_kinematics(
        self, joint_values: List[float], vel: List[float], dt: float = 0.02
    ) -> List[float]:
        """
        Update joint angles based on a desired end-effector velocity.

        This method maps a desired Cartesian end-effector velocity into joint
        space and integrates the result over a single time step.

        Args:
            joint_values (List[float]): Current joint angles in radians.
            vel (List[float]): Desired end-effector linear velocity [vx, vy].
            dt (float, optional): Integration time step in seconds.
                Defaults to 0.02.

        Returns:
            List[float]: Updated joint angles in radians after one time step.
        """
        new_joint_values = joint_values.copy()
        return new_joint_values


    def calc_robot_points(self, joint_values: List[float], Hlist: List[np.ndarray], radians: bool = True) -> None:
        """
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transformation matrices**
        to compute cumulative transforms and determine the positions of all
        joints and the end effector in the base frame.

        It updates internal state in-place:
            - `self.points`: base/joint/EE points (homogeneous coordinates)
            - `self.ee`: end-effector position and orientation
            - `self.H_ee`: cumulative base->EE transform
            - `self.EE_axes`: end-effector axis endpoints for visualization

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            H (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4),
                where H[i] is the transform contributed by joint i alone. If None, zeros are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None
        """
        return None

class KinovaRobotTemplate(BaseRobot):
    """
    Template model of the 6-DOF Kinova robot arm.

    Attributes:
        l1, l2, l3, l4, l5, l6 (float): Link lengths (meters).
        joint_values (list[float]): Current joint angles (radians).
        joint_limits (list[list[float]]): Joint angle limits (radians).
        joint_vel_limits (list[list[float]]): Joint velocity limits (rad/s).
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (5).
        points (list[np.ndarray | None]): Joint positions (base + joints + EE).
        EE_axes (np.ndarray | None): End-effector axis endpoints for visualization.
        H_ee (np.ndarray | None): Cumulative end-effector transform (base -> EE).
    """

    
    def __init__(self) -> None:
        """
        Initialize the 6-DOF robot model with default geometry and joint limits.
        """
        super().__init__()

        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7 = 0.156, 0.128, 0.410, 0.208, 0.105, 0.105, 0.0615
        
        self.joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Joint limits (in radians)
        self.joint_limits = [
            [-2 * np.pi, 2 * np.pi],
            [-0.71 * np.pi, 0.71 * np.pi],
            [-0.82 * np.pi, 0.82 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-0.66 * np.pi, 0.66 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]

        self.joint_vel_limits = [
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
        ]
        
        self.ee = ut.EndEffector()
        self.num_dof = 6
        self.points = [None] * (self.num_dof + 2)


    def calc_robot_points(
            self, joint_values: List[float], Hlist: List[np.ndarray], radians: bool = True
    ) -> None:
        """ 
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transforms** to compute cumulative
        transforms and determine the positions of all joints and the end effector in the
        base frame.

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            Hlist (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4).
                If None, zero matrices are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None: This method updates internal state. 
        """

        if not radians: # Convert degrees to radians if the input is in degrees
            joint_values = [np.deg2rad(theta) for theta in joint_values]

        self.joint_values = joint_values.copy()

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, len(self.points)):
            self.points[i] = H_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = H_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(self.H_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE_axes = np.array([self.H_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])