"""rl_environment.xml helpers + minimal gripper oscillation viewer."""

import math
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

# This is the helper function to map gripper to one input
from gripper_kinova2f import gripper_ctrl_from_ratio

SCRIPT_DIR = Path(__file__).resolve().parent
XML_PATH = (SCRIPT_DIR.parent / "rl_environment.xml").resolve()

SPEED = 1.2

# list of finger joints, names used in the urdf file
_FINGER = ("RIGHT_BOTTOM", "RIGHT_TIP", "LEFT_BOTTOM", "LEFT_TIP")

def get_joint_limits():
    """
    Returns joint limits for the main joints
    """
    low = np.array([-2.69,-2.69,-2.69,-2.59,-2.57,-2.59])
    high = low * -1
    return low,high

def get_joints(m, d):
    """
    Current positions for arm joints J0 - J5.
    
    Args: 
        m: mujoco model
        d: mujoco data

    Returns:
        np.array: Current positions for arm joints J0 - J5
    """
    joint_values = []
    for i in range(6):
        name = f"J{i}"
        val = d.joint(name).qpos[0]
        joint_values.append(val)
    return np.array(joint_values, dtype=float)

def get_gripper_joints(m, d):
    """
    Current positions for finger joints (same order as set_joints actuators 6 - 9).

    Args:
        m: mujoco model
        d: mujoco data

    Returns:
        np.array: Current positions for finger joints
    """
    joint_values = []
    for name in _FINGER:
        val = d.joint(name).qpos[0]
        joint_values.append(val)
    return np.array(joint_values, dtype=float)

def get_joint_velocities(m, d):
    """
    Current velocities for arm joints J0 - J5.

    Args:
        m: mujoco model
        d: mujoco data

    Returns:
        np.array: Current velocities for arm joints J0 - J5
    """
    joint_velocities = []
    for i in range(6):
        name = f"J{i}"
        vel = d.joint(name).qvel[0]
        joint_velocities.append(vel)
    return np.array(joint_velocities, dtype=float)

def get_gripper_open_close(m, d):
    """
    We abstract the gripper as a single DOF instead of 4

    Returns: 
        Float between 0 if the gripper is closed and 1 if the gripper is open
    """
    gripper_joints = get_gripper_joints(m,d)
    rb = gripper_joints[0] # 0.96 at open
    lb = gripper_joints[2] # -0.96 at open
    value = (rb - lb)/(2 * 0.96)
    return value

def set_joints(m, d, cmd_joints, gripper_ratio=None):
    """
    Actuator commands: 6 arm positions + 4 finger.
    
    Args:
        m: mujoco model
        d: mujoco data
        cmd_joints: commands for arm joints
        gripper_ratio: gripper ratio

    Returns:
        None
    """
    # convert the command joints into a numpy array
    cmd_joints = np.asarray(cmd_joints, dtype=float)
    control_vector = np.zeros(10)
    control_vector[:6] = cmd_joints[:6]

    # gripper control logic
    if gripper_ratio is not None:
        gripper_ctrl = gripper_ctrl_from_ratio(gripper_ratio)
    else:
        gripper_ctrl = get_gripper_joints(m, d)

    control_vector[6:10] = gripper_ctrl
    d.ctrl[:] = control_vector

def ee_to_block_pos(m, d):
    """
    Calculates the Euclidean distance between the 'pinch_site' (EE) 
    and the 'cube' (target).
    
    Args:
        m: mujoco model
        d: mujoco data

    Returns:
        float: Euclidean distance between the 'pinch_site' and the 'cube'
    """
    # Get IDs for the target geometry and the end-effector site
    block_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")

    # Cartesian positions (x, y, z)
    block_pos = d.geom_xpos[block_id]

    site_name = "pinch_site"
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    ee_pos = d.site_xpos[sid]   # world position (x, y, z)
    
    dist = block_pos - ee_pos
    return dist

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Oscillate gripper to match sim behavior
            oscillation = math.sin(time.time() * SPEED)
            ratio = (oscillation + 1.0) * 0.5

            gripper_signals = gripper_ctrl_from_ratio(ratio)
            data.ctrl[6:10] = gripper_signals

            mujoco.mj_step(model, data)

             # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)