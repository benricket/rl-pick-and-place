import time
import os
import sys
import numpy as np

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../kinova_gen3/rl_scene.xml')
d = mujoco.MjData(m)
iter_count = 0

def get_joints(m,d):
    """
    docstring
    """
    joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']
    joint_vals = [d.joint(name).qpos for name in joint_names]
    return np.array(joint_vals)

def set_joints(m,d,cmd_joints,gripper=0.0):
    """
    docstring
    """
    # gripper goes from 0 (open) to 255 (closed)
    if cmd_joints.size == 7: # we don't have control for the gripper
        cmd_joints = np.append(cmd_joints,gripper)
        print(cmd_joints)
    d.ctrl = cmd_joints[:]
    

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    if iter_count == 0:
        geom_name = 'cube'
        geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        position = d.geom_xpos[geom_id]
        #print(f"{geom_name} position: {position}")

    if iter_count % 50 == 0:
        joints = get_joints(m,d)
        joints += 0.005
        set_joints(m,d,joints,1.0)
        #print(f"control: {d.ctrl}")


    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
    
    iter_count = (iter_count + 1) % 500