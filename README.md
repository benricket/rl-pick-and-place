# Kinova Gen3 Lite Control via Reinforcement Learning
This repository contains a MuJoCo & Gymnasium-based reinforcement learning environment for the Kinova Gen3 Lite (6-DOF) robot arm. The primary task is training the arm to reach a target cube using joint-space control.

---

## Main Environment: `ArmEnv` (`gym_env_angle.py`)
The `ArmEnv` class is a Gymnasium-compatible environment that interfaces with MuJoCo to simulate the robot's kinematics and dynamics.
### 1. Observation Space
The environment provides a 31-dimensional vector to the agent:
* **End-Effector Pose:** Cartesian position $(x, y, z)$ and Euler angles (ZXY).
* **Target Position:** The $(x, y, z)$ coordinates of the cube.
* **Joint State:** Current positions and velocities for all 6 arm joints.
* **Limit Data:** Distance to the upper and lower joint limits for each joint.
* **Gripper State:** Binary indicator of open/closed status and proximity to the box.

### 2. Action Space
The agent outputs a vector of 7 continuous actions scaled between $[-1, 1]$:
* **Actions 0-5:** Delta joint position changes ($\Delta q$) for joints J0 through J5.
* **Action 6:** Gripper control (mapped to an open/close ratio via helper scripts).

### 3. Reward Function
The reward structure is designed to guide the agent through the stages of the task:
* **Movement Reward:** Positive reinforcement for reducing the Euclidean distance between the EE and the cube.
* **Distance Penalty:** A baseline penalty proportional to the distance from the target.
* **End Effector Alignment Reward:** Encourages the gripper to point toward the cube using cosine similarity.
* **Velocity Penalty:** Penalties for high joint velocities and for approaching hardware joint limits.
* **Success:** A significant sparse reward for successfully reaching the target zone.

---

## Code Tutorial

### Prerequisites
To download the necesarry libraries, use the `requirements.txt` file. If using pip, this looks like `pip install -r requirements.txt`. 

### Running the Model
* To run the model, use `train_vec_env.py`. 
* To view the model, use `view_learned_policy.py`
