# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

-------------------------------------------------------------------------------------------------------------------------
Students should only edit README.md below this line.

# Team Additions

## Contact Sensor Update
Add the following lines to the setup_scene function, so that the contact sensor information from the simulation can be fed to the scene. 
```
# In Rob6323Go2Env._setup_scene
self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
self.scene.sensors["contact_sensor"] = self._contact_sensor # <---- Added
```
To prevent the rewards from breaking the training, decrease the base minimum height from 0.2 to 0.05.
```
# In Rob6323Go2EnvCfg
base_height_min = 0.05  # Terminate
```
## Feet2Contact Reward (Optional) 
This reward is optional. If hopping is ever observed in your simulation, or you want to finetune the gait to be more symmetrical, this reward ensures that. 

### Scale Reward
Add the scale for the reward. Adjust the number as needed till you reach your desired gait. Start from as high as 1 and decrease until desired results are reached. This reward is harsh so a small scale is necessary.
```
# In Rob6323Go2EnvCfg
base_height_min = 0.00  # Originally Deactivated: Can start with 0.0009
```
### Reward Logic
This reward penalizes more or less than two feet on the ground. The forces for each of the foot are collected. The num_contacts finds the number of feet in contact with the ground. The final rew_foot2contact finds the percentage and includes a negative sign to penalize the incorrect amount of feet.
```
# In Rob6323Go2Env._get_rewards
        # Added Logic
        foot_contact_forces_z = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, 2]
        num_contacts = (foot_contact_forces_z > 1.0).sum(1).float()
        rew_foot2contact = - torch.abs(num_contacts - 2) / 2.0

        rewards = {
            # ... Other rewards
            "foot2contact": rew_foot2contact * self.cfg.foot2contact_reward_scale,  # <--- Added
```

## Torque Regularization
To generate smoother movement and prevent the application of sudden torques to the Unitree Go2, torque regularization is included in the project.
### Initialization
A new attribute is created in the initialization function. This attribute will hold the torque information for the entire class.
```
# In Rob6323Go2Env.__init__
self.torques = torch.zeros(self.num_envs, 12, device=self.device)
```
### Sending Torque Information to Attribute

The torques are already calculated in _apply_action(self). Use this calculation as the value for self.torques.
```
# In Rob6323Go2Env._apply_action
def _apply_action(self) -> None:
        # Compute PD torques
        torques_pd = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
        
        # Apply actuator friction model
        tau_stiction = self.F_s * torch.tanh(self.robot.data.joint_vel / 0.1)
        tau_viscous = self.mu_v * self.robot.data.joint_vel
        tau_friction = tau_stiction + tau_viscous

        torques = torch.clip(torques_pd - tau_friction, -self.torque_limits, self.torque_limits)
        self.torques = torques  # <----- Added

... rest of function
```

### Torque Reward Scale
We need to give the model a scale to better understand the penalty for having high torques. Add this reward to your configuration file. Any value smaller than 0.0001 should work.
```
# In Rob6323Go2EnvCfg
 torque_reward_scale = -0.00001  # Penalty for high torque magnitude
```
### Apply Torque Regularization Reward
Now that we have the torques, we can calculate the penalty and send it to the rewards dictionary to prevent large values of torques!

```
# In Rob6323Go2Env._get_rewards
rew_torque = torch.sum(torch.square(self.torques), dim=1)
rewards = {
            # ... other rewards
            "rew_torque": rew_torque * self.cfg.torque_reward_scale,
        }
```

##  Actuator Friction Model with Randomization
To create a more realistic simulation, we add an actuator friction/viscous model where the the paramaters are randomized per episode. Before adding friction to the model, the torques must be subtracted to find the PD controller torque.


### Define Friction Actuator Parameters
Add the following parameters to the config.

```
# In Rob6323Go2EnvCfg
mu_v_lim = 0.3
F_s_lim = 2.5 
```

### Initialization of Friction Actuator Parameters
Define new attributes for the viscous and static friction.
```
# In Rob6323Go2Env.__init__
self.mu_v = torch.zeros(self.num_envs, 12, device=self.device)
self.F_s = torch.zeros(self.num_envs, 12, device=self.device)
```

### Adjust _apply_action calculations
The PD torques need to be calculated by using the PD controller. After calculating the viscious and static friction, they are combined together and subtracted from the original PD torques found to find the total torque.

```
# In Rob6323Go2Env._apply_action
def _apply_action(self) -> None:
        # Compute PD torques
        torques_pd = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel # <--- Added
        
        # Apply actuator friction model 
        tau_stiction = self.F_s * torch.tanh(self.robot.data.joint_vel / 0.1)                    # <--- Added
        tau_viscous = self.mu_v * self.robot.data.joint_vel                                      # <--- Added
        tau_friction = tau_stiction + tau_viscous                                                # <--- Added

        torques = torch.clip(torques_pd - tau_friction, -self.torque_limits, self.torque_limits) # <--- Added
        self.torques = torques  # For torque regularization

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)
```

### Reset Actuator Friction Parameters
For each episode, the static and viscious friction values need to be reset since the friction for each episode is different.
```
# In Rob6323Go2Env._reset_idx
self.mu_v[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) * self.cfg.mu_v_lim).expand(-1, 12)
self.F_s[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) * self.cfg.F_s_lim).expand(-1, 12)
```
-------------------------------------------------------------------------------------------------------------------------
# Tutorial Additions
To convert the minimal implementation of DirectRLEnv to a more robust walking policy for the Unitree Go2, additional rewards and functions need to be added to the config and env files. 

## PD Controller
Define custom gains and update the robot_cfg in the config. Add the following import:
```
from isaaclab.actuators import ImplicitActuatorCfg
```
Initalize parameters in the __init__ function. Add the following _pre_physics_step and _apply_action to Rob6323Go2EnvCfg.
```
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self._actions = actions.clone()
    # Compute desired joint positions from policy actions
    self.desired_joint_pos = (
        self.cfg.action_scale * self._actions 
        + self.robot.data.default_joint_pos
    )
```

## Rewards
The following rewards need to be added. With enough rewards, we can describe the behavior we want the robot to learn on its own. For the raibert_heuristic_reward, add a raibert_heuristic function describing the symmetric gait.

- action_rate: Penalizes high frequency oscillations with first and second derivative
- orient_reward: Prevent orientation
- lin_vel_z_reward: Prevent bouncing
- dof_vel_reward: Prevents high joint velocities
- ang_vel_xy_reward: Penalize angular velocity in the X and Y plane
- raibert_heuristic_reward: Footwork that follows trotting symmetric gait
- feet_clearance_reward: Lifting feet during a swing
- tracking_contacts_shaped_force_reward: Grounding feet during a stance

## Reward Scales
Add the following reward scales in the config.
```
# ... rewards given
action_rate_reward_scale = -0.1  # Added: Step 1.1 Update Configuration 
orient_reward_scale = -5.0
lin_vel_z_reward_scale = -0.02
dof_vel_reward_scale = -0.0001
ang_vel_xy_reward_scale = -0.001
raibert_heuristic_reward_scale = -10.0
feet_clearance_reward_scale = -30.0
tracking_contacts_shaped_force_reward_scale = 4.0
```
## Reward Implementations

### Apply Action Rate
Initialize a torch tensor to record past actions. Find the difference between the current and last self._actions. The second derivative (Current - 2* Last + 2nd Last Value) is calculated and added to the first derivative.

### Orientation
Use projected_gravity_b to penalize non-vertical orientation. Use the following line in the _get_rewards function.
```
# In Rob6323Go2Env._get_rewards
rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
```
### Vertical Velocity
Use the z component of the base linear velocity. Use the following line in the _get_rewards function.
```
# In Rob6323Go2Env._get_rewards
rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
```
### Joint Velocities
joint_vel has the values for the joint velocities. 
```
# In Rob6323Go2Env._get_rewards
rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

```
### Angular Velocity
The angular velocities can be accessed by root_ang_vel_b.
```
# In Rob6323Go2Env._get_rewards
rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
```
### _reset_idx and Episode Sums
Gait indicies and last actions must be reset for every episode.
After defining the reward in the rewards dictionary in the _get_rewards function, the reward must be added to the episode_sums list in the __init__.

### Raibert Heuristic and Step Contact Targets
There are a few steps for refining the gait of the Unitree Go2.
The first step is to calculate where the feet would be due to command velocity. This function is _step_contact_targets. Raibert Heuristic defines the error of where the feet should be compared to where they are now. This error is sent to the _get_rewards function.

### Feet Clearance and Shaped Forces
By referencing the Legacy Isaac Gym code, the following logic was added to the rewards function to calculate the rewards.

```
# In Rob6323Go2Env._get_rewards
# Feet clearance reward
phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
foot_height = self.foot_positions_w[:, :, 2]
target_height = 0.08 * phases + 0.02

rew_foot_clearance = torch.square(target_height - foot_height) * (1.0 - self.desired_contact_states)
rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1)

# Contact tracking shaped by forces reward
foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :], dim=-1)
desired_contact = self.desired_contact_states
rew_tracking_contacts_shaped_force = 0.
for i in range(4):
    rew_tracking_contacts_shaped_force += - (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))        
rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force / 4 # avg over 4 feet
```

