# Description: This script is used to simulate the full model of the robot in mujoco
import pathlib

# Authors:
# Giulio Turrisi, Daniel Ordonez
import time
from os import PathLike
from pprint import pprint

import numpy as np

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm
import pickle
import multiprocessing
from multiprocessing import Process

# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco

# PyMPC controller imports
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

from params import *
def run_sim(seed=0, postfix='a'):
    from quadruped_pympc import config as cfg

    qpympc_cfg = cfg
    render = False
    np.set_printoptions(precision=3, suppress=True)
    for trial in range(3, N_TRIALS):
        qpympc_cfg = cfg
        # Added by Noah -----------
        if trial < 4:
            base_vel_cmd = 'forward'
        else:
            base_vel_cmd = ['forward', 'rotate']
        if trial == 6:
            qpympc_cfg.simulation_params['gait'] = 'full_stance'
        # End added by Noah -----------
        np.random.seed(seed)
        ref_base_lin_vel = REF_BASE_LIN_VELS[trial]
        ref_base_ang_vel = REF_BASE_ANG_VELS[trial]
    

        # OG Code ----------------------------------------------------------------------
        robot_name = qpympc_cfg.robot
        hip_height = qpympc_cfg.hip_height
        robot_leg_joints = qpympc_cfg.robot_leg_joints
        robot_feet_geom_names = qpympc_cfg.robot_feet_geom_names
        scene_name = qpympc_cfg.simulation_params["scene"]
        simulation_dt = qpympc_cfg.simulation_params["dt"]

        # Save all observables available.
        state_obs_names = list(QuadrupedEnv.ALL_OBS)  # + list(IMU.ALL_OBS)

        # Create the quadruped robot environment -----------------------------------------------------------
        env = QuadrupedEnv(
            robot=robot_name,
            scene=scene_name,
            sim_dt=simulation_dt,
            ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,  # pass a float for a fixed value
            ref_base_ang_vel=ref_base_ang_vel,  # pass a float for a fixed value
            ground_friction_coeff=FRICTION_COEFF,  # pass a float for a fixed value
            base_vel_command_type=base_vel_cmd,  # "forward", "random", "forward+rotate", "human"
            state_obs_names=tuple(state_obs_names),  # Desired quantities in the 'state' vec
        )
        # Added by Noah: Set the heading vector to go in the x or y direction
        env.zz_base_heading_vel_vec = BASE_HEADING_VEL_VECS[trial]
        # End Added by Noah ----------------------------
        pprint(env.get_hyperparameters())

        # Some robots require a change in the zero joint-space configuration. If provided apply it
        if qpympc_cfg.qpos0_js is not None:
            env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

        env.reset(random=False)
        if render:
            env.render()

        # Initialization of variables used in the main control loop --------------------------------

        # Torque vector
        tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
        # Torque limits
        tau_soft_limits_scalar = 0.9
        tau_limits = LegsAttr(
            FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
            FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
            RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
            RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
        )

        # Feet positions and Legs order
        feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
        legs_order = ["FL", "FR", "RL", "RR"]

        # Create HeightMap -----------------------------------------------------------------------
        if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
            from gym_quadruped.sensors.heightmap import HeightMap

            resolution_vfa = 0.04
            dimension_vfa = 7
            heightmaps = LegsAttr(
                FL=HeightMap(
                    n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
                ),
                FR=HeightMap(
                    n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
                ),
                RL=HeightMap(
                    n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
                ),
                RR=HeightMap(
                    n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData
                ),
            )
        else:
            heightmaps = None

        # Quadruped PyMPC controller initialization -------------------------------------------------------------
        # mpc_frequency = qpympc_cfg.simulation_params["mpc_frequency"]
        quadrupedpympc_observables_names = (
            "ref_base_height",
            "ref_base_angles",
            "ref_feet_pos",
            "nmpc_GRFs",
            "nmpc_footholds",
            "swing_time",
            "phase_signal",
            "lift_off_positions",
        )

        quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
            initial_feet_pos=env.feet_pos,
            legs_order=tuple(legs_order),
            feet_geom_id=env._feet_geom_id,
            quadrupedpympc_observables_names=quadrupedpympc_observables_names,
        )

        # Data recording -------------------------------------------------------------------------------------------
        root_path = pathlib.Path(RECORDING_PATH)
        root_path.mkdir(exist_ok=True)
        save_path = pathlib.Path(
            root_path
            / f"{robot_name}/{scene_name}"
            / f"trial{trial}"
            / f"lin_vel={ref_base_lin_vel} ang_vel={ref_base_ang_vel} friction={FRICTION_COEFF}"
            / f"ep={N_EPISODES}_steps={int(NUM_SECONDS_PER_EPISODE // simulation_dt):d}"
        )

        # -----------------------------------------------------------------------------------------------------------
        RENDER_FREQ = 30  # Hz
        N_STEPS_PER_EPISODE = int(NUM_SECONDS_PER_EPISODE // simulation_dt)
        last_render_time = time.time()

        state_obs_history, ctrl_state_history = [], []
        num_terminations = 0
        ctr = 0
        for episode_num in range(N_EPISODES // (NUM_PROCESSES)):
            ep_state_history, ep_ctrl_state_history, ep_time = [], [], []
            for _ in tqdm(range(N_STEPS_PER_EPISODE), desc=f"Ep:{episode_num:d}-steps:", total=N_STEPS_PER_EPISODE):
                # Update value from SE or Simulator ----------------------
                feet_pos = env.feet_pos(frame="world")
                hip_pos = env.hip_positions(frame="world")
                base_lin_vel = env.base_lin_vel(frame="world")
                base_ang_vel = env.base_ang_vel(frame="world")
                base_ori_euler_xyz = env.base_ori_euler_xyz
                base_pos = env.base_pos
                com_pos = env.com

                # Get the reference base velocity in the world frame
                ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

                # Get the inertia matrix
                if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                    inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
                else:
                    inertia = qpympc_cfg.inertia.flatten()

                # Get the qpos and qvel
                qpos, qvel = env.mjData.qpos, env.mjData.qvel
                # Idx of the leg
                legs_qvel_idx = env.legs_qvel_idx  # leg_name: [idx1, idx2, idx3] ...
                legs_qpos_idx = env.legs_qpos_idx  # leg_name: [idx1, idx2, idx3] ...
                joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR, RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)

                # Get Centrifugal, Coriolis, Gravity, Friction for the swing controller
                legs_mass_matrix = env.legs_mass_matrix
                legs_qfrc_bias = env.legs_qfrc_bias
                legs_qfrc_passive = env.legs_qfrc_passive

                # Compute feet jacobians
                feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
                feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)

                # Compute feet velocities
                feet_vel = LegsAttr(**{leg_name: feet_jac[leg_name] @ env.mjData.qvel for leg_name in legs_order})

                # Quadruped PyMPC controller --------------------------------------------------------------
                tau = quadrupedpympc_wrapper.compute_actions(
                    com_pos,
                    base_pos,
                    base_lin_vel,
                    base_ori_euler_xyz,
                    base_ang_vel,
                    feet_pos,
                    hip_pos,
                    joints_pos,
                    heightmaps,
                    legs_order,
                    simulation_dt,
                    ref_base_lin_vel,
                    ref_base_ang_vel,
                    env.step_num,
                    qpos,
                    qvel,
                    feet_jac,
                    feet_jac_dot,
                    feet_vel,
                    legs_qfrc_passive,
                    legs_qfrc_bias,
                    legs_mass_matrix,
                    legs_qpos_idx,
                    legs_qvel_idx,
                    tau,
                    inertia,
                    env.mjData.contact,
                )
                # Limit tau between tau_limits
                for leg in ["FL", "FR", "RL", "RR"]:
                    tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                    tau[leg] = np.clip(tau[leg], tau_min, tau_max)

                # Set control and mujoco step -------------------------------------------------------------------------
                action = np.zeros(env.mjModel.nu)
                action[env.legs_tau_idx.FL] = tau.FL
                action[env.legs_tau_idx.FR] = tau.FR
                action[env.legs_tau_idx.RL] = tau.RL
                action[env.legs_tau_idx.RR] = tau.RR

                # action_noise = np.random.normal(0, 2, size=env.mjModel.nu)
                # action += action_noise

                # Apply the action to the environment and evolve sim --------------------------------------------------
                state, reward, is_terminated, is_truncated, info = env.step(action=action)

                # Get Controller state observables
                ctrl_state = quadrupedpympc_wrapper.get_obs()

                # Store the history of observations and control -------------------------------------------------------
                base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
                ctrl_state["base_poz_z_err"] = base_poz_z_err

                ep_state_history.append(state)
                ep_time.append(env.simulation_time)
                ep_ctrl_state_history.append(ctrl_state)

                if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                    env.render()
                    last_render_time = time.time()

                # Reset the environment if the episode is terminated ------------------------------------------------
                if env.step_num >= N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                    hist_path = save_path / f"state_hist_{ctr}{postfix}.pkl"
                    ctr += 1
                    hist_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(hist_path, 'wb') as f:
                        pickle.dump(ep_state_history, f)
                    if is_terminated:
                        num_terminations += 1
                        print(f"Environment terminated on trial: {trial}")
                    else:
                        state_obs_history.append(ep_state_history)
                        ctrl_state_history.append(ep_ctrl_state_history)     

                    env.reset(random=False)
                    quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))

        # Save all of the state information for one trial
        # with open(hist_path, 'wb') as f:
        #     pickle.dump(state_obs_history, f)
        with open(save_path / f"failures_{postfix}.pkl", 'wb') as f:
            pickle.dump(num_terminations, f)

        env.close()

if __name__ == "__main__":
    
    # run_sim()
    
    # Run the simulation in parallel
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    postfix = 'a'
    for i in range(NUM_PROCESSES):        
        p = Process(
            target=run_sim,
            args=(i*NUM_PROCESSES, postfix),
        )
        p.start()
        time.sleep(2)
        processes.append(p)
        postfix = chr(ord(postfix) + 1)

    for p in processes:
        p.join()

