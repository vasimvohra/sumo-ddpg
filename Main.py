"""
SUMO 4x4 Intersection V2X Platoon DDPG Training
Combines traffic light control with V2X communication optimization
"""

import numpy as np
import os
import scipy.io
import time
from pathlib import Path

# Import your existing DDPG components
from ddpg_torch import Agent
from SUMO_V2X_Environment import SUMOV2XEnvironment

# SUMO Configuration (from your working 4x4simple.py)
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"  # Update this path
SUMO_CFG = r"platoon.sumocfg"  # Your working 4x4 config
USE_GUI = True

# V2X Parameters (from your original project)
size_platoon = 4
n_veh = 20
n_platoon = int(n_veh / size_platoon)
n_RB = 3
n_S = 2
max_power = 30  # dBm
V2I_min = 540  # bps/Hz
bandwidth = int(180000)
V2V_size = int((4000) * 8)

def main():
    for z in range(7, 11):
        start = time.time()

        label = f'sumo_v2x_model_{z}'
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_folder_path = os.path.join(current_dir, "model", label)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Initialize SUMO-V2X integrated environment
        env = SUMOV2XEnvironment(
            n_veh=n_veh, size_platoon=size_platoon, n_RB=n_RB,
            V2I_min=V2I_min,  bandwidth=bandwidth, V2V_size=V2V_size,
            sumo_config=SUMO_CFG, use_gui=USE_GUI
        )

        # Initialize new game with SUMO
        env.new_random_game()

        # DDPG Agent setup
        n_input = len(env.get_state(idx=0))
        n_output = 3  # channel, mode, power

        agent = Agent(alpha=0.0001, beta=0.001, input_dims=n_input,
                     tau=0.005, n_actions=n_output, gamma=0.99,
                     max_size=100000, C_fc1_dims=1024, C_fc2_dims=512,
                     C_fc3_dims=256, A_fc1_dims=1024, A_fc2_dims=512,
                     batch_size=64, n_agents=n_platoon)

        # Training parameters
        # Training parameters - UPDATED
        n_episode = 100  # Fewer episodes, but much longer
        episode_duration_seconds = 1  # 5 minutes per episode
        n_step_per_episode = int(episode_duration_seconds / env.time_fast)  # 300,000 steps


        print(f"Episode duration: {episode_duration_seconds}s ({n_step_per_episode:,} steps)")
        print(f"Total episodes: {n_episode}")
        print(f"Estimated training time per episode: {episode_duration_seconds / 60:.1f} minutes")

        # Training data collection
        record_reward_ = np.zeros([n_episode], dtype=np.float16)
        per_total_user_ = np.zeros([n_platoon, n_episode], dtype=np.float16)
        AoI_total = np.zeros([n_platoon, n_episode], dtype=np.float16)

        print(f"Starting SUMO-V2X Training for {n_episode} episodes...")

        for i_episode in range(n_episode):
            print(f"Episode {i_episode}/{n_episode}")

            # Reset environment - creates new platoons in SUMO
            env.reset()

            # Episode metrics
            record_reward = np.zeros([n_step_per_episode], dtype=np.float16)
            record_AoI = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
            per_total_user = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)

            # Reset V2X communication state
            env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
            env.active_links = np.ones(n_platoon, dtype=bool)

            if i_episode == 0:
                env.AoI = np.ones(n_platoon) * 100

            # Get initial states
            state_old_all = []
            for i in range(n_platoon):
                state = env.get_state(idx=i)
                state_old_all.append(state)

            # Episode training loop
            for i_step in range(n_step_per_episode):
                state_new_all = []
                action_all = []
                action_all_training = np.zeros([n_platoon, n_output], dtype=int)

                # Agent chooses actions
                action = agent.choose_action(np.asarray(state_old_all).flatten())
                action = np.clip(action, -0.999, 0.999)
                action_all.append(action)

                # Convert actions to discrete choices
                for i in range(n_platoon):
                    action_all_training[i, 0] = ((action[0+i*n_output]+1)/2) * n_RB  # RB selection
                    action_all_training[i, 1] = ((action[1+i*n_output]+1)/2) * n_S   # Mode selection
                    action_all_training[i, 2] = np.round(np.clip(((action[2+i*n_output]+1)/2) * max_power, 1, max_power))  # Power

                # Execute actions in SUMO-V2X environment
                training_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand_R, V2V_success = \
                    env.act_for_training(action_all_training)

                # Advance SUMO simulation
                env.sumo_step()

                # Store metrics
                record_reward[i_step] = global_reward
                for i in range(n_platoon):
                    per_total_user[i, i_step] = training_reward[i]
                    record_AoI[i, i_step] = env.AoI[i]

                # Update channel conditions
                env.renew_channels_fastfading()
                env.Compute_Interference(action_all_training)

                # Get new states
                for i in range(n_platoon):
                    state_new = env.get_state(i)
                    state_new_all.append(state_new)

                done = (i_step == n_step_per_episode - 1)

                # Store experience and learn
                agent.remember(np.asarray(state_old_all).flatten(),
                             np.asarray(action_all).flatten(),
                             global_reward, np.asarray(state_new_all).flatten(), done)
                agent.learn()

                # Update states
                for i in range(n_platoon):
                    state_old_all[i] = state_new_all[i]

                if i_step % 50 == 0:
                    print(f"  Step {i_step}, Global Reward: {global_reward:.4f}, V2V Success: {V2V_success:.2f}")

            # Episode statistics
            record_reward_[i_episode] = np.mean(record_reward)
            per_total_user_[:, i_episode] = np.mean(per_total_user, axis=1)
            AoI_total[:, i_episode] = np.mean(record_AoI, axis=1)

            # Save model periodically
            if i_episode % 50 == 0:
                agent.save_models()
                print(f"Saved model at episode {i_episode}")

        # Save final results
        print('Training Done. Saving final results...')
        reward_path = os.path.join(model_folder_path, 'reward.mat')
        reward_path_per = os.path.join(model_folder_path, 'per_total_user_.mat')
        AoI_path = os.path.join(model_folder_path, 'AoI.mat')

        scipy.io.savemat(reward_path, {'reward': record_reward_})
        scipy.io.savemat(reward_path_per, {'reward_per': per_total_user_})
        scipy.io.savemat(AoI_path, {'AoI': AoI_total})

        agent.save_models()
        env.close()

        end = time.time()
        print(f"Training completed in {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
