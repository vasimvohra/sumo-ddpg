"""
SUMO_V2X_Environment.py - CONTROLLED VERSION
Max vehicles with continuous flow for larger scenarios
"""

import numpy as np
import math
import traci
import random
import time


class SUMOV2XEnvironment:
    def __init__(self, n_veh, size_platoon, n_RB, V2I_min, bandwidth, V2V_size,
                 sumo_config, use_gui=False):

        # CONTROLLED TRAFFIC PARAMETERS
        self.max_total_vehicles = 25  # Never exceed this number
        self.max_platoons = 5  # Maximum platoons at once
        self.size_platoon = size_platoon
        self.n_platoons = self.max_platoons  # For communication arrays
        self.n_veh = self.max_total_vehicles
        self.last_creation_time = 0
        self.platoon_cooldown = 2
        # Rest of your original __init__ parameters...
        self.n_RB = n_RB
        self.V2I_min = V2I_min
        self.bandwidth = bandwidth
        self.V2V_demand_size = V2V_size

        # SUMO Configuration
        self.sumo_config = sumo_config
        self.use_gui = use_gui
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
       
        self.sumo_running = False

        # Import channel models
        from Classes.Environment_Platoon import V2Vchannels, V2Ichannels
        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()

        # Vehicle tracking
        self.platoon_leaders = []
        self.platoon_followers = {}
        self.vehicle_ids = []
        self.vehicle_positions = {}
        self.platoon_counter = 0  # For unique IDs

        # Communication parameters (same as original)
        self.time_fast = 0.001
        self.time_slow = 0.1
        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.bsAntGain, self.bsNoiseFigure = 8, 5
        self.vehAntGain, self.vehNoiseFigure = 3, 9

        # Initialize arrays using max_platoons
        self.AoI = np.ones(self.max_platoons, dtype=np.float32) * 100
        self.V2V_demand = np.zeros(self.max_platoons)
        self.active_links = np.ones(self.max_platoons, dtype=bool)
        self.Interference_all = np.zeros(self.max_platoons) + self.sig2
        self.individual_time_limit = np.zeros(self.max_platoons)

        # Additional communication arrays
        self.platoon_V2I_Interference = np.zeros(self.max_platoons)
        self.platoon_V2I_Signal = np.zeros(self.max_platoons)
        self.platoon_V2V_Interference = np.zeros([self.max_platoons, self.size_platoon - 1])
        self.platoon_V2V_Signal = np.zeros([self.max_platoons, self.size_platoon - 1])
        self.V2I_Interference_all = np.zeros(self.max_platoons)
        self.V2V_Interference_all = np.zeros([self.max_platoons, self.size_platoon - 1])
        self.interplatoon_rate = np.zeros(self.max_platoons)
        self.intraplatoon_rate = np.zeros(self.max_platoons)

    def _start_sumo(self):
    
      if not self.sumo_running:
          # FIXED: Use correct binary names for Linux/Colab
          if self.use_gui:
              self.sumo_binary = "sumo-gui"  # Won't work in Colab anyway
          else:
              self.sumo_binary = "sumo"      # Correct for headless
          
          cmd = [self.sumo_binary, "-c", self.sumo_config,
                "--no-warnings", "--step-length", "1.0",
                "--no-step-log", "--time-to-teleport", "-1",
                f"--max-num-vehicles", str(self.max_total_vehicles),
                "--end", "7200"]
          
          # CRITICAL: Force no GUI in Colab
          if self.use_gui:
              print("Warning: GUI not available in Colab, using headless mode")
              self.use_gui = False
              self.sumo_binary = "sumo"
              cmd[0] = "sumo"  # Update command
          
          try:
              import traci
              traci.start(cmd)
              self.sumo_running = True
              print(f"SUMO started successfully with command: {' '.join(cmd)}")
          except Exception as e:
              print(f"Error starting SUMO: {e}")
              print(f"Command attempted: {' '.join(cmd)}")
              raise


    def _stop_sumo(self):
        """Stop SUMO simulation"""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False

    def _create_initial_platoons(self):
        """Create controlled number of initial platoons"""
        self.platoon_leaders = []
        self.platoon_followers = {}
        self.vehicle_ids = []
        self.platoon_counter = 0

        # Routes for larger scenario (adjust based on your network)
        routes = [
            "horizontal", "route_sn_straight", "route_we_straight", "route_ew_straight",
            "route_nw_left", "route_se_left", "route_en_left", "route_ws_left",
            "route_ne_right", "route_sw_right", "route_es_right", "route_wn_right"
        ]

        # Create only 2-3 initial platoons, not max
        initial_platoons = min(3, self.max_platoons)

        for p_idx in range(initial_platoons):
            self._create_single_platoon(p_idx, routes[p_idx % len(routes)])

    def _create_single_platoon(self, platoon_idx, route_id):
        """Create one platoon with strict control"""
        current_time = traci.simulation.getTime()
        leader_id = f"platoon_{self.platoon_counter}_leader"

        try:
            # Create leader
            traci.vehicle.add(leader_id, routeID=route_id, depart=str(current_time))
            traci.vehicle.setVehicleClass(leader_id, "passenger")
            traci.vehicle.setSpeedMode(leader_id, 31)
            traci.vehicle.setColor(leader_id, (255, 0, 0))  # Red for visibility

            self.platoon_leaders.append(leader_id)
            self.vehicle_ids.append(leader_id)
            self.platoon_followers[leader_id] = []

            # Create followers
            for car_idx in range(1, self.size_platoon):
                follower_id = f"platoon_{self.platoon_counter}_car_{car_idx}"
                follower_depart = current_time + car_idx * 2.0

                traci.vehicle.add(follower_id, routeID=route_id, depart=str(follower_depart))
                traci.vehicle.setVehicleClass(follower_id, "passenger")
                traci.vehicle.setSpeedMode(follower_id, 31)
                traci.vehicle.setTau(follower_id, 0.5)
                traci.vehicle.setMinGap(follower_id, 2.0)
                traci.vehicle.setColor(follower_id, (0, 255, 0))  # Green

                self.platoon_followers[leader_id].append(follower_id)
                self.vehicle_ids.append(follower_id)

            print(f"Created platoon {self.platoon_counter} at time {current_time}")
            self.platoon_counter += 1

        except traci.TraCIException as e:
            print(f"Could not create platoon: {e}")

    def _smart_platoon_management(self):
        """Improved smart platoon creation with cooldown timer"""
        current_time = traci.simulation.getTime()
        current_vehicles = len(traci.vehicle.getIDList())
        current_platoons = len(self.platoon_leaders)

        # Clean up first
        self._cleanup_exited_vehicles()

        # Check all conditions
        if current_platoons >= self.max_platoons:
            return

        vehicles_after_new_platoon = current_vehicles + self.size_platoon
        if vehicles_after_new_platoon > self.max_total_vehicles:
            return

        # NEW: Use cooldown instead of modulo
        if (current_time - self.last_creation_time) < self.platoon_cooldown:
            return

        # Passed all checks - create platoon
        routes = [
            "horizontal", "route_sn_straight", "route_we_straight", "route_ew_straight",
            "route_nw_left", "route_se_left", "route_en_left", "route_ws_left",
            "route_ne_right", "route_sw_right", "route_es_right", "route_wn_right"
        ]

        route_id = random.choice(routes)
        unique_base = f"smart_{self.platoon_counter}_{random.randint(1000, 9999)}"
        leader_id = f"{unique_base}_leader"

        # Check if exists
        existing_vehicles = set(traci.vehicle.getIDList())
        if leader_id in existing_vehicles:
            return

        try:
            # Create leader
            traci.vehicle.add(leader_id, routeID=route_id, depart=str(current_time))
            traci.vehicle.setVehicleClass(leader_id, "passenger")
            traci.vehicle.setSpeedMode(leader_id, 31)
            traci.vehicle.setColor(leader_id, (0, 0, 255))  # Blue for smart platoons

            self.platoon_leaders.append(leader_id)
            self.vehicle_ids.append(leader_id)
            self.platoon_followers[leader_id] = []

            # Create followers
            for car_idx in range(1, self.size_platoon):
                follower_id = f"{unique_base}_car_{car_idx}"
                follower_depart = current_time + car_idx * 2.0

                if follower_id not in existing_vehicles:
                    traci.vehicle.add(follower_id, routeID=route_id, depart=str(follower_depart))
                    traci.vehicle.setVehicleClass(follower_id, "passenger")
                    traci.vehicle.setSpeedMode(follower_id, 31)
                    traci.vehicle.setTau(follower_id, 0.5)
                    traci.vehicle.setMinGap(follower_id, 2.0)
                    traci.vehicle.setColor(follower_id, (100, 100, 255))  # Light blue

                    self.platoon_followers[leader_id].append(follower_id)
                    self.vehicle_ids.append(follower_id)

           # print(f"Added smart platoon {self.platoon_counter} at time {current_time}")
            self.platoon_counter += 1
            self.last_creation_time = current_time  # Update cooldown timer

        except traci.TraCIException as e:
            print(f"Could not create smart platoon: {e}")

    def _cleanup_exited_vehicles(self):
        """Clean cleanup - same as your original"""
        current_vehicles = set(traci.vehicle.getIDList())
        self.vehicle_ids = [v_id for v_id in self.vehicle_ids if v_id in current_vehicles]

        exited_leaders = []
        for leader_id in self.platoon_leaders:
            if leader_id not in current_vehicles:
                exited_leaders.append(leader_id)

        for leader_id in exited_leaders:
            self.platoon_leaders.remove(leader_id)
            if leader_id in self.platoon_followers:
                del self.platoon_followers[leader_id]

    def sumo_step(self):
        """Controlled SUMO step with debugging"""
        if self.sumo_running:
            traci.simulationStep()

            # Print controlled vehicle info every 10 seconds
            current_time = traci.simulation.getTime()
            if int(current_time) % 10 == 0:
                vehicle_count = len(traci.vehicle.getIDList())
                platoon_count = len(self.platoon_leaders)
                #print(
                 #   f"Time {current_time}: {vehicle_count}/{self.max_total_vehicles} vehicles, {platoon_count}/{self.max_platoons} platoons")

            self._update_vehicle_positions()
            self._smart_platoon_management()  # Smart creation instead of continuous

        # ... (keep ALL your original communication methods exactly the same)

    def reset(self):
        """Reset with controlled traffic"""
        self._stop_sumo()
        self._start_sumo()
        self._create_initial_platoons()  # Use controlled creation

        # Let simulation settle
        for _ in range(30):
            self.sumo_step()

        # Reset communication state
        self.AoI.fill(100)
        self.V2V_demand = self.V2V_demand_size * np.ones(self.max_platoons)
        self.active_links = np.ones(self.max_platoons, dtype=bool)
        self.individual_time_limit = self.time_slow * np.ones(self.max_platoons)

    def new_random_game(self):
        """Initialize controlled game"""
        if not self.sumo_running:
            self._start_sumo()
            self._create_initial_platoons()

            for _ in range(20):
                self.sumo_step()
                if not self.sumo_running:
                    break

        # Reset communication state
        self.V2V_demand = self.V2V_demand_size * np.ones(self.max_platoons)
        self.active_links = np.ones(self.max_platoons, dtype=bool)
        self.AoI = np.ones(self.max_platoons) * 100
        self.individual_time_limit = self.time_slow * np.ones(self.max_platoons)
    def _update_vehicle_positions(self):
        """Update vehicle positions from SUMO"""
        self.vehicle_positions.clear()
        if not self.sumo_running:
            return

        current_vehicles = traci.vehicle.getIDList()

        for v_id in self.vehicle_ids:
            if v_id in current_vehicles:
                try:
                    self.vehicle_positions[v_id] = traci.vehicle.getPosition(v_id)
                except traci.TraCIException:
                    pass

    def calculate_channel_conditions(self):
        """Calculate V2V and V2I channel conditions"""
        self.V2V_channels_abs = np.zeros((self.n_veh, self.n_veh))
        self.V2I_channels_abs = np.zeros(self.n_veh)

        for i, veh_i_id in enumerate(self.vehicle_ids):
            if i >= self.n_veh:  # Safety check
                break
            for j, veh_j_id in enumerate(self.vehicle_ids):
                if j >= self.n_veh:  # Safety check
                    break
                if i != j and veh_i_id in self.vehicle_positions and veh_j_id in self.vehicle_positions:
                    pos_i = self.vehicle_positions[veh_i_id]
                    pos_j = self.vehicle_positions[veh_j_id]
                    self.V2V_channels_abs[i, j] = self.V2Vchannels.get_path_loss(pos_i, pos_j)

        for i, veh_id in enumerate(self.vehicle_ids):
            if i >= self.n_veh:  # Safety check
                break
            if veh_id in self.vehicle_positions:
                pos = self.vehicle_positions[veh_id]
                self.V2I_channels_abs[i] = self.V2Ichannels.get_path_loss(pos)

        self.renew_channels_fastfading()

    def renew_channels_fastfading(self):
        """Add fast fading"""
        v2v_fast = np.random.normal(0, 1, (self.n_veh, self.n_veh, self.n_RB)) + \
                   1j * np.random.normal(0, 1, (self.n_veh, self.n_veh, self.n_RB))
        self.V2V_channels_with_fastfading = np.repeat(
            self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2) - \
            20 * np.log10(np.abs(v2v_fast) / math.sqrt(2))

        v2i_fast = np.random.normal(0, 1, (self.n_veh, self.n_RB)) + \
                   1j * np.random.normal(0, 1, (self.n_veh, self.n_RB))
        self.V2I_channels_with_fastfading = np.repeat(
            self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1) - \
            20 * np.log10(np.abs(v2i_fast) / math.sqrt(2))

    def get_state(self, idx):
        """Get state for platoon leader idx"""
        # Safety check
        if idx >= self.n_platoons:
            return np.zeros(10)  # Return default state

        self.calculate_channel_conditions()

        leader_global_idx = idx * self.size_platoon

        # Check array bounds
        if leader_global_idx >= len(self.V2I_channels_abs):
            return np.zeros(10)

        V2I_abs = (self.V2I_channels_abs[leader_global_idx] - 60) / 60.0
        V2I_fast = (self.V2I_channels_with_fastfading[leader_global_idx, :] -
                   self.V2I_channels_abs[leader_global_idx] + 10) / 35

        follower_indices = leader_global_idx + 1 + np.arange(self.size_platoon - 1)
        follower_indices = follower_indices[follower_indices < len(self.V2V_channels_abs)]

        if len(follower_indices) > 0:
            V2V_abs = (self.V2V_channels_abs[leader_global_idx, follower_indices] - 60) / 60.0
            V2V_fast = (self.V2V_channels_with_fastfading[leader_global_idx, follower_indices, :] -
                       self.V2V_channels_abs[leader_global_idx, follower_indices].reshape(
                           len(follower_indices), 1) + 10) / 35
        else:
            V2V_abs = np.zeros(self.size_platoon - 1)
            V2V_fast = np.zeros((self.size_platoon - 1, self.n_RB))

        Interference = (-self.Interference_all[idx] - 60) / 60 if idx < len(self.Interference_all) else 0
        AoI_levels = self.AoI[idx] / (int(self.time_slow / self.time_fast)) if idx < len(self.AoI) else 0
        V2V_load_remaining = np.asarray([self.V2V_demand[idx] / self.V2V_demand_size]) if idx < len(self.V2V_demand) else np.asarray([0])

        state = np.concatenate((
            np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1),
            np.reshape(V2V_abs, -1), np.reshape(V2V_fast, -1),
            np.reshape(Interference, -1), np.reshape(AoI_levels, -1),
            V2V_load_remaining
        ), axis=0)

        return np.nan_to_num(state, nan=-1.0)

    def Revenue_function(self, quantity, threshold):
        """Revenue function from Environment_Platoon"""
        return 1 if quantity >= threshold else 0

    def Age_of_Information(self, V2I_rate):
        """Age of Information from Environment_Platoon"""
        for i in range(self.n_platoons):
            if V2I_rate[i] >= self.V2I_min:
                self.AoI[i] = 1
            else:
                self.AoI[i] += 1
            if self.AoI[i] >= (self.time_slow / self.time_fast):
                self.AoI[i] = (self.time_slow / self.time_fast)
        return self.AoI

    def Compute_Performance_Reward_Train(self, platoons_actions):
        """Compute Performance and Reward for Training"""
        sub_selection = platoons_actions[:, 0].astype('int').reshape(self.n_platoons, 1)
        platoon_decision = platoons_actions[:, 1].astype('int').reshape(self.n_platoons, 1)
        power_selection = platoons_actions[:, 2].reshape(self.n_platoons, 1)

        # Initialize interference and signal arrays
        self.platoon_V2I_Interference = np.zeros(self.n_platoons)
        self.platoon_V2I_Signal = np.zeros(self.n_platoons)
        self.platoon_V2V_Interference = np.zeros([self.n_platoons, self.size_platoon - 1])
        self.platoon_V2V_Signal = np.zeros([self.n_platoons, self.size_platoon - 1])

        # Compute interference
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                for k in range(len(indexes)):
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 0:
                        self.platoon_V2I_Interference[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] -
                                  self.V2I_channels_with_fastfading[indexes[k, 0] * self.size_platoon, i] +
                                  self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 1:
                        for l in range(self.size_platoon - 1):
                            self.platoon_V2V_Interference[indexes[j, 0], l] += \
                                10 ** ((power_selection[indexes[k, 0], 0] -
                                      self.V2V_channels_with_fastfading[indexes[k, 0] * self.size_platoon,
                                                                      indexes[j, 0] * self.size_platoon + (l + 1), i] +
                                      2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # Compute signals
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                if platoon_decision[indexes[j, 0], 0] == 0:
                    self.platoon_V2I_Signal[indexes[j, 0]] = 10 ** (
                        (power_selection[indexes[j, 0], 0] -
                         self.V2I_channels_with_fastfading[indexes[j, 0] * self.size_platoon, i] +
                         self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

                elif platoon_decision[indexes[j, 0], 0] == 1:
                    for l in range(self.size_platoon - 1):
                        self.platoon_V2V_Signal[indexes[j, 0], l] += 10 ** (
                            (power_selection[indexes[j, 0], 0] -
                             self.V2V_channels_with_fastfading[indexes[j, 0] * self.size_platoon,
                                                             indexes[j, 0] * self.size_platoon + (l + 1), i] +
                             2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # Calculate rates
        V2I_Rate = np.log2(1 + np.divide(self.platoon_V2I_Signal,
                                       (self.platoon_V2I_Interference + self.sig2)))
        V2V_Rate = np.log2(1 + np.divide(self.platoon_V2V_Signal,
                                       (self.platoon_V2V_Interference + self.sig2)))

        self.interplatoon_rate = V2I_Rate * self.time_fast * self.bandwidth
        self.intraplatoon_rate = (V2V_Rate * self.time_fast * self.bandwidth).min(axis=1)

        platoons_AoI = self.Age_of_Information(self.interplatoon_rate)

        # Update demands
        self.V2V_demand -= self.intraplatoon_rate
        self.V2V_demand[self.V2V_demand <= 0] = 0
        self.individual_time_limit -= self.time_fast

        # Update active links
        self.active_links[np.multiply(self.active_links, self.V2V_demand <= 0)] = 0

        reward_elements = self.intraplatoon_rate / 10000
        reward_elements[self.V2V_demand <= 0] = 1

        return platoons_AoI, self.interplatoon_rate, self.intraplatoon_rate, self.V2V_demand, reward_elements

    def Compute_Interference(self, platoons_actions):
        """Compute Interference from Environment_Platoon"""
        sub_selection = platoons_actions[:, 0].copy().astype('int').reshape(self.n_platoons, 1)
        platoon_decision = platoons_actions[:, 1].copy().astype('int').reshape(self.n_platoons, 1)
        power_selection = platoons_actions[:, 2].copy().reshape(self.n_platoons, 1)

        V2I_Interference_state = np.zeros(self.n_platoons) + self.sig2
        V2V_Interference_state = np.zeros([self.n_platoons, self.size_platoon - 1]) + self.sig2

        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                for k in range(len(indexes)):
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 0:
                        V2I_Interference_state[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] -
                                  self.V2I_channels_with_fastfading[indexes[k, 0] * self.size_platoon, i] +
                                  self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 1:
                        for l in range(self.size_platoon - 1):
                            V2V_Interference_state[indexes[j, 0], l] += \
                                10 ** ((power_selection[indexes[k, 0], 0] -
                                      self.V2V_channels_with_fastfading[indexes[k, 0] * self.size_platoon,
                                                                      indexes[j, 0] * self.size_platoon + (l + 1), i] +
                                      2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        self.V2I_Interference_all = 10 * np.log10(V2I_Interference_state)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference_state)

        for i in range(self.n_platoons):
            if platoon_decision[i, 0] == 0:
                self.Interference_all[i] = self.V2I_Interference_all[i]
            else:
                self.Interference_all[i] = np.max(self.V2V_Interference_all[i, :])

    def act_for_training(self, actions):
        """FIXED - Training method from Environment_Platoon"""
        per_user_reward = np.zeros(self.n_platoons)
        action_temp = actions.copy()
        platoon_AoI, C_rate, V_rate, Demand, elements = self.Compute_Performance_Reward_Train(action_temp)

        # Calculate V2V success
        V2V_success = 1 - np.sum(self.active_links) / self.n_platoons

        for i in range(self.n_platoons):
            per_user_reward[i] = (-4.95) * (Demand[i] / self.V2V_demand_size) - \
                               platoon_AoI[i] / 20 + (0.05) * self.Revenue_function(C_rate[i], self.V2I_min) - \
                               0.5 * math.log(max(action_temp[i, 2], 1), 5)
            

        global_reward = np.mean(per_user_reward)

        return per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success

    def act_for_testing(self, actions):
        """Testing method from Environment_Platoon"""
        action_temp = actions.copy()
        platoon_AoI, C_rate, V_rate, Demand, elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / self.n_platoons

        return platoon_AoI, C_rate, V_rate, Demand, elements, V2V_success

    def reset(self):
      """Reset environment for new episode"""
      print("Resetting environment...")
      
      # 1. Stop SUMO
      self._stop_sumo()
      
      # 2. Clear ALL data structures  
      self.vehicle_ids.clear()
      self.vehicle_positions.clear() 
      self.platoon_leaders.clear()
      self.platoon_followers.clear()
      self.platoon_counter = 0
      
      # 3. Start fresh
      self._start_sumo()
      self._create_initial_platoons()
      
      # 4. Let simulation settle
      for _ in range(15):
          if self.sumo_running:
              self.sumo_step()
      
      # 5. Reset communication arrays based on ACTUAL platoon count
      actual_platoons = len(self.platoon_leaders)
      
      # Reset arrays
      self.AoI.fill(100)
      self.V2V_demand = self.V2V_demand_size * np.ones(self.max_platoons)
      self.active_links = np.ones(self.max_platoons, dtype=bool)
      
      # Set inactive platoons to False
      if actual_platoons < self.max_platoons:
          self.active_links[actual_platoons:] = False  # Inactive platoons
      
      self.individual_time_limit = self.time_slow * np.ones(self.max_platoons)
      
      print(f"Reset complete: {actual_platoons} platoons, active_links sum: {np.sum(self.active_links)}")


   
    def close(self):
        """Clean shutdown"""
        self._stop_sumo()



    def _create_initial_platoons(self):
        """Create initial set of platoons - only 2-3 to start"""
        self.platoon_leaders = []
        self.platoon_followers = {}
        self.vehicle_ids = []
        self.platoon_counter = 0  # Track total platoons created

        routes = [
            "horizontal", "route_sn_straight", "route_we_straight", "route_ew_straight",
            "route_nw_left", "route_se_left", "route_en_left", "route_ws_left",
            "route_ne_right", "route_sw_right", "route_es_right", "route_wn_right"
        ]

        # Start with only 2-3 platoons, not all 5
        initial_platoons = 3
        for p_idx in range(initial_platoons):
            self._create_single_platoon(p_idx, routes[p_idx % len(routes)])

    def _create_single_platoon(self, platoon_id, route_id):
        """Create a single platoon with given ID and route"""
        current_time = traci.simulation.getTime()
        leader_id = f"platoon_{platoon_id}_leader"

        try:
            # Create leader
            traci.vehicle.add(leader_id, routeID=route_id, depart=str(current_time))
            traci.vehicle.setVehicleClass(leader_id, "passenger")
            traci.vehicle.setSpeedMode(leader_id, 31)
            traci.vehicle.setColor(leader_id, (255, 0, 0))  # Red for leaders

            self.platoon_leaders.append(leader_id)
            self.vehicle_ids.append(leader_id)
            self.platoon_followers[leader_id] = []

            # Create followers
            for car_idx in range(1, self.size_platoon):
                follower_id = f"platoon_{platoon_id}_car_{car_idx}"
                follower_depart_time = current_time + car_idx * 2.0

                traci.vehicle.add(follower_id, routeID=route_id, depart=str(follower_depart_time))
                traci.vehicle.setVehicleClass(follower_id, "passenger")
                traci.vehicle.setSpeedMode(follower_id, 31)
                traci.vehicle.setTau(follower_id, 0.5)
                traci.vehicle.setMinGap(follower_id, 2.0)
                traci.vehicle.setColor(follower_id, (0, 255, 0))  # Green for followers

                self.platoon_followers[leader_id].append(follower_id)
                self.vehicle_ids.append(follower_id)

            print(f"Created platoon {platoon_id} with route {route_id} at time {current_time}")
            self.platoon_counter += 1

        except traci.TraCIException as e:
            print(f"Warning: Could not create platoon {platoon_id}: {e}")

    def _generate_continuous_platoons(self):
        """IMPROVED - Generate new platoons continuously to maintain 5 max"""
        if not self.sumo_running:
            return

        current_time = traci.simulation.getTime()
        self._cleanup_exited_vehicles()

        current_active_platoons = len(self.platoon_leaders)

        # Only create new platoon if we have less than max_platoons
        if current_active_platoons < self.max_platoons:
            # Create new platoon every 10-15 seconds
            if int(current_time) % random.randint(1, 5) == 0:
                self._add_new_continuous_platoon()

    def _add_new_continuous_platoon(self):
        """Add a single new platoon with guaranteed unique IDs"""
        routes = [
            "horizontal", "route_sn_straight", "route_we_straight", "route_ew_straight",
            "route_nw_left", "route_se_left", "route_en_left", "route_ws_left",
            "route_ne_right", "route_sw_right", "route_es_right", "route_wn_right"
        ]

        current_time = traci.simulation.getTime()
        route_id = random.choice(routes)

        # FIXED: Create truly unique ID using counter + random suffix
        unique_base = f"platoon_{self.platoon_counter}_{random.randint(10000, 99999)}"
        leader_id = f"{unique_base}_leader"

        # Check if vehicle already exists
        existing_vehicles = set(traci.vehicle.getIDList())
        if leader_id in existing_vehicles:
            print(f"Vehicle {leader_id} already exists. Skipping.")
            return

        try:
            # Create leader
            traci.vehicle.add(leader_id, routeID=route_id, depart=str(current_time))
            traci.vehicle.setVehicleClass(leader_id, "passenger")
            traci.vehicle.setSpeedMode(leader_id, 31)
            traci.vehicle.setColor(leader_id, (255, 255, 0))  # Yellow

            self.platoon_leaders.append(leader_id)
            self.vehicle_ids.append(leader_id)
            self.platoon_followers[leader_id] = []

            # Create followers with unique IDs
            for car_idx in range(1, self.size_platoon):
                follower_id = f"{unique_base}_car_{car_idx}"
                follower_depart_time = current_time + car_idx * 2.0

                if follower_id not in existing_vehicles:
                    traci.vehicle.add(follower_id, routeID=route_id, depart=str(follower_depart_time))
                    traci.vehicle.setVehicleClass(follower_id, "passenger")
                    traci.vehicle.setSpeedMode(follower_id, 31)
                    traci.vehicle.setTau(follower_id, 0.5)
                    traci.vehicle.setMinGap(follower_id, 2.0)
                    traci.vehicle.setColor(follower_id, (0, 255, 255))  # Cyan

                    self.platoon_followers[leader_id].append(follower_id)
                    self.vehicle_ids.append(follower_id)

            print(f"Added continuous platoon {self.platoon_counter} at time {current_time:.1f}")
            self.platoon_counter += 1

        except traci.TraCIException as e:
            print(f"Warning: Could not create continuous platoon: {e}")

    def reset(self):
        """Reset environment for new episode"""
        self._stop_sumo()
        self._start_sumo()
        self._create_initial_platoons()  # Use new initial creation method

        # Let simulation settle with fewer steps
        for _ in range(30):
            self.sumo_step()

        # Reset communication state
        self.AoI.fill(100)
        self.V2V_demand = self.V2V_demand_size * np.ones(self.max_platoons)
        self.active_links = np.ones(self.max_platoons, dtype=bool)
        self.individual_time_limit = self.time_slow * np.ones(self.max_platoons)

    def new_random_game(self):
        """Initialize new game with continuous traffic"""
        if not self.sumo_running:
            self._start_sumo()
            self._create_initial_platoons()

            # Let simulation settle
            for _ in range(20):
                self.sumo_step()
                if not self.sumo_running:
                    break

        # Reset communication state
        self.V2V_demand = self.V2V_demand_size * np.ones(self.max_platoons)
        self.active_links = np.ones(self.max_platoons, dtype=bool)
        self.AoI = np.ones(self.max_platoons) * 100
        self.individual_time_limit = self.time_slow * np.ones(self.max_platoons)

        # Initialize channels safely
        if self.sumo_running:
            self.calculate_channel_conditions()
