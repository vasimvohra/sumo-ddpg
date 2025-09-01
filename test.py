import traci
import os
import sys


def run_sumo_headless():
    """Simple script to run SUMO configuration without GUI"""

    # Path to your config file (adjust path as needed)
    sumo_config = os.path.abspath("platoon.sumocfg")

    # Check if config file exists
    if not os.path.exists(sumo_config):
        print(f"Error: Config file not found at {sumo_config}")
        print("Please check the path to your platoon.sumocfg file")
        return

    # SUMO command for headless mode
    cmd = [
        "sumo",  # Use headless SUMO binary
        "-c", sumo_config,  # Configuration file
        "--no-warnings",  # Suppress warnings
        "--no-step-log",  # Suppress step logging
        "--step-length", "1.0",  # 1 second time steps
        "--end", "300"  # Run for 300 seconds (5 minutes)
    ]

    try:
        print(f"Starting SUMO with config: {sumo_config}")
        print(f"Command: {' '.join(cmd)}")

        # Start TraCI connection
        traci.start(cmd)
        print("✅ SUMO started successfully in headless mode!")

        # Run simulation
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            # Print progress every 50 steps
            if step % 50 == 0:
                vehicles = len(traci.vehicle.getIDList())
                sim_time = traci.simulation.getTime()
                print(f"Step {step}: Time {sim_time}s, Vehicles: {vehicles}")

        print(f"✅ Simulation completed after {step} steps!")

    except Exception as e:
        print(f"❌ Error running SUMO: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure SUMO is installed and 'sumo' is in your PATH")
        print("2. Check that platoon.sumocfg exists and is valid")
        print("3. Verify all referenced files (.net.xml, .rou.xml) exist")

    finally:
        try:
            traci.close()
            print("TraCI connection closed.")
        except:
            pass


if __name__ == "__main__":
    run_sumo_headless()
