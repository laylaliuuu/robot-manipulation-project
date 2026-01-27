import numpy as np
import pybullet as p
import pybullet_data
import time
from robot_controller import RobotController
from language_parser import CommandParser
from perception import PerceptionModule

# Configuration
USE_GUI = True  # Set to True if you want to see the simulation
TRIALS_PER_SCENARIO = 3

def setup_scenario(green_loc_type, white_loc_type):
    if p.isConnected():
        p.resetSimulation()
    else:
        p.connect(p.GUI if USE_GUI else p.DIRECT)
    
    # Always set path after reset
    data_path = pybullet_data.getDataPath()
    # print(f"DEBUG: pybullet data path: {data_path}")
    p.setAdditionalSearchPath(data_path)
    p.setGravity(0, 0, -9.81)
    
    try:
        p.loadURDF("plane.urdf")
    except Exception as e:
        print(f"Error loading plane.urdf. Path: {data_path}")
        raise e
    
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", [1.25, 0, 0], useFixedBase=True)
    
    # Get table height
    t_min, t_max = p.getAABB(table_id)
    table_z = t_max[2]
    
    # Define zones
    def get_pos(loc_type, offset_y=0):
        if loc_type == "table":
            # On table: x=0.65, y within [-0.2, 0.2]
            return [0.65, offset_y, table_z + 0.03]
        elif loc_type == "floor":
            # On floor: x=0.45, y within [-0.2, 0.2]
            # Must be safe from robot base (x=0) and table legs (x=0.5?)
            # Robot min reach ~0.35. Table starts ~0.5?
            # Let's verify reachability or just try.
            return [0.45, offset_y, 0.03]
        else:
            raise ValueError(f"Unknown loc_type: {loc_type}")

    green_pos = get_pos(green_loc_type, 0.2)
    white_pos = get_pos(white_loc_type, -0.2)
    
    # Add small jitter
    green_pos[0] += np.random.uniform(-0.02, 0.02)
    green_pos[1] += np.random.uniform(-0.02, 0.02)
    white_pos[0] += np.random.uniform(-0.02, 0.02)
    white_pos[1] += np.random.uniform(-0.02, 0.02)
    
    green_id = p.loadURDF("cube_small.urdf", green_pos, useFixedBase=False)
    p.changeVisualShape(green_id, -1, rgbaColor=[0, 1, 0, 1])
    
    white_id = p.loadURDF("cube_small.urdf", white_pos, useFixedBase=False)
    p.changeVisualShape(white_id, -1, rgbaColor=[1, 1, 1, 1])
    
    # Settle
    for _ in range(50): p.stepSimulation()
    
    object_map = {
        "white_cube": white_id,
        "green_cube": green_id,
        "table": table_id,
        "the_table": table_id
    }
    
    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    controller = RobotController(robot_id, 9, 10, perception, parser)
    controller.table_top_z = float(table_z)
    
    return controller

def run_test(scenario_name, green_loc, white_loc):
    print(f"\n[{scenario_name}] Starting {TRIALS_PER_SCENARIO} trials...")
    print(f"  Configuration: Green ({green_loc}) - White ({white_loc})")
    
    successes = 0
    for i in range(TRIALS_PER_SCENARIO):
        try:
            controller = setup_scenario(green_loc, white_loc)
            controller.wait(0.5)
            
            cmd = "put green cube on top of white cube"
            # print(f"  Trial {i+1}: Executing '{cmd}'...")
            
            ok, msg = controller.execute_command(cmd)
            
            if ok:
                print(f"  Trial {i+1}: Success")
                successes += 1
            else:
                print(f"  Trial {i+1}: Fail - {msg}")
                
        except Exception as e:
            print(f"  Trial {i+1}: Error - {e}")
            import traceback
            traceback.print_exc()

    print(f"RESULTS for {scenario_name}: {successes}/{TRIALS_PER_SCENARIO}")
    return successes

def main():
    print("=== Testing All Reliability Scenarios ===")
    
    # 1. Table Stack (Standard)
    # Green on Table, White on Table -> Stack on Table
    run_test("Table Stack", "table", "table")
    
    # 2. Floor Stack
    # Green on Floor, White on Floor -> Stack on Floor
    run_test("Floor Stack", "floor", "floor")
    
    # 3. Table -> Floor
    # Green on Table, White on Floor -> Pick from Table, Place on Floor
    run_test("Table to Floor", "table", "floor")
    
    # 4. Floor -> Table
    # Green on Floor, White on Table -> Pick from Floor, Place on Table
    run_test("Floor to Table", "floor", "table")
    
    if p.isConnected():
        p.disconnect()

if __name__ == "__main__":
    main()
