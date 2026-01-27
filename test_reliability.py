
import numpy as np
import pybullet as p
import pybullet_data
import time
from robot_controller import RobotController
from language_parser import CommandParser
from perception import PerceptionModule

def jitter_xy(base_xyz, xy_noise=0.05):
    """
    Returns a new xyz where x,y are randomly jittered.
    Keeps it within reasonable reach (0.45 to 0.75m from base).
    """
    x, y, z = base_xyz
    x_new = x + np.random.uniform(-xy_noise, xy_noise)
    y_new = y + np.random.uniform(-xy_noise, xy_noise)

    # Clamp to safe reliable region
    # < 0.35 is too close (collision with self/base)
    # > 0.85 is too far (singularity)
    x_new = max(0.40, min(0.78, x_new))
    y_new = max(-0.35, min(0.35, y_new))
    return [x_new, y_new, float(z)]

def setup_world(hard_mode=False):
    if p.isConnected():
        p.resetSimulation()
    else:
        p.connect(p.GUI)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", [1.25, 0, 0], useFixedBase=True)

    # Calculate safe table surface z
    t_min, t_max = p.getAABB(table_id)
    table_z = t_max[2]

    # Spawn configuration
    if hard_mode:
        # Hard: random spots on table
        white_pos = jitter_xy([0.65, -0.20, table_z + 0.03], 0.15)
        green_pos = jitter_xy([0.65, 0.20, table_z + 0.03], 0.15)
    else:
        # Easy: somewhat fixed reachable spots
        white_pos = jitter_xy([0.60, -0.15, table_z + 0.03], 0.05)
        green_pos = jitter_xy([0.60, 0.15, table_z + 0.03], 0.05)

    white_id = p.loadURDF("cube_small.urdf", white_pos, useFixedBase=False)
    p.changeVisualShape(white_id, -1, rgbaColor=[1, 1, 1, 1])

    green_id = p.loadURDF("cube_small.urdf", green_pos, useFixedBase=False)
    p.changeVisualShape(green_id, -1, rgbaColor=[0, 1, 0, 1])

    # Let physics settle
    for _ in range(50):
        p.stepSimulation()

    object_map = {
        "white_cube": white_id,
        "green_cube": green_id,
        "table": table_id,
        "the_table": table_id
    }
    
    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    controller = RobotController(robot_id, 9, 10, perception, parser)

    # Heuristic drop zone if needed (between the two spawn zones)
    controller.table_drop_xy = np.array([0.62, 0.0], dtype=float)
    controller.table_top_z = float(table_z)
    
    return controller

def run_trials(cmd, trials=10, hard_mode=False):
    print(f"\n[{cmd.upper()}] Starting {trials} trials (Hard Mode: {hard_mode})...")
    successes = 0
    
    for i in range(trials):
        try:
            controller = setup_world(hard_mode=hard_mode)
            
            # Additional settle
            controller.wait(0.5)
            
            print(f"\n--- Trial {i+1}/{trials} ---")
            ok, msg = controller.execute_command(cmd)
            
            if ok:
                print(f"✅ Success: {msg}")
                successes += 1
            else:
                print(f"❌ Failure: {msg}")
                
        except Exception as e:
            print(f"❌ Critical Error in trial {i+1}: {e}")
            import traceback
            traceback.print_exc()
            # Try to reconnect if physics died
            if not p.isConnected():
                p.connect(p.GUI)

    print(f"\nRESULTS for '{cmd}': {successes}/{trials} ({int(successes/trials*100)}%)")
    return successes

def main():
    # Ensure GUI is up
    if not p.isConnected():
        p.connect(p.GUI)
    
    # 1. Basic Reliability (Green -> White)
    run_trials("put green cube on top of white cube", trials=5, hard_mode=False)

    # 2. The User's Requested Test (White -> Green) with valid jitter
    run_trials("put white cube on top of green cube", trials=5, hard_mode=True)

    # Keep window open briefly
    print("\nTests complete. Closing in 2 seconds...")
    time.sleep(2.0)
    p.disconnect()

if __name__ == "__main__":
    main()
