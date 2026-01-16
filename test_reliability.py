import numpy as np
import pybullet as p
import pybullet_data
from robot_controller import RobotController
from language_parser import CommandParser
from perception import PerceptionModule

def jitter_xy(base_xyz, xy_noise=0.08):
    """
    Returns a new xyz where x,y are randomly jittered by ±xy_noise.
    Keep z the same.
    """
    x, y, z = base_xyz
    x = float(x + np.random.uniform(-xy_noise, xy_noise))
    y = float(y + np.random.uniform(-xy_noise, xy_noise))

    # Clamp to a safe reachable region for Panda in your scene
    x = max(0.35, min(0.80, x))
    y = max(-0.35, min(0.35, y))
    return [x, y, float(z)]

def setup_world():
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", [1.25,0,0], useFixedBase=True)

    white_start = jitter_xy([0.62, 0.06, 0.66], xy_noise=0.10)
    green_start = jitter_xy([0.62, -0.06, 0.66], xy_noise=0.10)

    white_id = p.loadURDF("cube_small.urdf", white_start, useFixedBase=False)
    p.changeVisualShape(white_id, -1, rgbaColor=[1,1,1,1])

    green_id = p.loadURDF("cube_small.urdf", green_start, useFixedBase=False)
    p.changeVisualShape(green_id, -1, rgbaColor=[0,1,0,1])


    object_map = {"white_cube": white_id, "green_cube": green_id, "table": table_id, "the_table": table_id}
    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    controller = RobotController(robot_id, 9, 10, perception, parser)

    controller.table_drop_xy = np.array([0.62, 0.0], dtype=float)
    controller.table_top_z = None
    return controller

def run_trials(cmd, trials=10):
    success = 0
    for i in range(trials):
        controller = setup_world()
        ok, msg = controller.execute_command(cmd)
        print(f"[{i+1}/{trials}] {ok} - {msg}")
        if ok:
            success += 1
        for _ in range(240):
            p.stepSimulation()
    return success

def main():
    p.connect(p.GUI)

    # s1 = run_trials("put white cube on the table", 10)
    # print("put-on-table success:", s1, "/ 10")

    s2 = run_trials("put green cube on top of white cube", 10)
    print("stack success:", s2, "/ 10")

    p.disconnect()

if __name__ == "__main__":
    main()
