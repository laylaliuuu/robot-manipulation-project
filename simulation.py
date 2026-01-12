import time
import threading
import pybullet as p
import pybullet_data
import numpy as np

from perception import PerceptionModule
from language_parser import CommandParser
from robot_controller import RobotController


# ----------------------------
# Utilities
# ----------------------------
def get_controllable_joints(robot_id):
    joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            joint_indices.append(i)
    return joint_indices


def apply_keyboard_control_with_gripper(robot_id, arm_joint_indices, gripper_left, gripper_right, step_size=0.02):
    """
    Hold keys to move continuously.
    Smaller step_size prevents jitter.
    """
    if not arm_joint_indices:
        return

    keys = p.getKeyboardEvents()

    joint_states = p.getJointStates(robot_id, arm_joint_indices)
    current_positions = [s[0] for s in joint_states]

    key_map = {
        ord('q'): (0, +1), ord('a'): (0, -1),
        ord('w'): (1, +1), ord('s'): (1, -1),
        ord('e'): (2, +1), ord('d'): (2, -1),
        ord('r'): (3, +1), ord('f'): (3, -1),
        ord('t'): (4, +1), ord('g'): (4, -1),
        ord('y'): (5, +1), ord('h'): (5, -1),
        ord('u'): (6, +1), ord('j'): (6, -1),
    }

    moved = False
    for k, (joint_offset, direction) in key_map.items():
        if k in keys and (keys[k] & p.KEY_IS_DOWN):
            if joint_offset < len(current_positions):
                current_positions[joint_offset] += direction * step_size
                moved = True

    if moved:
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=current_positions,
            forces=[87] * len(arm_joint_indices),
            positionGains=[0.06] * len(arm_joint_indices),
        )

    # Gripper keys
    if ord('o') in keys and (keys[ord('o')] & p.KEY_WAS_TRIGGERED):
        print(">>> Opening gripper...")
        target = 0.04
        p.setJointMotorControl2(robot_id, gripper_left, p.POSITION_CONTROL, targetPosition=target, force=200.0)
        p.setJointMotorControl2(robot_id, gripper_right, p.POSITION_CONTROL, targetPosition=target, force=200.0)

    if ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED):
        print(">>> Closing gripper...")
        target = 0.0
        p.setJointMotorControl2(robot_id, gripper_left, p.POSITION_CONTROL, targetPosition=target, force=200.0)
        p.setJointMotorControl2(robot_id, gripper_right, p.POSITION_CONTROL, targetPosition=target, force=200.0)


def find_gripper_joints(robot_id):
    finger_joints = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        if "finger" in name.lower():
            finger_joints.append(i)

    print("\n=== Finger-related joints found ===")
    for i in finger_joints:
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        jtype = info[2]
        lo, hi = info[8], info[9]
        print(f"  idx={i:2d} name={name:25s} type={jtype} limits=[{lo}, {hi}]")

    if len(finger_joints) < 2:
        raise RuntimeError("Could not find 2 finger joints.")

    return finger_joints[0], finger_joints[1]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ----------------------------
# Builder helpers
# ----------------------------
def spawn_object(base_name: str, pos_xyz):
    if base_name in ("red_block", "green_cube", "white_cube"):
        uid = p.loadURDF("cube_small.urdf", pos_xyz, useFixedBase=False)
        if base_name == "red_block":
            p.changeVisualShape(uid, -1, rgbaColor=[1, 0, 0, 1])
        elif base_name == "green_cube":
            p.changeVisualShape(uid, -1, rgbaColor=[0, 1, 0, 1])
        else:
            p.changeVisualShape(uid, -1, rgbaColor=[1, 1, 1, 1])
        return uid

    if base_name == "blue_sphere":
        uid = p.loadURDF("sphere_small.urdf", pos_xyz, useFixedBase=False)
        p.changeVisualShape(uid, -1, rgbaColor=[0, 0, 1, 1])
        return uid


    raise ValueError(f"Unknown object type: {base_name}")


def make_unique_name(object_map, base: str):
    if base not in object_map:
        return base
    k = 2
    while f"{base}_{k}" in object_map:
        k += 1
    return f"{base}_{k}"


# ----------------------------
# Main
# ----------------------------
def main():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        [0, 0, 0],
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True
    )
    gripper_left_index, gripper_right_index = find_gripper_joints(robot_id)

    joint_indices = get_controllable_joints(robot_id)
    arm_joint_indices = [j for j in joint_indices if j < 7]

    table_pos = [1.25, 0, 0.0]
    table_id = p.loadURDF(
        "table/table.urdf",
        table_pos,
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True
    )

    table_aabb_min, table_aabb_max = p.getAABB(table_id)
    table_top_z = table_aabb_max[2]
    z_on_table = table_top_z + 0.03

    y_min = table_aabb_min[1]
    y_max = table_aabb_max[1]
    edge_x_near_robot = table_aabb_min[0]

    p.resetDebugVisualizerCamera(
        cameraDistance=1.35,
        cameraYaw=35,
        cameraPitch=-30,
        cameraTargetPosition=[table_pos[0], table_pos[1], table_top_z]
    )

    object_map = {}
    last_spawned = []
    object_map["the_table"] = table_id 
    object_map["table"] = table_id 



    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    controller = RobotController(
    robot_id,
    gripper_left_id=gripper_left_index,
    gripper_right_id=gripper_right_index,
    perception_module=perception,
    parser=parser,
    )   

    # ------------------------------------------------------------
    # Command UI: type in terminal -> click RUN (toggle) or press R
    # ------------------------------------------------------------
    pending_cmd = {"text": ""}

    def terminal_input_loop():
        while True:
            cmd = input("\nType command (it will NOT run yet): ").strip()
            if not cmd:
                continue
            pending_cmd["text"] = cmd
            print(f"[PENDING] {cmd}")
            print("Now click RUN PENDING (toggle) in PyBullet (or press R in the sim window).")

    threading.Thread(target=terminal_input_loop, daemon=True).start()

    def run_pending():
        cmd = pending_cmd["text"]
        if not cmd:
            print("[RUN] No pending command. Type one in terminal first.")
            return
        print(f"\n[RUN] Executing: {cmd}")
        ok, msg = controller.execute_command(cmd)
        print(("✓ " if ok else "✗ ") + msg)

    # ------------------------------------------------------------
    # BUILDER MODE UI
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDER MODE + COMMAND RUN (WORKS ON YOUR PYBULLET BUILD)")
    print("=" * 60)
    print("Spawn controls (use either hotkeys or sliders + toggles):")
    print("  1/2/3/4 select object type")
    print("  B spawn  |  X delete last  |  L list objects")
    print("")
    print("Commands:")
    print("  Type a command in terminal (stored as PENDING)")
    print("  Then click RUN PENDING (toggle) OR press R in the sim window")
    print("=" * 60 + "\n")

    spawn_y_slider = p.addUserDebugParameter("Spawn Y", y_min + 0.08, y_max - 0.08, 0.0)
    depth_slider = p.addUserDebugParameter("Depth onto table", 0.06, 0.22, 0.12)

    # UI toggles: 0..1 (must flip back to 0 to trigger again)
    btn_red = p.addUserDebugParameter("SELECT: RED", 0, 1, 0)
    btn_green = p.addUserDebugParameter("SELECT: GREEN", 0, 1, 0)
    btn_blue = p.addUserDebugParameter("SELECT: BLUE", 0, 1, 0)
    btn_white = p.addUserDebugParameter("SELECT: WHITE", 0, 1, 0)

    btn_spawn = p.addUserDebugParameter("SPAWN NOW (toggle)", 0, 1, 0)
    btn_delete = p.addUserDebugParameter("DELETE LAST (toggle)", 0, 1, 0)
    btn_list = p.addUserDebugParameter("LIST OBJECTS (toggle)", 0, 1, 0)

    btn_run = p.addUserDebugParameter("RUN PENDING (toggle)", 0, 1, 0)

    selected = "red_block"

    prev = {
        "red": 0.0, "green": 0.0, "blue": 0.0, "white": 0.0,
        "spawn": 0.0, "delete": 0.0, "list": 0.0,
        "run": 0.0
    }

    # PyBullet GUI can be flaky: never crash because a param read fails
    def safe_read(param_id, label):
        try:
            return float(p.readUserDebugParameter(param_id))
        except Exception as e:
            print(f"[WARN] Failed to read {label}: {e}. Ignoring this frame.")
            return None

    def edge_click(param_id, key, label):
        """
        Trigger once on rising edge (<=0.5 -> >0.5).
        Must toggle back below 0.5 to click again.
        """
        v = safe_read(param_id, label)
        if v is None:
            return False
        was = prev[key]
        prev[key] = v
        return (was <= 0.5 and v > 0.5)

    def do_spawn():
        y = safe_read(spawn_y_slider, "Spawn Y slider")
        depth = safe_read(depth_slider, "Depth slider")
        if y is None or depth is None:
            return

        y = clamp(y, y_min + 0.08, y_max - 0.08)
        x = edge_x_near_robot + depth

        # ✅ store drop spot that matches spawn location
        controller.table_drop_xy = np.array([x, y], dtype=float)
        controller.table_drop_z_on_table = float(z_on_table)
        print("[DROP SPOT SET]", [round(x,3), round(y,3), round(z_on_table,3)])

        uname = make_unique_name(object_map, selected)
        uid = spawn_object(selected, [x, y, z_on_table])
        object_map[uname] = uid
        last_spawned.append((uname, uid))

        print(f"[SPAWN] {uname} at {[round(x,3), round(y,3), round(z_on_table,3)]}")

    def do_delete():
        if last_spawned:
            name, uid = last_spawned.pop()
            try:
                p.removeBody(uid)
            except Exception:
                pass
            object_map.pop(name, None)
            print(f"[DELETE] {name}")
        else:
            print("[DELETE] Nothing to delete.")

    def do_list():
        print("\n[OBJECTS]")
        for name, uid in object_map.items():
            pos, _ = p.getBasePositionAndOrientation(uid)
            print(f"  {name:14s} at {[round(pos[0],3), round(pos[1],3), round(pos[2],3)]}")
        print("")

    # Loop
    while True:
        keys = p.getKeyboardEvents()

        # ---- hotkeys (most reliable)
        if ord('1') in keys and (keys[ord('1')] & p.KEY_WAS_TRIGGERED):
            selected = "red_block"
            print("[SELECT] red_block")
        if ord('2') in keys and (keys[ord('2')] & p.KEY_WAS_TRIGGERED):
            selected = "green_cube"
            print("[SELECT] green_cube")
        if ord('3') in keys and (keys[ord('3')] & p.KEY_WAS_TRIGGERED):
            selected = "blue_sphere"
            print("[SELECT] blue_sphere")
        if ord('4') in keys and (keys[ord('4')] & p.KEY_WAS_TRIGGERED):
            selected = "white_cube"
            print("[SELECT] white_cube")

        if ord('b') in keys and (keys[ord('b')] & p.KEY_WAS_TRIGGERED):
            print("[KEY] B pressed -> spawn")
            do_spawn()

        if ord('x') in keys and (keys[ord('x')] & p.KEY_WAS_TRIGGERED):
            print("[KEY] X pressed -> delete last")
            do_delete()

        if ord('l') in keys and (keys[ord('l')] & p.KEY_WAS_TRIGGERED):
            print("[KEY] L pressed -> list")
            do_list()

        # Run pending command hotkey (press R in the sim window)
        if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
            print("[KEY] R pressed -> run pending command")
            run_pending()

        # ---- UI toggles (must toggle back to 0 to reuse)
        if edge_click(btn_red, "red", "SELECT RED"):
            selected = "red_block"
            print("[UI] Selected: red_block (toggle)")
        if edge_click(btn_green, "green", "SELECT GREEN"):
            selected = "green_cube"
            print("[UI] Selected: green_cube (toggle)")
        if edge_click(btn_blue, "blue", "SELECT BLUE"):
            selected = "blue_sphere"
            print("[UI] Selected: blue_sphere (toggle)")
        if edge_click(btn_white, "white", "SELECT WHITE"):
            selected = "white_cube"
            print("[UI] Selected: white_cube (toggle)")

        if edge_click(btn_spawn, "spawn", "SPAWN"):
            print("[UI] SPAWN NOW toggled -> spawn")
            do_spawn()

        if edge_click(btn_delete, "delete", "DELETE"):
            print("[UI] DELETE LAST toggled -> delete")
            do_delete()

        if edge_click(btn_list, "list", "LIST"):
            print("[UI] LIST toggled -> list")
            do_list()

        if edge_click(btn_run, "run", "RUN PENDING"):
            print("[UI] RUN PENDING toggled -> run")
            run_pending()

        # manual arm control
        apply_keyboard_control_with_gripper(robot_id, arm_joint_indices, gripper_left_index, gripper_right_index)

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
