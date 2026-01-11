import time
import pybullet as p
import pybullet_data
import numpy as np
from perception import PerceptionModule
from language_parser import CommandParser
from robot_controller import RobotController


def get_controllable_joints(robot_id):
    num_joints = p.getNumJoints(robot_id)
    joint_indices = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_type = info[2]
        # 0 = REVOLUTE, 1 = PRISMATIC, 2 = SPHERICAL, etc.
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            joint_indices.append(i)
    return joint_indices


def apply_keyboard_control(robot_id, joint_indices, step_size=0.5):
    keys = p.getKeyboardEvents()

    # Get current joint states
    joint_states = p.getJointStates(robot_id, joint_indices)
    current_positions = [s[0] for s in joint_states]

    # Map keys to joint index + direction
    key_map = {
        ord('q'): (0, +1),
        ord('a'): (0, -1),
        ord('w'): (1, +1),
        ord('s'): (1, -1),
        ord('e'): (2, +1),
        ord('d'): (2, -1),
        ord('r'): (3, +1),
        ord('f'): (3, -1),
        ord('t'): (4, +1),
        ord('g'): (4, -1),
        ord('y'): (5, +1),
        ord('h'): (5, -1),
        ord('u'): (6, +1),
        ord('j'): (6, -1),
    }

    # Update positions based on keys pressed
    for k, (joint_offset, direction) in key_map.items():
        if k in keys and keys[k] & p.KEY_WAS_TRIGGERED:
            if joint_offset < len(current_positions):
                current_positions[joint_offset] += direction * step_size

    # Send updated target positions to motors
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=current_positions
    )


def control_gripper(robot_id, action):
    gripper_left = 9
    gripper_right = 10

    if action == 'open':
        target_pos = 0.04
    elif action == 'close':
        target_pos = 0.0
    else:
        return

    # Use same strong force as RobotController (200N)
    p.setJointMotorControl2(
        robot_id, gripper_left, p.POSITION_CONTROL,
        targetPosition=target_pos, force=200.0
    )
    p.setJointMotorControl2(
        robot_id, gripper_right, p.POSITION_CONTROL,
        targetPosition=target_pos, force=200.0
    )

    for _ in range(240):
        p.stepSimulation()


def apply_keyboard_control_with_gripper(robot_id, arm_joint_indices, gripper_left, gripper_right, step_size=0.5):
    # Safety: if no valid joints, do nothing
    if not arm_joint_indices:
        return

    keys = p.getKeyboardEvents()

    # ARM control
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

    for k, (joint_offset, direction) in key_map.items():
        if k in keys and (keys[k] & p.KEY_IS_DOWN):
            if joint_offset < len(current_positions):
                current_positions[joint_offset] += direction * step_size

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=arm_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=current_positions,
        forces=[87] * len(arm_joint_indices)
    )

    # GRIPPER open/close
    gripper_left, gripper_right = 9, 10

    if ord('o') in keys and (keys[ord('o')] & p.KEY_WAS_TRIGGERED):
        print(">>> Opening gripper...")
        target = 0.04
        p.setJointMotorControl2(
            robot_id, gripper_left, p.POSITION_CONTROL,
            targetPosition=target, force=200.0
        )
        p.setJointMotorControl2(
            robot_id, gripper_right, p.POSITION_CONTROL,
            targetPosition=target, force=200.0
        )

    if ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED):
        print(">>> Closing gripper...")
        target = 0.0
        p.setJointMotorControl2(
            robot_id, gripper_left, p.POSITION_CONTROL,
            targetPosition=target, force=200.0
        )
        p.setJointMotorControl2(
            robot_id, gripper_right, p.POSITION_CONTROL,
            targetPosition=target, force=200.0
        )

def find_gripper_joints(robot_id):
    """Return (left_finger_joint_index, right_finger_joint_index) by name."""
    finger_joints = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        jtype = info[2]
        # Most panda finger joints are PRISMATIC and contain 'finger'
        if "finger" in name.lower():
            finger_joints.append((i, name, jtype, info[8], info[9]))

    print("\n=== Finger-related joints found ===")
    for (i, name, jtype, lo, hi) in finger_joints:
        print(f"  idx={i:2d} name={name:25s} type={jtype} limits=[{lo}, {hi}]")

    if len(finger_joints) < 2:
        raise RuntimeError("Could not find 2 finger joints. Check URDF/joint names printed above.")

    # Typically there are exactly two: panda_finger_joint1 and panda_finger_joint2
    left = finger_joints[0][0]
    right = finger_joints[1][0]
    return left, right

def main():
    # Connect to physics server with GUI
    physics_client = p.connect(p.GUI)

    # Set where PyBullet looks for example assets (plane, URDFs, etc.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Basic environment: gravity + ground plane
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")

    # Load Franka Panda robot arm
    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        start_pos,
        start_orientation,
        useFixedBase=True
    )
    gripper_left_index, gripper_right_index = find_gripper_joints(robot_id)


    # Colored objects
    red_block_id = p.loadURDF("cube_small.urdf", [0.7, 0.2, 0.05], useFixedBase=False)
    p.changeVisualShape(red_block_id, -1, rgbaColor=[1, 0, 0, 1])

    green_cube_id = p.loadURDF("cube_small.urdf", [0.3, 0.3, 0.05], useFixedBase=False)
    p.changeVisualShape(green_cube_id, -1, rgbaColor=[0, 1, 0, 1])

    blue_sphere_id = p.loadURDF(
        "sphere_small.urdf", [0.5, -0.2, 0.05], useFixedBase=False
    )
    p.changeVisualShape(blue_sphere_id, -1, rgbaColor=[0, 0, 1, 1])

    # Extra cube (kept for completeness; used as 'cube' in object_map)
    cube_start_pos = [0.7, 0, 0.05]
    cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = p.loadURDF(
        "cube_small.urdf",
        cube_start_pos,
        cube_start_orientation,
        useFixedBase=False
    )

    # Map from names to object IDs (for perception)
    object_map = {
        'red_block': red_block_id,
        'green_cube': green_cube_id,
        'blue_sphere': blue_sphere_id,
        'white_cube': cube_id
    }

    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()

    controller = RobotController(
        robot_id,
        gripper_left_id=gripper_left_index,
        gripper_right_id=gripper_right_index,
        perception_module=perception,
        parser=parser
)

    print("\n" + "=" * 50)
    print("PERCEPTION TEST")
    print("=" * 50)

    test_objects = ['red_block', 'green_cube', 'blue_sphere', 'white_cube']
    for obj in test_objects:
        pos = perception.detect_object(obj)
        print(f"{obj}: {pos}")
        assert pos is not None, f"Failed to detect {obj}"

    print("\nAll objects detected successfully!\n")

    # Keyboard joint indices
    joint_indices = get_controllable_joints(robot_id)
    arm_joint_indices = [j for j in joint_indices if j < 7]

    print("\n" + "=" * 50)
    print("KEYBOARD CONTROLS")
    print("=" * 50)
    print("\nARM CONTROL: Q/A, W/S, E/D, R/F, T/G, Y/H, U/J")
    print("GRIPPER: O (open), P (close)")
    print("=" * 50 + "\n")

    # Test full pipeline
    print("\nStep 3: Testing Full Pipeline")
    print("-" * 60)

    demo_commands = [
        "place the red block on the white cube",
        "pick up the green cube",
        "place the green cube on the red cube",
        "place the blue cube on green cube"
    ]

    for cmd in demo_commands:
        success, msg = controller.execute_command(cmd)
        status = "✓" if success else "✗"
        print(f"  {status} {msg}")
        time.sleep(1.0)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60 + "\n")

    # Keep simulation running for manual keyboard control
    if not arm_joint_indices:
        print("No arm joints found; skipping keyboard control loop.")
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    else:
        while True:
            apply_keyboard_control_with_gripper(robot_id, arm_joint_indices, gripper_left_index, gripper_right_index)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

if __name__ == "__main__":
    main()
