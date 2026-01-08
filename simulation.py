import time
import pybullet as p
import pybullet_data
from perception import PerceptionModule


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
    
    p.setJointMotorControl2(robot_id, gripper_left, p.POSITION_CONTROL, 
                            targetPosition=target_pos, force=20.0)
    p.setJointMotorControl2(robot_id, gripper_right, p.POSITION_CONTROL, 
                            targetPosition=target_pos, force=20.0)
    
    for _ in range(240):
        p.stepSimulation()

def apply_keyboard_control_with_gripper(robot_id, arm_joint_indices, step_size=0.5):
    keys = p.getKeyboardEvents()

    # ARM control (your existing mapping)
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
        if k in keys and (keys[k] & p.KEY_IS_DOWN):  # hold key to keep moving
            if joint_offset < len(current_positions):
                current_positions[joint_offset] += direction * step_size

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=arm_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=current_positions,
        forces=[87]*len(arm_joint_indices)  # helps it actually move
    )

    # GRIPPER open/close
    gripper_left, gripper_right = 9, 10

    if ord('o') in keys and (keys[ord('o')] & p.KEY_WAS_TRIGGERED):
        print(">>> Opening gripper...")
        target = 0.04
        p.setJointMotorControl2(robot_id, gripper_left, p.POSITION_CONTROL, targetPosition=target, force=40)
        p.setJointMotorControl2(robot_id, gripper_right, p.POSITION_CONTROL, targetPosition=target, force=40)

    if ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED):
        print(">>> Closing gripper...")
        target = 0.0
        p.setJointMotorControl2(robot_id, gripper_left, p.POSITION_CONTROL, targetPosition=target, force=40)
        p.setJointMotorControl2(robot_id, gripper_right, p.POSITION_CONTROL, targetPosition=target, force=40)

def main():
    # Connect to physics server with GUI
    physics_client = p.connect(p.GUI)

    # Set where PyBullet looks for example assets (plane, URDFs, etc.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Basic environment: gravity + ground plane
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")

    # Load a simple robot arm (KUKA iiwa is included with PyBullet examples)
    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # robot_id = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation, useFixedBase=True)
    # robot_id = p.loadURDF("husky/husky.urdf", start_pos, start_orientation)
    robot_id = p.loadURDF("franka_panda/panda.urdf", start_pos, start_orientation, useFixedBase=True)

    red_block_id = p.loadURDF("cube_small.urdf", [0.7, 0.2, 0.05], useFixedBase=False)
    p.changeVisualShape(red_block_id, -1, rgbaColor=[1, 0, 0, 1])

    green_cube_id = p.loadURDF("cube_small.urdf", [0.3, 0.3, 0.05], useFixedBase=False)
    p.changeVisualShape(green_cube_id, -1, rgbaColor=[0, 1, 0, 1])

    blue_sphere_id = p.loadURDF("sphere_small.urdf", [0.5, -0.2, 0.05], useFixedBase=False)
    p.changeVisualShape(blue_sphere_id, -1, rgbaColor=[0, 0, 1, 1])

    for i in range(p.getNumJoints(robot_id)):
        print(p.getJointInfo(robot_id, i))


     # Add a small cube as the object to pick up
    cube_start_pos = [0.7, 0, 0.05]
    cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = p.loadURDF("cube_small.urdf",
                         cube_start_pos,
                         cube_start_orientation,
                         useFixedBase=False)
    object_map = {
    'red_block': red_block_id,
    'green_cube': green_cube_id,
    'blue_sphere': blue_sphere_id,
    'cube': cube_id
    }
    perception = PerceptionModule(robot_id, object_map)
    

    print("\n" + "="*50)
    print("PERCEPTION TEST")
    print("="*50)

    test_objects = ['red_block', 'green_cube', 'blue_sphere','cube']
    for obj in test_objects:
        pos = perception.detect_object(obj)
        print(f"{obj}: {pos}")
        assert pos is not None, f"Failed to detect {obj}"

        print("\nAll objects detected successfully!\n")

    joint_indices = get_controllable_joints(robot_id)
    arm_joint_indices = [j for j in joint_indices if j < 7]
    print("\n" + "="*50)
    print("KEYBOARD CONTROLS")
    print("="*50)
    print("\nARM CONTROL: Q/A, W/S, E/D, R/F, T/G, Y/H, U/J")
    print("GRIPPER: O (open), P (close)")
    print("="*50 + "\n")

    while True:
        apply_keyboard_control_with_gripper(robot_id, arm_joint_indices)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
if __name__ == "__main__":
    main()