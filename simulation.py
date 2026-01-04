import time
import pybullet as p
import pybullet_data

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
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation, useFixedBase=True)

     # Add a small cube as the object to pick up
    cube_start_pos = [0.7, 0, 0.05]
    cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = p.loadURDF("cube_small.urdf",
                         cube_start_pos,
                         cube_start_orientation,
                         useFixedBase=False)
    joint_indices = get_controllable_joints(robot_id)

    while True:
        apply_keyboard_control(robot_id, joint_indices)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
if __name__ == "__main__":
    main()