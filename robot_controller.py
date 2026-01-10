import pybullet as p
import numpy as np
from perception import PerceptionModule
from language_parser import CommandParser



class RobotController:
    """
    High-level controller that integrates perception + language + robotics.
    Controls a Franka Panda arm in PyBullet (joints 0–6 as arm, 9 & 10 as gripper).
    """

    def __init__(self, robot_id, gripper_left_id, gripper_right_id,
                 perception_module, parser):
        self.robot_id = robot_id
        self.gripper_left = gripper_left_id
        self.gripper_right = gripper_right_id
        self.perception = perception_module
        self.parser = parser
        self.held_object_id = None


        # Robot configuration
        self.arm_joints = list(range(7))   # Joints 0-6 for Franka arm
        self.end_effector_link = 11        # Franka end-effector link index [web:33][web:37]
        self.gripper_force = 200.0         # Strong enough to actually grasp objects [web:44]
        self.move_speed = 1.0
    def wait(self, seconds=0.5):
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
    def move_to_position(self, target_pos, duration=2.0):
        joint_target = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link,
            targetPosition=target_pos
        )

        # current joint positions
        current = [p.getJointState(self.robot_id, j)[0] for j in self.arm_joints]

        steps = max(1, int(duration * 240))
        for t in range(steps):
            alpha = (t + 1) / steps
            blended = [(1 - alpha) * c + alpha * jt for c, jt in zip(current, joint_target[:7])]

            p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joints,
                p.POSITION_CONTROL,
                targetPositions=blended,
                forces=[50] * 7,          # lower force helps stability
                positionGains=[0.03] * 7  # softer controller (less violent)
            )
            p.stepSimulation()

        return True

    def open_gripper(self):
        """Open gripper by moving fingers apart."""
        p.setJointMotorControl2(
            self.robot_id,
            self.gripper_left,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=self.gripper_force
        )
        p.setJointMotorControl2(
            self.robot_id,
            self.gripper_right,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=self.gripper_force
        )

        # Let gripper move
        for _ in range(240):  # ~1 second at 240 Hz
            p.stepSimulation()

    def close_gripper(self):
        """Close gripper to grasp object."""
        p.setJointMotorControl2(
            self.robot_id,
            self.gripper_left,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=self.gripper_force
        )
        p.setJointMotorControl2(
            self.robot_id,
            self.gripper_right,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=self.gripper_force
        )

        # Let gripper close
        for _ in range(240):  # ~1 second at 240 Hz
            p.stepSimulation()

    def pick_object(self, object_name):
            obj_pos = self.perception.detect_object(object_name)
            if obj_pos is None:
                return False, f"Could not find '{object_name}'"

            obj_id = self.perception.object_map.get(object_name)
            if obj_id is None:
                return False, f"Object id not found for '{object_name}'"

            print(f"Picking up '{object_name}' at {obj_pos}")
            self.wait(0.5)

            # open before approaching (important)
            self.open_gripper()

            # move above
            approach_pos = obj_pos + np.array([0, 0, 0.2])
            if not self.move_to_position(approach_pos, duration=2.0):
                return False, "Failed to move above object"

            grasp_pos = obj_pos + np.array([0, 0, 0.02])  # 2cm above base center
            if not self.move_to_position(grasp_pos, duration=1.5):
                return False, "Failed to reach grasp pose"
            self.wait(0.3)

            # close gripper
            self.close_gripper()

            # lift
            lift_pos = obj_pos + np.array([0, 0, 0.25])
            if not self.move_to_position(lift_pos, duration=2.0):
                return False, "Failed to lift object"

            # verify object moved upward
            new_pos, _ = p.getBasePositionAndOrientation(obj_id)
            if new_pos[2] < obj_pos[2] + 0.05:
                return False, "Grasp failed (object did not lift)"

            self.held_object_id = obj_id
            return True, f"Successfully picked up '{object_name}'"

    def place_object(self, target_name):
        target_pos = self.perception.detect_object(target_name)
        if target_pos is None:
            return False, f"Could not find target '{target_name}'"

        target_id = self.perception.object_map.get(target_name)
        if target_id is None:
            return False, f"Target id not found for '{target_name}'"

        if self.held_object_id is None:
            return False, "No object currently held"

        print(f"Placing object on '{target_name}' at {target_pos}")

        # --- compute "top of target" and "half height of held object" using AABB ---
        t_min, t_max = p.getAABB(target_id)
        o_min, o_max = p.getAABB(self.held_object_id)

        target_top_z = t_max[2]
        obj_half_h = 0.5 * (o_max[2] - o_min[2])

        # desired place position: center aligned in XY, Z sits on top + tiny epsilon
        place_pos = np.array([target_pos[0], target_pos[1], target_top_z + obj_half_h + 0.002])

        # --- approach above ---
        approach_pos = place_pos + np.array([0, 0, 0.15])
        if not self.move_to_position(approach_pos, duration=2.0):
            return False, "Failed to move above target"

        # --- descend close to placement height (don’t drop from above) ---
        pre_place = place_pos + np.array([0, 0, 0.02])
        if not self.move_to_position(pre_place, duration=1.0):
            return False, "Failed to descend near placement"

        if not self.move_to_position(place_pos, duration=0.8):
            return False, "Failed to reach placement pose"

        # settle, then release
        self.wait(0.5)
        self.open_gripper()
        self.wait(0.5)

        # retreat up (prevents knocking it)
        if not self.move_to_position(approach_pos, duration=1.2):
            return False, "Failed to retreat"

        self.held_object_id = None
        return True, f"Successfully placed object on '{target_name}'"

    def execute_command(self, command: str):
            """
            Execute a natural language command end-to-end.

            Returns: (success: bool, message: str)
            """
            print(f"\n>>> Executing: '{command}'")

            # Normalize command
            command = self.parser.normalize_command(command)

            # Parse command
            actions = self.parser.parse(command)
            if actions is None:
                return False, "Could not understand command"

            print(f"Parsed actions: {actions}")

            # Execute each action
            for action, obj, target in actions:

                # how many retries per action
                tries = 1
                if action == "pick":
                    tries = 5
                elif action == "place":
                    tries = 3

                success, msg = False, ""

                for attempt in range(tries):
                    if tries > 1:
                        print(f"  → {action} attempt {attempt+1}/{tries}")

                    if action == "pick":
                        success, msg = self.pick_object(obj)

                    elif action == "place":
                        success, msg = self.place_object(obj)

                    elif action == "open_gripper":
                        self.open_gripper()
                        success, msg = True, "Opened gripper"

                    elif action == "close_gripper":
                        self.close_gripper()
                        success, msg = True, "Closed gripper"

                    else:
                        success, msg = False, f"Unknown action: {action}"

                    if success:
                        break

                    # small settle before retry (helps physics a lot)
                    self.wait(0.4)

                if not success:
                    return False, f"Action '{action}' failed: {msg}"

                print(f"✓ {msg}")

            return True, "Command executed successfully"
