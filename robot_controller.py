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
        # Cache gripper max opening from joint limits (per finger upper limit)
        left_info = p.getJointInfo(self.robot_id, self.gripper_left)
        right_info = p.getJointInfo(self.robot_id, self.gripper_right)
        left_hi = left_info[9]
        right_hi = right_info[9]
        self.gripper_max_open = 2.0 * min(left_hi, right_hi)   # total opening (both fingers)



        # Robot configuration
        self.arm_joints = list(range(7))   # Joints 0-6 for Franka arm
        self.end_effector_link = 11        # Franka end-effector link index [web:33][web:37]
        self.gripper_force = 200.0         # Strong enough to actually grasp objects [web:44]
        self.move_speed = 1.0
    def wait(self, seconds=0.5):
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
    def move_to_position(self, target_pos, duration=2.0, use_down_orientation=False):
        # tool pointing “down” (you can tweak later)
        down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        if use_down_orientation:
            joint_target = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_pos,
                targetOrientation=down_orn
            )
        else:
            joint_target = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_pos
            )

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
                forces=[50] * 7,
                positionGains=[0.03] * 7
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
        """Close gripper gradually to avoid bumping the object away, then squeeze."""
        # Read current finger positions (start from wherever they are)
        left_pos = p.getJointState(self.robot_id, self.gripper_left)[0]
        right_pos = p.getJointState(self.robot_id, self.gripper_right)[0]

        # We'll close toward 0.0 in small steps
        steps = 20
        gentle_force = 60.0      # gentle close first (prevents flicking)
        squeeze_force = self.gripper_force  # 200.0 (your strong hold)
        stalled_count = 0

        for i in range(steps):
            # linearly move target toward 0
            alpha = (i + 1) / steps
            target = (1 - alpha) * max(left_pos, right_pos) + alpha * 0.0

            p.setJointMotorControl2(
                self.robot_id, self.gripper_left, p.POSITION_CONTROL,
                targetPosition=target, force=gentle_force
            )
            p.setJointMotorControl2(
                self.robot_id, self.gripper_right, p.POSITION_CONTROL,
                targetPosition=target, force=gentle_force
            )

            # simulate a bit so the fingers move
            for _ in range(12):
                p.stepSimulation()

            # Detect "stall": if fingers stop changing much, likely contacted object
            new_left = p.getJointState(self.robot_id, self.gripper_left)[0]
            new_right = p.getJointState(self.robot_id, self.gripper_right)[0]

            if abs(new_left - target) > 0.004 and abs(new_right - target) > 0.004:
                stalled_count += 1
            else:
                stalled_count = 0

            # If we stall for a couple steps, switch to squeeze force and hold
            if stalled_count >= 2:
                p.setJointMotorControl2(
                    self.robot_id, self.gripper_left, p.POSITION_CONTROL,
                    targetPosition=0.0, force=squeeze_force
                )
                p.setJointMotorControl2(
                    self.robot_id, self.gripper_right, p.POSITION_CONTROL,
                    targetPosition=0.0, force=squeeze_force
                )
                for _ in range(240):  # hold ~1 second
                    p.stepSimulation()
                return

        # If we never stalled, still finish with a strong hold at closed
        p.setJointMotorControl2(
            self.robot_id, self.gripper_left, p.POSITION_CONTROL,
            targetPosition=0.0, force=squeeze_force
        )
        p.setJointMotorControl2(
            self.robot_id, self.gripper_right, p.POSITION_CONTROL,
            targetPosition=0.0, force=squeeze_force
        )
        for _ in range(240):
            p.stepSimulation()

        """Close gripper to grasp object."""
        p.setJointMotorControl2(
            self.robot_id,
            self.gripper_left,
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

            # ---- Check if object fits in gripper opening ----
            o_min, o_max = p.getAABB(obj_id)
            dx = o_max[0] - o_min[0]
            dy = o_max[1] - o_min[1]
            obj_width = max(dx, dy)  # conservative
            if obj_width > self.gripper_max_open - 0.002:
                return False, f"Object too wide for gripper (obj ~{obj_width:.3f}m, gripper max ~{self.gripper_max_open:.3f}m)"

            print(f"Picking up '{object_name}' at {obj_pos}")
            self.wait(0.3)

            # open before approaching
            self.open_gripper()

            # Keep wrist consistent (down) during approach + grasp for better accuracy
            approach_pos = obj_pos + np.array([0, 0, 0.20])
            if not self.move_to_position(approach_pos, duration=2.0, use_down_orientation=True):
                return False, "Failed to move above object"

            grasp_pos = obj_pos + np.array([0, 0, 0.02])
            if not self.move_to_position(grasp_pos, duration=1.2, use_down_orientation=True):
                return False, "Failed to reach grasp pose"

            self.wait(0.2)

            # close gradually (prevents knocking it away)
            self.close_gripper()
            self.wait(0.2)

            # lift
            lift_pos = obj_pos + np.array([0, 0, 0.25])
            if not self.move_to_position(lift_pos, duration=2.0, use_down_orientation=True):
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
