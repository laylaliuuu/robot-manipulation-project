import pybullet as p
import numpy as np
from perception import PerceptionModule
from language_parser import CommandParser


class RobotController:
    """
    High-level controller that integrates perception + language + robotics.
    Controls a Franka Panda arm in PyBullet (joints 0–6 as arm, 9 & 10 as gripper).
    """

    def __init__(self, robot_id, gripper_left_id, gripper_right_id, perception_module, parser):
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
        self.gripper_max_open = 2.0 * min(left_hi, right_hi)

        self.elbow_link = 4
        self.elbow_min_z = 0.18
        self.elbow_weight = 0.6

        self.arm_joints = list(range(7))
        self.end_effector_link = 11
        self.gripper_force = 200.0
        self.move_speed = 1.0

    def wait(self, seconds=0.5):
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()

    def _elbow_bias_restpose(self, target_pos):
        dist = float(np.linalg.norm(np.array(target_pos[:2])))
        elbow = -2.0 - 0.6 * np.tanh(dist * 2.0)
        return [
            0.0,
            -0.9,
            0.8,
            elbow,
            0.0,
            2.2,
            0.9
        ]

    def move_to_position(self, target_pos, duration=2.0, use_down_orientation=False):
        if use_down_orientation:
            down_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
            joint_target = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_pos,
                targetOrientation=down_orientation
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
        p.setJointMotorControl2(self.robot_id, self.gripper_left, p.POSITION_CONTROL,
                                targetPosition=0.04, force=self.gripper_force)
        p.setJointMotorControl2(self.robot_id, self.gripper_right, p.POSITION_CONTROL,
                                targetPosition=0.04, force=self.gripper_force)
        for _ in range(240):
            p.stepSimulation()

    def close_gripper(self):
        left_pos = p.getJointState(self.robot_id, self.gripper_left)[0]
        right_pos = p.getJointState(self.robot_id, self.gripper_right)[0]
        steps = 20
        gentle_force = 60.0
        squeeze_force = self.gripper_force
        stalled_count = 0

        for i in range(steps):
            alpha = (i + 1) / steps
            target = (1 - alpha) * max(left_pos, right_pos) + alpha * 0.0

            p.setJointMotorControl2(self.robot_id, self.gripper_left, p.POSITION_CONTROL,
                                    targetPosition=target, force=gentle_force)
            p.setJointMotorControl2(self.robot_id, self.gripper_right, p.POSITION_CONTROL,
                                    targetPosition=target, force=gentle_force)

            for _ in range(12):
                p.stepSimulation()

            new_left = p.getJointState(self.robot_id, self.gripper_left)[0]
            new_right = p.getJointState(self.robot_id, self.gripper_right)[0]

            if abs(new_left - target) > 0.004 and abs(new_right - target) > 0.004:
                stalled_count += 1
            else:
                stalled_count = 0

            if stalled_count >= 2:
                break

        p.setJointMotorControl2(self.robot_id, self.gripper_left, p.POSITION_CONTROL,
                                targetPosition=0.0, force=squeeze_force)
        p.setJointMotorControl2(self.robot_id, self.gripper_right, p.POSITION_CONTROL,
                                targetPosition=0.0, force=squeeze_force)
        for _ in range(240):
            p.stepSimulation()

    # ------------------------------------------------------------
    # Key new piece: re-center after pick to reach far targets
    # ------------------------------------------------------------
    def go_to_safe_carry_pose(self):
        ee = np.array(p.getLinkState(self.robot_id, self.end_effector_link)[4])

        # lift up first
        lift = np.array([ee[0], ee[1], max(ee[2], 0.55)])
        self.move_to_position(lift, duration=1.2, use_down_orientation=False)
        self.wait(0.2)

        # recenter (tune x if you want more reach: try 0.50–0.60)
        carry = np.array([0.45, 0.0, 0.55])
        self.move_to_position(carry, duration=1.6, use_down_orientation=False)
        self.wait(0.2)


    # ------------------------------------------------------------
    # Pick (keep your “works” version: no forced down orientation)
    # ------------------------------------------------------------
    def pick_object(self, object_name):
        obj_pos = self.perception.detect_object(object_name)
        if obj_pos is None:
            return False, f"Could not find '{object_name}'"

        obj_id = self.perception.object_map.get(object_name)
        if obj_id is None:
            return False, f"Object id not found for '{object_name}'"

        # Check object fits in gripper opening
        o_min, o_max = p.getAABB(obj_id)
        dx = o_max[0] - o_min[0]
        dy = o_max[1] - o_min[1]
        obj_width = max(dx, dy)

        if obj_width > self.gripper_max_open - 0.002:
            return False, (
                f"Object too wide for gripper (obj ~{obj_width:.3f}m, "
                f"gripper max ~{self.gripper_max_open:.3f}m)"
            )

        print(f"Picking up '{object_name}' at {obj_pos}")
        self.wait(0.3)
        self.open_gripper()

        approach_pos = obj_pos + np.array([0, 0, 0.20])
        if not self.move_to_position(approach_pos, duration=2.0, use_down_orientation=True):
            return False, "Failed to move above object"

        grasp_pos = obj_pos + np.array([0, 0, 0.02])
        if not self.move_to_position(grasp_pos, duration=1.2, use_down_orientation=True):
            return False, "Failed to reach grasp pose"

        self.wait(0.2)
        self.close_gripper()
        self.wait(0.2)

        lift_pos = obj_pos + np.array([0, 0, 0.25])
        if not self.move_to_position(lift_pos, duration=2.0, use_down_orientation=True):
            return False, "Failed to lift object"

        new_pos, _ = p.getBasePositionAndOrientation(obj_id)
        if new_pos[2] < obj_pos[2] + 0.05:
            return False, "Grasp failed (object did not lift)"

        self.held_object_id = obj_id
        return True, f"Successfully picked up '{object_name}'"


    # ------------------------------------------------------------
    # Place (key change: do NOT force down orientation so it can reach far)
    # ------------------------------------------------------------
    def place_object(self, target_name):
        target_id = self.perception.object_map.get(target_name)
        if target_id is None:
            return False, f"Target id not found for '{target_name}'"

        if self.held_object_id is None:
            return False, "No object currently held"

        t_min, t_max = p.getAABB(target_id)
        target_xy = 0.5 * (np.array(t_min[:2]) + np.array(t_max[:2]))
        target_top_z = t_max[2]

        o_min, o_max = p.getAABB(self.held_object_id)
        obj_half_h = 0.5 * (o_max[2] - o_min[2])

        desired_obj_pos = np.array([
            target_xy[0],
            target_xy[1],
            target_top_z + obj_half_h + 0.002
        ])

        # APPROACH: higher + flexible orientation
        approach_pos = desired_obj_pos + np.array([0, 0, 0.28])
        if not self.move_to_position(approach_pos, duration=2.2, use_down_orientation=False):
            return False, "Failed to move above target"
        self.wait(0.15)

        # DESCEND: still flexible
        pre_place = desired_obj_pos + np.array([0, 0, 0.07])
        if not self.move_to_position(pre_place, duration=1.4, use_down_orientation=False):
            return False, "Failed to descend near target"
        self.wait(0.15)

        # Optional XY micro-correction (still flexible orientation)
        gain = 0.6
        for _ in range(12):
            obj_now, _ = p.getBasePositionAndOrientation(self.held_object_id)
            obj_xy = np.array(obj_now[:2])
            err = target_xy - obj_xy
            if np.linalg.norm(err) < 0.0015:
                break

            ee_now = np.array(p.getLinkState(self.robot_id, self.end_effector_link)[4])
            ee_target = ee_now + np.array([gain * err[0], gain * err[1], 0.0])
            self.move_to_position(ee_target, duration=0.30, use_down_orientation=False)
            self.wait(0.10)

        # FINAL DESCEND: flexible (this is what preserves reach)
        if not self.move_to_position(desired_obj_pos, duration=1.0, use_down_orientation=False):
            return False, "Failed to reach place pose"

        self.wait(1.0)
        self.open_gripper()
        self.wait(1.0)

        # retreat
        if not self.move_to_position(approach_pos, duration=1.2, use_down_orientation=False):
            return False, "Failed to retreat"

        self.held_object_id = None
        return True, f"Successfully placed object on '{target_name}'"


    def execute_command(self, command: str):
        print(f"\n>>> Executing: '{command}'")

        command = self.parser.normalize_command(command)
        actions = self.parser.parse(command)
        if actions is None:
            return False, "Could not understand command"

        print(f"Parsed actions: {actions}")

        for action, obj, target in actions:
            tries = 1
            if action == "pick":
                tries = 5
            elif action == "place":
                tries = 3

            success, msg = False, ""
            for attempt in range(tries):
                if tries > 1:
                    print(f" → {action} attempt {attempt+1}/{tries}")

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

                self.wait(0.4)

            if not success:
                return False, f"Action '{action}' failed: {msg}"

            print(f"✓ {msg}")

        return True, "Command executed successfully"






