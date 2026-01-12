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

        self.arm_joints = list(range(7))
        self.end_effector_link = 11
        self.gripper_force = 200.0

        # Optional "home" arm pose (stable default)
        self.home_joints = [0.0, -0.7, 0.0, -2.2, 0.0, 2.0, 0.8]
        self.table_drop_xy = np.array([0.75, 0.0], dtype=float)  # default fallback
        self.table_drop_z_on_table = 0.65     

    def wait(self, seconds=0.5):
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()

    # -----------------------------
    # IK helpers
    # -----------------------------
    def _elbow_bias_restpose(self, target_pos):
        dist = float(np.linalg.norm(np.array(target_pos[:2])))
        elbow = -2.0 - 0.6 * np.tanh(dist * 2.0)
        return [0.0, -0.9, 0.8, elbow, 0.0, 2.2, 0.9]

    def _get_joint_limits(self):
        lows, highs, ranges = [], [], []
        for j in self.arm_joints:
            info = p.getJointInfo(self.robot_id, j)
            lo, hi = info[8], info[9]
            # Guard weird URDF limits
            if lo > hi or lo < -10 or hi > 10:
                lo, hi = -2.9, 2.9
            lows.append(lo)
            highs.append(hi)
            ranges.append(hi - lo)
        return lows, highs, ranges

    def _ik(self, target_pos, use_down_orientation):
        lows, highs, ranges = self._get_joint_limits()
        rest = self._elbow_bias_restpose(target_pos)

        if use_down_orientation:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
            sol = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_pos,
                targetOrientation=target_orn,
                lowerLimits=lows,
                upperLimits=highs,
                jointRanges=ranges,
                restPoses=rest,
                maxNumIterations=160,
                residualThreshold=1e-4,
            )
        else:
            sol = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_pos,
                lowerLimits=lows,
                upperLimits=highs,
                jointRanges=ranges,
                restPoses=rest,
                maxNumIterations=120,
                residualThreshold=1e-4,
            )

        return sol[:7]

    # -----------------------------
    # NON-DESTRUCTIVE reachability check (fixes your err=0.179 issue)
    # -----------------------------
    def _fk_error_for_joints(self, joint_positions, target_pos):
        saved = [p.getJointState(self.robot_id, j)[0] for j in self.arm_joints]

        for j, q in zip(self.arm_joints, joint_positions):
            p.resetJointState(self.robot_id, j, q)

        ee_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link)[4])
        err = float(np.linalg.norm(ee_pos - np.array(target_pos)))

        for j, q in zip(self.arm_joints, saved):
            p.resetJointState(self.robot_id, j, q)

        return err

    def is_reachable(self, target_pos, use_down_orientation=False, tol=0.04):
        """
        Pure kinematic check: IK -> FK error, no stepping, no moving the sim.
        tol default 4cm (more realistic).
        """
        x, y, z = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
        r_xy = float(np.linalg.norm([x, y]))

        # Workspace guardrails (tune later)
        if r_xy < 0.12 or r_xy > 0.95:
            return False, f"Out of XY workspace (r={r_xy:.3f})"
        if z < 0.02 or z > 0.90:
            return False, f"Out of Z workspace (z={z:.3f})"

        jt = self._ik(target_pos, use_down_orientation)
        err = self._fk_error_for_joints(jt, target_pos)

        if err > tol:
            return False, f"IK could not reach pose (FK err={err:.3f}m)"
        return True, "OK"

    # -----------------------------
    # Motion
    # -----------------------------
    def move_to_position(self, target_pos, duration=2.0, use_down_orientation=False):
        joint_target = self._ik(target_pos, use_down_orientation)

        current = [p.getJointState(self.robot_id, j)[0] for j in self.arm_joints]
        steps = max(1, int(duration * 240))

        for t in range(steps):
            alpha = (t + 1) / steps
            blended = [(1 - alpha) * c + alpha * jt for c, jt in zip(current, joint_target)]
            p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joints,
                p.POSITION_CONTROL,
                targetPositions=blended,
                forces=[80] * 7,
                positionGains=[0.06] * 7
            )
            p.stepSimulation()

        return True

    def go_home(self):
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joints,
            p.POSITION_CONTROL,
            targetPositions=self.home_joints,
            forces=[80] * 7,
            positionGains=[0.06] * 7
        )
        for _ in range(240):
            p.stepSimulation()

    # -----------------------------
    # Gripper
    # -----------------------------
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

    # -----------------------------
    # Pick
    # -----------------------------
    def pick_object(self, object_name):
        obj_pos = self.perception.detect_object(object_name)
        if obj_pos is None:
            return False, f"Could not find '{object_name}'"

        obj_pos = np.array(obj_pos, dtype=float)

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
        self.open_gripper()

        # ------------------------------------------------------------
        # 1) APPROACH: flexible orientation (easier IK near reach limit)
        # ------------------------------------------------------------
        approach_pos = obj_pos + np.array([0, 0, 0.10])  # <- was 0.20, too high for table setup
        ok, reason = self.is_reachable(approach_pos, use_down_orientation=False)
        if not ok:
            return False, f"Unreachable approach pose: {reason}"
        self.move_to_position(approach_pos, duration=2.0, use_down_orientation=False)

        # ------------------------------------------------------------
        # 2) GRASP: try down orientation, fallback if it fails
        # ------------------------------------------------------------
        grasp_pos = obj_pos + np.array([0, 0, 0.015])

        ok, reason = self.is_reachable(grasp_pos, use_down_orientation=True)
        use_down = True
        if not ok:
            ok2, reason2 = self.is_reachable(grasp_pos, use_down_orientation=False)
            if not ok2:
                return False, f"Unreachable grasp pose: {reason} | fallback: {reason2}"
            use_down = False

        self.move_to_position(grasp_pos, duration=1.2, use_down_orientation=use_down)

        self.wait(0.1)
        self.close_gripper()
        self.wait(0.1)

        # ------------------------------------------------------------
        # 3) LIFT: flexible orientation again
        # ------------------------------------------------------------
        lift_pos = obj_pos + np.array([0, 0, 0.18])
        ok, reason = self.is_reachable(lift_pos, use_down_orientation=False)
        if not ok:
            return False, f"Unreachable lift pose: {reason}"
        self.move_to_position(lift_pos, duration=1.5, use_down_orientation=False)

        # Verify object lifted
        new_pos, _ = p.getBasePositionAndOrientation(obj_id)
        if new_pos[2] < obj_pos[2] + 0.05:
            return False, "Grasp failed (object did not lift)"

        self.held_object_id = obj_id
        return True, f"Successfully picked up '{object_name}'"

        # -----------------------------
        # Place
        # -----------------------------
    def place_object(self, target_name):
        self.go_home()
        target_id = self.perception.object_map.get(target_name)
        if target_id is None:
            return False, f"Target id not found for '{target_name}'"

        if self.held_object_id is None:
            return False, "No object currently held"

        # object height (so it sits on the surface, not inside it)
        o_min, o_max = p.getAABB(self.held_object_id)
        obj_half_h = 0.5 * (o_max[2] - o_min[2])

        # ----------------------------
        # TABLE: use the saved drop spot from builder spawn logic
        # ----------------------------
        if target_name in ("table", "the_table"):
            target_xy = np.array(self.table_drop_xy, dtype=float)
            target_top_z = float(self.table_drop_z_on_table)+ 0.02  # this is "table surface z"
        else:
            # generic target: use its AABB center
            t_min, t_max = p.getAABB(target_id)
            target_xy = 0.5 * (np.array(t_min[:2]) + np.array(t_max[:2]))
            target_top_z = t_max[2]

        desired_obj_pos = np.array([
            target_xy[0],
            target_xy[1],
            target_top_z + obj_half_h + 0.002
        ])

        print("[PLACE] desired_obj_pos =", [round(x, 3) for x in desired_obj_pos])

        # Approach above
        approach_pos = desired_obj_pos + np.array([0, 0, 0.10])
        ok, reason = self.is_reachable(approach_pos, use_down_orientation=False)
        if not ok:
            return False, f"Unreachable place approach: {reason}"
        self.move_to_position(approach_pos, duration=1.6, use_down_orientation=False)
        self.wait(0.05)

        # Pre-place
        pre_place = desired_obj_pos + np.array([0, 0, 0.04])
        ok, reason = self.is_reachable(pre_place, use_down_orientation=False)
        if not ok:
            return False, f"Unreachable pre-place pose: {reason}"
        self.move_to_position(pre_place, duration=1.0, use_down_orientation=False)
        self.wait(0.05)

        # Final
        ok, reason = self.is_reachable(desired_obj_pos, use_down_orientation=False)
        if not ok:
            return False, f"Unreachable place pose: {reason}"
        self.move_to_position(desired_obj_pos, duration=0.8, use_down_orientation=False)

        self.wait(0.1)
        self.open_gripper()
        self.wait(0.1)

        # Retreat
        self.move_to_position(pre_place, duration=0.7, use_down_orientation=False)
        self.move_to_position(approach_pos, duration=0.9, use_down_orientation=False)

        self.held_object_id = None
        return True, f"Successfully placed object on '{target_name}'"

    # -----------------------------
    # Command loop
    # -----------------------------
    def execute_command(self, command: str):
        self.go_home()
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

                # Optional: reset to a known good pose before retrying (helps a lot)
                self.go_home()
                self.wait(0.2)

            if not success:
                return False, f"Action '{action}' failed: {msg}"

            print(f"✓ {msg}")
        self.go_home()
        return True, "Command executed successfully"
