import pybullet as p
import numpy as np
from perception import PerceptionModule
from language_parser import CommandParser
import time
import json
import os
import uuid


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
        self.table_top_z = None
        # -----------------------------
        # Logging (JSONL)
        # -----------------------------
        self.log_path = "attempts.jsonl"

        # -----------------------------
        # Settle + success verification
        # -----------------------------
        self.settle_seconds = 0.60      # wait after release
        self.settle_check_dt = 0.10     # sampling interval (0.10s => 6 samples over 0.6s)
        self.max_xy_drift = 0.05        # allowed XY drift during settle window
        self.max_tilt_deg_table = 35.0  # table can be tilted but not insane
        self.max_tilt_deg_stack = 25.0  # stacking should be tighter

        # -----------------------------
        # ML Mode Toggle
        # -----------------------------
        self.use_ml = False  # Set to True to use ML-guided selection
        self.ml_model = None
        self.ml_model_path = "models/place_rf.joblib"
        
        # Try to load ML model if it exists
        if os.path.exists(self.ml_model_path):
            try:
                import joblib
                model_data = joblib.load(self.ml_model_path)
                if isinstance(model_data, dict):
                    self.ml_model = model_data['model']
                else:
                    self.ml_model = model_data
                print(f"[ML] Model loaded from {self.ml_model_path}")
            except Exception as e:
                print(f"[ML] Failed to load model: {e}")
                self.ml_model = None

        self.run_id = str(uuid.uuid4())[:8]
        self.episode_id = 0
        self.step_id = 0
        self.scene_seed = None
    
    def enable_ml_mode(self, enabled=True):
        """Enable or disable ML-guided candidate selection."""
        if enabled and self.ml_model is None:
            print("[ML] Cannot enable ML mode: model not loaded")
            return False
        self.use_ml = enabled
        mode = "ML-guided" if enabled else "Heuristic-only"
        print(f"[ML] Mode set to: {mode}")
        return True

    def set_episode(self, episode_id: int, scene_seed: int):
        self.episode_id = int(episode_id)
        self.scene_seed = int(scene_seed)
        self.step_id = 0
    def _log_attempt(self, row: dict):
        row = dict(row)
        row.setdefault("ts", time.time())
        row.setdefault("run_id", self.run_id)
        row.setdefault("episode_id", self.episode_id)
        row.setdefault("scene_seed", self.scene_seed)
        row.setdefault("step_id", self.step_id)
        self.step_id += 1

        # Write JSONL
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            # Don't crash the robot if logging fails
            print("[LOGGING ERROR]", e)

    def generate_table_candidates(self, held_id, table_id, num=16):
        t_min, t_max = p.getAABB(table_id)
        table_top_z = float(t_max[2])

        o_min, o_max = p.getAABB(held_id)
        obj_half_h = 0.5 * float(o_max[2] - o_min[2])

        # try to keep y near your drop lane; clamp to safe band
        y0 = float(getattr(self, "table_drop_xy", [0.7, 0.0])[1])
        y0 = max(-0.25, min(0.25, y0))

        xs = np.linspace(0.82, 0.62, num)  # reachable band
        cands = []
        for x in xs:
            cands.append(np.array([float(x), float(y0), float(table_top_z + obj_half_h + 0.010)], dtype=float))
        return cands


    def generate_stack_candidates(self, held_id, target_id, grid=5, step=0.012):
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        target_xy0 = np.array(target_pos[:2], dtype=float)

        t_min, t_max = p.getAABB(target_id)
        target_top_z = float(t_max[2])

        o_min, o_max = p.getAABB(held_id)
        obj_half_h = 0.5 * float(o_max[2] - o_min[2])

        z = target_top_z + obj_half_h + 0.014

        offsets = []
        mid = grid // 2
        for i in range(grid):
            for j in range(grid):
                dx = (i - mid) * step
                dy = (j - mid) * step
                dist = abs(dx) + abs(dy)
                offsets.append((dist, dx, dy))
        offsets.sort(key=lambda x: x[0])

        cands = []
        for _, dx, dy in offsets:
            cands.append(np.array([target_xy0[0] + dx, target_xy0[1] + dy, z], dtype=float))
        return cands
    
    def score_candidate_heuristic(self, cand_xyz):
        # reachable approach + reachable final (use your existing reachability function)
        approach = cand_xyz + np.array([0, 0, 0.14], dtype=float)

        ok1, _ = self.is_reachable(approach, use_down_orientation=False)
        ok2, _ = self.is_reachable(cand_xyz, use_down_orientation=False)
        if not (ok1 and ok2):
            return -1e9

        jt = self._ik(cand_xyz, use_down_orientation=False)
        err = self._fk_error_for_joints(jt, cand_xyz)  # smaller better
        return -float(err)


    def choose_candidate(self, candidates, explore_p=0.20, top_k=5):
        """
        Returns: (chosen_xyz, meta_dict)
        explore_p: fraction of the time we pick randomly among top_k (for data diversity)
        
        If self.use_ml is True and model is loaded, uses ML-guided selection.
        Otherwise uses heuristic-only selection.
        """
        # Score with heuristic (always needed)
        scored = [(self.score_candidate_heuristic(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        # filter out totally impossible (-1e9)
        valid = [(s, c) for (s, c) in scored if s > -1e8]
        if not valid:
            return None, {"reason": "no_reachable_candidates", "num_candidates": len(candidates)}

        # ML-guided selection (if enabled)
        if self.use_ml and self.ml_model is not None:
            # Score each valid candidate with ML
            ml_scored = []
            for heur_score, xyz in valid:
                # Get features for ML prediction
                # Note: We need target info - get from current placement context
                # For now, use simplified features
                features = [
                    xyz[0],  # des_x
                    xyz[1],  # des_y
                    xyz[2],  # des_z
                    heur_score,  # heuristic_score
                    0.65,  # target_top_z (approximate)
                    0.025,  # obj_half_h (approximate)
                    0.016,  # clearance (approximate)
                ]
                try:
                    ml_prob = self.ml_model.predict_proba([features])[0][1]
                except:
                    ml_prob = 0.5  # Fallback if prediction fails
                
                ml_scored.append((ml_prob, heur_score, xyz))
            
            # Sort by ML probability (highest first)
            ml_scored.sort(key=lambda x: x[0], reverse=True)
            
            # Pick best according to ML
            ml_prob, chosen_score, chosen = ml_scored[0]
            chosen_rank = 0
            explored = False
            
            meta = {
                "num_candidates": len(candidates),
                "num_valid": len(valid),
                "explored": explored,
                "chosen_rank": int(chosen_rank),
                "chosen_score": float(chosen_score),
                "ml_probability": float(ml_prob),
                "selection_mode": "ML",
                "top_scores": [float(s) for (s, _) in valid[:min(8, len(valid))]],
                "top_xyz": [c.tolist() for (_, c) in valid[:min(8, len(valid))]],
                "top_ml_probs": [float(p) for (p, _, _) in ml_scored[:min(8, len(ml_scored))]],
            }
            return chosen, meta

        # Heuristic-only selection (default)
        import random
        k = min(top_k, len(valid))
        if random.random() < explore_p:
            idx = random.randrange(k)
            chosen_score, chosen = valid[idx]
            chosen_rank = idx
            explored = True
        else:
            chosen_score, chosen = valid[0]
            chosen_rank = 0
            explored = False

        meta = {
            "num_candidates": len(candidates),
            "num_valid": len(valid),
            "explored": explored,
            "chosen_rank": int(chosen_rank),
            "chosen_score": float(chosen_score),
            "selection_mode": "Heuristic",
            "top_scores": [float(s) for (s, _) in valid[:min(8, len(valid))]],
            "top_xyz": [c.tolist() for (_, c) in valid[:min(8, len(valid))]],
        }
        return chosen, meta

    def _held_name(self):
        for name, oid in getattr(self.perception, "object_map", {}).items():
            if oid == self.held_object_id:
                return name
        return "held_object"

    def wait(self, seconds=0.5):
        steps = int(seconds * 240)
        for _ in range(steps):
            p.stepSimulation()

    # -----------------------------
    # IK helpers
    # -----------------------------
    def _elbow_bias_restpose(self, target_pos):
        # Improved rest pose to avoid table collisions and improve reachability
        dist = float(np.linalg.norm(np.array(target_pos[:2])))
        # Lift elbow higher for far reaches (like table) to avoid hitting table legs
        elbow = -1.8 - 0.8 * np.tanh(dist * 1.5)
        # Adjusted shoulder and wrist for better clearance
        return [0.0, -0.7, 0.0, elbow, 0.0, 2.0, 0.785]

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
                maxNumIterations=300,  # Increased for better convergence
                residualThreshold=5e-5,  # Tighter tolerance for accuracy
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
                maxNumIterations=250,  # Increased for better convergence
                residualThreshold=5e-5,  # Tighter tolerance for accuracy
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
    def _quat_to_tilt_deg(self, quat):
        """
        Tilt angle from world +Z in degrees.
        0° = upright, 90° = sideways.
        """
        rot = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        z_axis = rot[:, 2]  # local +Z axis in world coords
        cosang = float(np.clip(z_axis[2], -1.0, 1.0))
        tilt = float(np.degrees(np.arccos(cosang)))
        return tilt

    def _settle_and_measure(self, body_id, seconds=None):
        """
        Step sim and measure where the object ENDS UP.
        Returns: final_pos, final_quat, max_xy_drift, final_tilt_deg
        """
        if seconds is None:
            seconds = self.settle_seconds

        n = max(1, int(seconds / self.settle_check_dt))
        pos0, quat0 = p.getBasePositionAndOrientation(body_id)
        pos0 = np.array(pos0, dtype=float)

        max_drift = 0.0
        final_pos, final_quat = pos0, quat0

        for _ in range(n):
            self.wait(self.settle_check_dt)
            final_pos, final_quat = p.getBasePositionAndOrientation(body_id)
            final_pos = np.array(final_pos, dtype=float)

            drift = float(np.linalg.norm(final_pos[:2] - pos0[:2]))
            max_drift = max(max_drift, drift)

        tilt_deg = self._quat_to_tilt_deg(final_quat)
        return final_pos, final_quat, max_drift, tilt_deg

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
        if z < 0.02 or z > 1.15:
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
                forces=[150] * 7,  # Increased for better tracking with held objects
                positionGains=[0.1] * 7,  # Increased for more precise control
                velocityGains=[1.0] * 7  # Added velocity damping for stability
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

    def _stack_target_xy_topz(self, target_id):
        """
        More consistent stacking target than AABB center:

        - XY: use the target object's *base position* (true center), not AABB center.
        - Z: use AABB top (ok for height, much less sensitive than using it for XY).
        """
        pos, quat = p.getBasePositionAndOrientation(target_id)
        pos = np.array(pos, dtype=float)

        t_min, t_max = p.getAABB(target_id)
        target_top_z = float(t_max[2])

        target_xy = pos[:2]  # <- key change
        return target_xy, target_top_z


    # -----------------------------
    # Pick
    # -----------------------------
    def pick_object(self, object_name):
        obj_pos = self.perception.detect_object(object_name)
        if obj_pos is None:
            self._log_attempt({
                "action": "pick",
                "object": object_name,
                "target": None,
                "features": {"reason": "not_found"},
                "success": False,
                "fail_type": "not_found",
                "msg": f"Could not find '{object_name}'",
            })
            return False, f"Could not find '{object_name}'"

        obj_pos = np.array(obj_pos, dtype=float)

        obj_id = self.perception.object_map.get(object_name)
        if obj_id is None:
            self._log_attempt({
                "action": "pick",
                "object": object_name,
                "target": None,
                "features": {"obj_pos": [float(x) for x in obj_pos]},
                "success": False,
                "fail_type": "id_missing",
                "msg": f"Object id not found for '{object_name}'",
            })
            return False, f"Object id not found for '{object_name}'"

        # Features to log (for ML)
        features = {"obj_pos": [float(x) for x in obj_pos]}

        # Width check (LOG ONLY — don't early-exit for cubes)
        o_min, o_max = p.getAABB(obj_id)
        dx = float(o_max[0] - o_min[0])
        dy = float(o_max[1] - o_min[1])
        obj_width = float(max(dx, dy))
        features["obj_width"] = obj_width
        features["gripper_max_open"] = float(self.gripper_max_open)

        # IMPORTANT:
        # AABB width can inflate when the cube is tilted / wedged after a failed place.
        # For dataset collection we do NOT want this to block repicks.
        # So: log a flag, but keep going.
        features["width_flag_too_wide"] = bool(obj_width > self.gripper_max_open - 0.002)

        print(f"Picking up '{object_name}' at {obj_pos}")
        self.open_gripper()

        # 1) Approach
        approach_pos = obj_pos + np.array([0, 0, 0.10], dtype=float)
        features["approach"] = [float(x) for x in approach_pos]
        ok, reason = self.is_reachable(approach_pos, use_down_orientation=False)
        if not ok:
            self._log_attempt({
                "action": "pick",
                "object": object_name,
                "target": None,
                "features": features,
                "success": False,
                "fail_type": "unreachable_approach",
                "msg": f"Unreachable approach pose: {reason}",
            })
            return False, f"Unreachable approach pose: {reason}"
        self.move_to_position(approach_pos, duration=2.0, use_down_orientation=False)

        # 2) Grasp (down + fallback)
        grasp_pos = obj_pos + np.array([0, 0, 0.015], dtype=float)
        features["grasp"] = [float(x) for x in grasp_pos]
        ok, reason = self.is_reachable(grasp_pos, use_down_orientation=True)
        use_down = True
        if not ok:
            ok2, reason2 = self.is_reachable(grasp_pos, use_down_orientation=False)
            if not ok2:
                self._log_attempt({
                    "action": "pick",
                    "object": object_name,
                    "target": None,
                    "features": features,
                    "success": False,
                    "fail_type": "unreachable_grasp",
                    "msg": f"Unreachable grasp pose: {reason} | fallback: {reason2}",
                })
                return False, f"Unreachable grasp pose: {reason} | fallback: {reason2}"
            use_down = False
        features["use_down_grasp"] = 1.0 if use_down else 0.0

        self.move_to_position(grasp_pos, duration=1.2, use_down_orientation=use_down)
        self.wait(0.1)
        self.close_gripper()
        self.wait(0.1)

        # 3) Lift
        lift_pos = obj_pos + np.array([0, 0, 0.18], dtype=float)
        features["lift"] = [float(x) for x in lift_pos]
        ok, reason = self.is_reachable(lift_pos, use_down_orientation=False)
        if not ok:
            self._log_attempt({
                "action": "pick",
                "object": object_name,
                "target": None,
                "features": features,
                "success": False,
                "fail_type": "unreachable_lift",
                "msg": f"Unreachable lift pose: {reason}",
            })
            return False, f"Unreachable lift pose: {reason}"
        self.move_to_position(lift_pos, duration=1.5, use_down_orientation=False)

        # Verify lifted
        new_pos, _ = p.getBasePositionAndOrientation(obj_id)
        new_pos = np.array(new_pos, dtype=float)
        features["new_pos"] = [float(x) for x in new_pos]
        if new_pos[2] < obj_pos[2] + 0.05:
            self._log_attempt({
                "action": "pick",
                "object": object_name,
                "target": None,
                "features": features,
                "success": False,
                "fail_type": "grasp_failed",
                "msg": "Grasp failed (object did not lift)",
            })
            return False, "Grasp failed (object did not lift)"

        self.held_object_id = obj_id
        
        # COLLISION AVOIDANCE: Lift extra high and move away from table
        # Detect if we picked from table (high z position) vs floor (low z)
        picked_from_table = obj_pos[2] > 0.5  # Table is around 0.65m, floor is ~0.025m
        
        if picked_from_table:
            # Picked from table - need to lift high AND move away from table before descending
            print(f"[PICK] Detected table pickup, executing safe retreat...")
            
            # Step 1: Lift very high above table
            high_above_table = obj_pos + np.array([0, 0, 0.20], dtype=float)
            ok_high, _ = self.is_reachable(high_above_table, use_down_orientation=False)
            if ok_high:
                print(f"[PICK] Lifting high above table: {[round(float(x), 3) for x in high_above_table]}")
                self.move_to_position(high_above_table, duration=1.2, use_down_orientation=False)
                self.wait(0.1)
            
            # Step 2: Move toward robot base (away from table) while staying high
            retreat_pos = np.array([
                max(0.35, obj_pos[0] - 0.25),  # Move closer to robot (reduce x)
                obj_pos[1],  # Keep same y
                max(0.70, high_above_table[2])  # Stay high
            ], dtype=float)
            ok_retreat, _ = self.is_reachable(retreat_pos, use_down_orientation=False)
            if not ok_retreat:
                # Fallback: try a less aggressive retreat
                retreat_pos = np.array([
                    max(0.35, obj_pos[0] - 0.15),
                    obj_pos[1],
                    max(0.70, high_above_table[2])
                ], dtype=float)
                ok_retreat, _ = self.is_reachable(retreat_pos, use_down_orientation=False)

            if ok_retreat:
                print(f"[PICK] Retreating from table: {[round(float(x), 3) for x in retreat_pos]}")
                self.move_to_position(retreat_pos, duration=1.5, use_down_orientation=False)
                self.wait(0.1)
        else:
            # Picked from floor - just lift high to clear table
            extra_high_lift = obj_pos + np.array([0, 0, 0.25], dtype=float)
            ok_high, _ = self.is_reachable(extra_high_lift, use_down_orientation=False)
            if ok_high:
                print(f"[PICK] Lifting extra high to avoid table: {[round(float(x), 3) for x in extra_high_lift]}")
                self.move_to_position(extra_high_lift, duration=1.5, use_down_orientation=False)
                self.wait(0.1)
        
        self._log_attempt({
            "action": "pick",
            "object": object_name,
            "target": None,
            "features": features,
            "success": True,
            "fail_type": None,
            "msg": f"Successfully picked up '{object_name}'",
        })
        return True, f"Successfully picked up '{object_name}'"


    def place_object(self, target_name, desired_obj_pos=None, candidate_meta=None):
        # IMPORTANT: do NOT go_home here.
        # We assume we are already lifted safely after pick.

        target_id = self.perception.object_map.get(target_name)
        if target_id is None:
            self._log_attempt({
                "action": "place",
                "object": self._held_name(),
                "target": target_name,
                "features": {"reason": "target_missing"},
                "success": False,
                "fail_type": "target_missing",
                "msg": f"Target id not found for '{target_name}'",
            })
            return False, f"Target id not found for '{target_name}'"

        if self.held_object_id is None:
            self._log_attempt({
                "action": "place",
                "object": "none",
                "target": target_name,
                "features": {"reason": "no_held_object"},
                "success": False,
                "fail_type": "no_held_object",
                "msg": "No object currently held",
            })
            return False, "No object currently held"

        held_id = self.held_object_id
        held_name = self._held_name()

        # Held object half-height
        o_min, o_max = p.getAABB(held_id)
        obj_half_h = 0.5 * float(o_max[2] - o_min[2])

        is_table = target_name in ("table", "the_table")

        # If a desired position is provided (candidate planner / ML), use it.
        if desired_obj_pos is not None:
            desired_obj_pos = np.array(desired_obj_pos, dtype=float)

            # We still need target_top_z + clearance for logging thresholds
            if is_table:
                if getattr(self, "table_top_z", None) is None:
                    t_min, t_max = p.getAABB(target_id)
                    target_top_z = float(t_max[2])
                else:
                    target_top_z = float(self.table_top_z)
                clearance = 0.010
            else:
                # for stacking thresholds/logs, derive top_z from target AABB
                t_min, t_max = p.getAABB(target_id)
                target_top_z = float(t_max[2])
                clearance = 0.016

        else:
            # Otherwise compute desired position as before
            if is_table:
                target_xy = np.array(self.table_drop_xy, dtype=float)
                if getattr(self, "table_top_z", None) is None:
                    t_min, t_max = p.getAABB(target_id)
                    target_top_z = float(t_max[2])
                else:
                    target_top_z = float(self.table_top_z)
                clearance = 0.010
            else:
                # ✅ More consistent stacking: XY from base pose, Z from AABB top
                target_xy, target_top_z = self._stack_target_xy_topz(target_id)
                target_xy = np.array(target_xy, dtype=float)
                clearance = 0.016

            desired_obj_pos = np.array([
                float(target_xy[0]),
                float(target_xy[1]),
                float(target_top_z + obj_half_h + clearance),
            ], dtype=float)

        print("[PLACE] desired_obj_pos =", [round(float(x), 3) for x in desired_obj_pos])

        # COLLISION AVOIDANCE: Add safe waypoints based on target location
        # - Table placement: Arc over table from above
        # - Floor/low stacking: Retreat from table, then descend
        current_ee_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link)[4], dtype=float)
        target_is_low = desired_obj_pos[2] < 0.4  # Floor or low stack (< 0.4m)
        target_is_high = desired_obj_pos[2] > 0.5 # Table or stack on table (> 0.5m)
        start_is_low = current_ee_pos[2] < 0.4    # Starting from floor?

        if start_is_low and target_is_high:
            # ---------------------------------------------------------
            # Low -> High Strategy (Floor to Table)
            # ---------------------------------------------------------
            # We are likely on the floor. If we arc or move linear, we hit the table edge.
            # solution: Lift vertically to a safe height (e.g. > table height) at CURRENT XY first.
            print(f"[PLACE] Detected Low-to-High transition. Lifting vertically first...")
            
            # Aim for plenty of clearance above table
            # LOWERED SAFE Z: 0.85 -> 0.75 to be less aggressive.
            # Table is approx 0.65m. 0.75m clears it by 10cm.
            safe_z = max(0.75, desired_obj_pos[2] + 0.05)
            
            # Waypoint: Current XY, Safe Z
            lift_waypoint = np.array([current_ee_pos[0], current_ee_pos[1], safe_z], dtype=float)
            
            ok, reason = self.is_reachable(lift_waypoint, use_down_orientation=False)
            if not ok:
                # Fallback: maybe 0.75 is still too high? Try 0.70
                lift_waypoint[2] = 0.70
                ok, reason = self.is_reachable(lift_waypoint, use_down_orientation=False)
            
            if ok:
                print(f"[PLACE] Lifting to safe height: {[round(float(x), 3) for x in lift_waypoint]}")
                # INCREASED DURATION: 1.5 -> 2.0 (slightly faster than 3.0 but smoother than 1.5)
                self.move_to_position(lift_waypoint, duration=2.0)
                self.wait(0.1)
            else:
                print(f"[PLACE] Warning: Vertical lift unreachable ({reason}). trying standard path...")

        elif is_table:
            # Placing on table - arc over from above
            midpoint_xy = (current_ee_pos[:2] + desired_obj_pos[:2]) / 2.0
            safe_height = min(0.85, max(0.70, current_ee_pos[2] + 0.10, desired_obj_pos[2] + 0.20))
            safe_waypoint = np.array([midpoint_xy[0], midpoint_xy[1], safe_height], dtype=float)
            
            ok, reason = self.is_reachable(safe_waypoint, use_down_orientation=False)
            if ok:
                print(f"[PLACE] Moving to safe waypoint (table): {[round(float(x), 3) for x in safe_waypoint]}")
                self.move_to_position(safe_waypoint, duration=1.5)
                self.wait(0.1)
            else:
                # Fallback: try higher but less far out
                safe_waypoint[2] += 0.10
                ok, reason = self.is_reachable(safe_waypoint, use_down_orientation=False)
                if ok:
                    print(f"[PLACE] Moving to high safe waypoint: {[round(float(x), 3) for x in safe_waypoint]}")
                    self.move_to_position(safe_waypoint, duration=1.5)
                    self.wait(0.1)
                else:
                    print(f"[PLACE] Warning: All safe waypoints unreachable ({reason}), proceeding directly")
                
        elif target_is_low and current_ee_pos[2] > 0.6:
            # Placing on floor/low stack from high position (e.g., after picking from table)
            # Need to retreat from table area before descending
            print(f"[PLACE] Detected high-to-low placement, executing safe descent...")
            
            # Step 1: Move away from table while staying high
            retreat_waypoint = np.array([
                max(0.35, desired_obj_pos[0] - 0.15),  # Closer to robot, away from table
                desired_obj_pos[1],  # Same y as target
                max(0.70, current_ee_pos[2])  # Stay high
            ], dtype=float)
            ok_retreat, _ = self.is_reachable(retreat_waypoint, use_down_orientation=False)
            if not ok_retreat:
                # Fallback: less aggressive retreat
                retreat_waypoint[0] = max(0.35, desired_obj_pos[0] - 0.10)
                ok_retreat, _ = self.is_reachable(retreat_waypoint, use_down_orientation=False)

            if ok_retreat:
                print(f"[PLACE] Retreating from table area: {[round(float(x), 3) for x in retreat_waypoint]}")
                self.move_to_position(retreat_waypoint, duration=1.5)
                self.wait(0.1)
            
            # Step 2: Move to position above target (still high, but at target XY)
            above_target = np.array([
                desired_obj_pos[0],
                desired_obj_pos[1],
                max(0.50, desired_obj_pos[2] + 0.30)  # High above target
            ], dtype=float)
            ok_above, _ = self.is_reachable(above_target, use_down_orientation=False)
            if ok_above:
                print(f"[PLACE] Moving above target: {[round(float(x), 3) for x in above_target]}")
                self.move_to_position(above_target, duration=1.5)
                self.wait(0.1)

        # Approach (high) - increased height to avoid table collisions
        approach_pos = desired_obj_pos + np.array([0, 0, 0.18], dtype=float)
        ok, reason = self.is_reachable(approach_pos, use_down_orientation=False)
        if not ok:
            features = {
                "is_table": bool(is_table),
                "desired": [float(x) for x in desired_obj_pos],
                "detail": reason,
                "target_top_z": float(target_top_z),
                "obj_half_h": float(obj_half_h),
                "clearance": float(clearance),
            }
            if candidate_meta is not None:
                features["candidate_meta"] = candidate_meta

            self._log_attempt({
                "action": "place",
                "object": held_name,
                "target": target_name,
                "features": features,
                "success": False,
                "fail_type": "unreachable_approach",
                "msg": f"Unreachable place approach: {reason}",
            })
            return False, f"Unreachable place approach: {reason}"
        self.move_to_position(approach_pos, duration=1.5)  # Slower approach
        self.wait(0.1)  # Longer settle time

        # Pre-place
        pre_place = desired_obj_pos + np.array([0, 0, 0.06], dtype=float)
        ok, reason = self.is_reachable(pre_place, use_down_orientation=False)
        if not ok:
            features = {
                "is_table": bool(is_table),
                "desired": [float(x) for x in desired_obj_pos],
                "detail": reason,
                "target_top_z": float(target_top_z),
                "obj_half_h": float(obj_half_h),
                "clearance": float(clearance),
            }
            if candidate_meta is not None:
                features["candidate_meta"] = candidate_meta

            self._log_attempt({
                "action": "place",
                "object": held_name,
                "target": target_name,
                "features": features,
                "success": False,
                "fail_type": "unreachable_pre",
                "msg": f"Unreachable pre-place pose: {reason}",
            })
            return False, f"Unreachable pre-place pose: {reason}"
        self.move_to_position(pre_place, duration=1.2)  # Slower for precision
        self.wait(0.08)  # Longer settle

        # Final place (slow)
        ok, reason = self.is_reachable(desired_obj_pos, use_down_orientation=False)
        if not ok:
            features = {
                "is_table": bool(is_table),
                "desired": [float(x) for x in desired_obj_pos],
                "detail": reason,
                "target_top_z": float(target_top_z),
                "obj_half_h": float(obj_half_h),
                "clearance": float(clearance),
            }
            if candidate_meta is not None:
                features["candidate_meta"] = candidate_meta

            self._log_attempt({
                "action": "place",
                "object": held_name,
                "target": target_name,
                "features": features,
                "success": False,
                "fail_type": "unreachable_final",
                "msg": f"Unreachable place pose: {reason}",
            })
            return False, f"Unreachable place pose: {reason}"
        self.move_to_position(desired_obj_pos, duration=1.2)  # Very slow for accurate placement
        self.wait(0.15)  # Longer wait for object to settle in gripper

        # Release
        self.open_gripper()
        self.wait(0.2)  # Longer wait for clean release

        # Lift straight up slowly (don't drag sideways or disturb placement)
        lift_after_release = desired_obj_pos + np.array([0, 0, 0.16], dtype=float)
        ok, _ = self.is_reachable(lift_after_release, use_down_orientation=False)
        if ok:
            self.move_to_position(lift_after_release, duration=1.0)  # Slower lift
            self.wait(0.08)

        # ---- Truthful settle verification (LOCATION-BASED ONLY) ----
        final_pos, final_quat, max_drift, tilt_deg = self._settle_and_measure(held_id)
        final_pos = np.array(final_pos, dtype=float)

        xy_err = float(np.linalg.norm(final_pos[:2] - desired_obj_pos[:2]))

        # STRICTER THRESHOLDS for harder data collection (more failures for ML to learn from)
        # Target: ~60% success rate (40% failures) for balanced training data
        if is_table:
            xy_thresh = 0.06  # Stricter: was 0.12, now 0.06 (half as forgiving)
            z_ok = final_pos[2] > (target_top_z + obj_half_h - 0.015)  # Stricter: was 0.03
        else:
            xy_thresh = 0.04  # Stricter: was 0.08, now 0.04 (half as forgiving)
            z_ok = final_pos[2] > (target_top_z + obj_half_h - 0.010)  # Stricter: was 0.020

        drift_ok = max_drift <= self.max_xy_drift

        features = {
            "desired": [float(x) for x in desired_obj_pos],
            "final_pos": [float(x) for x in final_pos],
            "xy_err": float(xy_err),
            "tilt_deg": float(tilt_deg),          # logged only
            "max_xy_drift": float(max_drift),
            "z_ok": bool(z_ok),
            "drift_ok": bool(drift_ok),
            "is_table": bool(is_table),
            "xy_thresh": float(xy_thresh),
            "target_top_z": float(target_top_z),
            "obj_half_h": float(obj_half_h),
            "clearance": float(clearance),
        }
        if candidate_meta is not None:
            features["candidate_meta"] = candidate_meta

        if (xy_err > xy_thresh) or (not z_ok) or (not drift_ok):
            if not z_ok:
                fail_type = "fell"
            elif not drift_ok:
                fail_type = "drifted"
            else:
                fail_type = "xy_miss"

            self._log_attempt({
                "action": "place",
                "object": held_name,
                "target": target_name,
                "features": features,
                "success": False,
                "fail_type": fail_type,
                "msg": f"Place failed after settle (xy_err={xy_err:.3f}, drift={max_drift:.3f}, tilt_logged={tilt_deg:.1f})",
            })
            return False, f"Place failed after settle: xy_err={xy_err:.3f}, drift={max_drift:.3f}"

        self._log_attempt({
            "action": "place",
            "object": held_name,
            "target": target_name,
            "features": features,
            "success": True,
            "fail_type": None,
            "msg": f"Successfully placed object on '{target_name}'",
        })

        self.held_object_id = None
        return True, f"Successfully placed object on '{target_name}'"

    # -----------------------------
    # Command loop
    # -----------------------------
    def execute_command(self, command: str):
        # Start in a consistent pose for the command (optional).
        # If you hate this, you can remove it, but it can reduce weird starts.
        self.go_home()
        print(f"\n>>> Executing: '{command}'")

        norm = self.parser.normalize_command(command)
        actions = self.parser.parse(norm)
        if actions is None:
            # Optional command-level log
            try:
                self._log_attempt({
                    "action": "command",
                    "object": None,
                    "target": None,
                    "features": {"command": command, "normalized": norm, "parsed": None},
                    "success": False,
                    "fail_type": "parse_failed",
                    "msg": "Could not understand command",
                })
            except Exception:
                pass
            return False, "Could not understand command"

        print(f"Parsed actions: {actions}")

        last_picked_name = None  # so we can repick if place fails

        for action, obj, target in actions:
            # Default tries
            pick_tries = 5
            place_tries = 3  # you can tune this

            success, msg = False, ""

            if action == "pick":
                last_picked_name = obj
                for attempt in range(pick_tries):
                    if pick_tries > 1:
                        print(f" -- pick attempt {attempt+1}/{pick_tries}")

                    success, msg = self.pick_object(obj)
                    if success:
                        break

                    # recovery between pick attempts
                    self.go_home()
                    self.wait(0.2)

                if not success:
                    return False, f"Action '{action}' failed: {msg}"

                print(msg)
                continue

            if action == "place":
                # NOTE: your parser puts the target into `obj` for place
                target_name = obj

                for attempt in range(place_tries):
                    print(f" -- place attempt {attempt+1}/{place_tries}")

                    # Try placing directly from current lifted pose (NO go_home first)
                    success, msg = self.place_object(target_name)
                    if success:
                        break

                    # If place failed, recover + repick + retry place
                    self.go_home()
                    self.wait(0.2)

                    # If we know what we picked, repick it (because place failures often drop it)
                    if last_picked_name is not None:
                        repick_ok, repick_msg = self.pick_object(last_picked_name)
                        if not repick_ok:
                            # Try a couple extra repick attempts from home
                            repick_ok2 = False
                            for rp in range(2):
                                self.go_home()
                                self.wait(0.15)
                                repick_ok2, repick_msg = self.pick_object(last_picked_name)
                                if repick_ok2:
                                    break
                            repick_ok = repick_ok2

                        if not repick_ok:
                            return False, f"Recovery repick failed: {repick_msg}"

                if not success:
                    return False, f"Action '{action}' failed: {msg}"

                print(msg)
                continue

            # Simple actions
            if action == "open_gripper":
                self.open_gripper()
                print("Opened gripper")
                continue

            if action == "close_gripper":
                self.close_gripper()
                print("Closed gripper")
                continue

            return False, f"Unknown action: {action}"

        # End-of-command reset (optional)
        self.go_home()

        # Optional command-level log
        try:
            self._log_attempt({
                "action": "command",
                "object": None,
                "target": None,
                "features": {"command": command, "normalized": norm, "parsed": actions},
                "success": True,
                "fail_type": None,
                "msg": "Command executed successfully",
            })
        except Exception:
            pass

        return True, "Command executed successfully"
