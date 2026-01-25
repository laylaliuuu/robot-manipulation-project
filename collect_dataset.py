# collect_dataset.py
import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data

from robot_controller import RobotController
from perception import PerceptionModule
from language_parser import CommandParser

import joblib


# -----------------------------
# Config: match simulation.py
# -----------------------------
TABLE_POS = [1.25, 0.0, 0.0]        # EXACTLY like simulation.py
DROP_XY   = np.array([0.62, 0.00])  # your tuned safe lane from simulation.py

# Candidate settings
GRID = 5
STEP = 0.012

# How many place attempts per episode?
TOP_K = 5            # try top 5 first
EXPLORE_EXTRA = 4    # then try 4 random candidates (diversity)
MAX_PLACE_ATTEMPTS = TOP_K + EXPLORE_EXTRA

# Repick settings (after a failed place, object often drops)
REPICK_TRIES = 3

# Logging
RUNS_DIR = "attempts_runs"


def spawn_cube_on_table(xy, table_top_z, color):
    """
    Spawn a cube at the correct z so it rests on the table (not floating / not penetrating).
    """
    tmp_z = float(table_top_z + 0.20)
    uid = p.loadURDF("cube_small.urdf", [float(xy[0]), float(xy[1]), tmp_z], useFixedBase=False)
    p.changeVisualShape(uid, -1, rgbaColor=color)

    # friction helps a lot for pick + stacking (more stable contacts)
    p.changeDynamics(uid, -1, lateralFriction=1.2, rollingFriction=0.01, spinningFriction=0.01)

    # compute half height
    o_min, o_max = p.getAABB(uid)
    half_h = 0.5 * float(o_max[2] - o_min[2])

    # snap to resting z
    z = float(table_top_z + half_h + 0.002)
    p.resetBasePositionAndOrientation(uid, [float(xy[0]), float(xy[1]), z], [0, 0, 0, 1])
    return uid


def setup_world(seed: int):
    """
    Creates a fresh sim with robot + table + two cubes placed on the table top.
    Returns: (ctrl, table_id, white_id, green_id)
    """
    random.seed(seed)
    np.random.seed(seed)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    table_id = p.loadURDF("table/table.urdf", TABLE_POS, useFixedBase=True)
    p.changeDynamics(table_id, -1, lateralFriction=1.2, rollingFriction=0.01, spinningFriction=0.01)

    t_min, t_max = p.getAABB(table_id)
    table_top_z = float(t_max[2])

    # Spawn cubes in reachable band (keep it realistic but not silly-hard)
    white_xy = (float(np.random.uniform(0.58, 0.75)), float(np.random.uniform(-0.14, 0.14)))
    green_xy = (float(np.random.uniform(0.58, 0.75)), float(np.random.uniform(-0.14, 0.14)))

    white_id = spawn_cube_on_table(white_xy, table_top_z, [1, 1, 1, 1])
    green_id = spawn_cube_on_table(green_xy, table_top_z, [0, 1, 0, 1])

    object_map = {
        "table": table_id,
        "the_table": table_id,
        "white_cube": white_id,
        "green_cube": green_id,
    }

    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    ctrl = RobotController(robot_id, 9, 10, perception, parser)

    # Match simulation.py controller settings
    ctrl.table_drop_xy = DROP_XY.copy()
    ctrl.table_top_z = table_top_z

    return ctrl, table_id, white_id, green_id


def build_attempt_list(valid, top_k: int, explore_extra: int):
    """
    valid is list of (score, cand_xyz) sorted best->worst.
    We take top_k + explore_extra random from remaining.
    """
    K = min(top_k, len(valid))
    attempt_list = valid[:K].copy()

    remaining = valid[K:]
    if explore_extra > 0 and len(remaining) > 0:
        take = min(explore_extra, len(remaining))
        idxs = np.random.choice(len(remaining), size=take, replace=False)
        for i in idxs:
            attempt_list.append(remaining[int(i)])

    return attempt_list, K


def repick_green(ctrl: RobotController) -> bool:
    """
    Try to repick green cube after a failed placement.
    """
    ctrl.go_home()
    ctrl.wait(0.2)

    for _ in range(REPICK_TRIES):
        ok, _ = ctrl.pick_object("green_cube")
        if ok:
            return True
        ctrl.go_home()
        ctrl.wait(0.2)
    return False


def run_episode(ep: int, seed: int,
                top_k: int = TOP_K,
                explore_extra: int = EXPLORE_EXTRA,
                continue_after_success: bool = True):
    """
    Data-collection episode.
    We WANT multiple place attempts logged per episode, so ML has negatives too.

    continue_after_success=True means:
      - even if candidate 0 succeeds, we still try more candidates (for labels)
      - this is for dataset collection only (not for your final demo behavior)
    """
    ctrl, table_id, white_id, green_id = setup_world(seed)
    ctrl.set_episode(ep, seed)

    # settle physics
    ctrl.wait(1.0)

    # pick green once to start
    ok_pick, _ = ctrl.pick_object("green_cube")
    if not ok_pick:
        return False

    # generate + score candidates for stacking
    best_cand, meta = ctrl.choose_best_candidate_ml(
        ctrl.held_object_id, white_id, cands
    )

    if best_cand is None:
        return False

    ok_place, _ = ctrl.place_object(
        "white_cube",
        desired_obj_pos=best_cand,
        candidate_meta=meta
    )

    return bool(ok_place)
    scored = [(ctrl.score_candidate_heuristic(c), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    valid = [(s, c) for (s, c) in scored if s > -1e8]

    if not valid:
        ctrl._log_attempt({
            "action": "place",
            "object": "green_cube",
            "target": "white_cube",
            "features": {"candidate_meta": {"reason": "no_reachable_candidates", "num_candidates": len(cands)}},
            "success": False,
            "fail_type": "no_reachable_candidates",
            "msg": "No reachable candidates",
        })
        return False

    attempt_list, K = build_attempt_list(valid, top_k=top_k, explore_extra=explore_extra)

    episode_success = False
    attempts_made = 0

    for rank, (score, cand) in enumerate(attempt_list):
        if attempts_made >= MAX_PLACE_ATTEMPTS:
            break
        attempts_made += 1

        meta = {
            "num_candidates": int(len(cands)),
            "num_valid": int(len(valid)),
            "attempt_rank": int(rank),
            "heuristic_score": float(score),
            "attempt_pool": "random" if rank >= K else "topK",
            "top_scores": [float(s) for (s, _) in valid[:min(8, len(valid))]],
            "top_xyz": [c.tolist() for (_, c) in valid[:min(8, len(valid))]],
        }

        ok_place, _ = ctrl.place_object("white_cube", desired_obj_pos=cand, candidate_meta=meta)

        if ok_place:
            episode_success = True
            if not continue_after_success:
                return True

        # after each place attempt, we likely dropped the cube; repick for next attempt
        if rank != len(attempt_list) - 1:
            repicked = repick_green(ctrl)
            if not repicked:
                break

    return episode_success


def main(n_episodes: int = 300,
         base_seed: int = 1000,
         top_k: int = TOP_K,
         explore_extra: int = EXPLORE_EXTRA,
         continue_after_success: bool = True,
         use_gui: bool = False):
    """
    use_gui=True is a 10-episode sanity check to visually confirm it matches your sim.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RUNS_DIR, f"attempts_{run_tag}.jsonl")
    print("[LOG] writing to:", log_path)

    mode = p.GUI if use_gui else p.DIRECT
    p.connect(mode)

    successes = 0
    for ep in range(n_episodes):
        seed = base_seed + ep

        # create a fresh controller per episode, so we set log_path by monkey-patching after setup
        # easiest: run_episode creates controller, so we patch inside via global var is annoying.
        # Instead: set default log_path in RobotController after it's created:
        # We'll do it by temporarily wrapping setup_world via a local patch:
        ok = run_episode_with_log(ep, seed, log_path,
                                 top_k=top_k, explore_extra=explore_extra,
                                 continue_after_success=continue_after_success)
        successes += int(ok)

        if (ep + 1) % 50 == 0:
            print(f"[{ep+1}/{n_episodes}] episode_success_rate={successes/(ep+1):.3f}")

    p.disconnect()
    print("Final:", successes, "/", n_episodes)
    print("[DONE] dataset:", log_path)


def run_episode_with_log(ep, seed, log_path, top_k, explore_extra, continue_after_success):
    """
    Same as run_episode, but forces ctrl.log_path = log_path for this run.
    """
    ctrl, table_id, white_id, green_id = setup_world(seed)
    ctrl.log_path = log_path  # <-- force this run's log file
    ctrl.set_episode(ep, seed)

    ctrl.wait(1.0)

    ok_pick, _ = ctrl.pick_object("green_cube")
    if not ok_pick:
        return False

    cands = ctrl.generate_stack_candidates(ctrl.held_object_id, white_id, grid=GRID, step=STEP)

    scored = [(ctrl.score_candidate_heuristic(c), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    valid = [(s, c) for (s, c) in scored if s > -1e8]

    if not valid:
        ctrl._log_attempt({
            "action": "place",
            "object": "green_cube",
            "target": "white_cube",
            "features": {"candidate_meta": {"reason": "no_reachable_candidates", "num_candidates": len(cands)}},
            "success": False,
            "fail_type": "no_reachable_candidates",
            "msg": "No reachable candidates",
        })
        return False

    attempt_list, K = build_attempt_list(valid, top_k=top_k, explore_extra=explore_extra)

    episode_success = False
    attempts_made = 0

    for rank, (score, cand) in enumerate(attempt_list):
        if attempts_made >= (top_k + explore_extra):
            break
        attempts_made += 1

        meta = {
            "num_candidates": int(len(cands)),
            "num_valid": int(len(valid)),
            "attempt_rank": int(rank),
            "heuristic_score": float(score),
            "attempt_pool": "random" if rank >= K else "topK",
            "top_scores": [float(s) for (s, _) in valid[:min(8, len(valid))]],
            "top_xyz": [c.tolist() for (_, c) in valid[:min(8, len(valid))]],
        }

        ok_place, _ = ctrl.place_object("white_cube", desired_obj_pos=cand, candidate_meta=meta)

        if ok_place:
            episode_success = True
            if not continue_after_success:
                return True

        if rank != len(attempt_list) - 1:
            repicked = repick_green(ctrl)
            if not repicked:
                break

    return episode_success


if __name__ == "__main__":
    # 1) Quick GUI sanity check (visual) – do 10 episodes
    # main(n_episodes=10, base_seed=1000, top_k=5, explore_extra=2, continue_after_success=True, use_gui=True)

    # 2) Real dataset run (DIRECT) – start with 50, then scale up
    main(n_episodes=50, base_seed=1000, top_k=5, explore_extra=4, continue_after_success=True, use_gui=False)
    # main(n_episodes=300, base_seed=1000, top_k=5, explore_extra=4, continue_after_success=True, use_gui=False)
