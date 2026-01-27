
import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
from datetime import datetime

from robot_controller import RobotController
from perception import PerceptionModule
from language_parser import CommandParser

# -----------------------------
# Configuration
# -----------------------------
TABLE_POS = [1.25, 0.0, 0.0]
DROP_XY = np.array([0.62, 0.00])  # Safe drop zone for table placement

# Candidate generation settings
GRID_SIZE = 5           # 5x5 grid for stacking candidates
GRID_STEP = 0.012       # 1.2cm spacing between candidates
RADIAL_SAMPLES = 16     # Number of radial samples for table placement

# Data collection strategy
TOP_K = 5               # Try top-K best candidates (exploitation)
EXPLORE_EXTRA = 4       # Try random candidates (exploration for diversity)
MAX_PLACE_ATTEMPTS = TOP_K + EXPLORE_EXTRA

# Repick settings
REPICK_TRIES = 3        # How many times to retry picking after failed placement

# Logging
RUNS_DIR = "attempts_runs"
os.makedirs(RUNS_DIR, exist_ok=True)


def spawn_cube_on_table(xy, table_top_z, color, name="cube"):
    """
    Spawn a cube at the correct height to rest on the table.
    
    Args:
        xy: (x, y) position on table
        table_top_z: Height of table surface
        color: RGBA color [r, g, b, a]
        name: Identifier for debugging
    
    Returns:
        body_id: PyBullet body ID
    """
    # Spawn slightly above table to avoid penetration
    tmp_z = float(table_top_z + 0.20)
    uid = p.loadURDF("cube_small.urdf", 
                     [float(xy[0]), float(xy[1]), tmp_z], 
                     useFixedBase=False)
    p.changeVisualShape(uid, -1, rgbaColor=color)

    # Set friction for stable grasping and stacking
    p.changeDynamics(uid, -1, 
                     lateralFriction=1.2, 
                     rollingFriction=0.01, 
                     spinningFriction=0.01)

    # Compute object dimensions
    o_min, o_max = p.getAABB(uid)
    half_h = 0.5 * float(o_max[2] - o_min[2])

    # Snap to resting position on table
    z = float(table_top_z + half_h + 0.002)
    p.resetBasePositionAndOrientation(uid, 
                                      [float(xy[0]), float(xy[1]), z], 
                                      [0, 0, 0, 1])
    return uid


def setup_world(seed: int):
    """
    Create a fresh simulation environment with robot + table + objects.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        (ctrl, table_id, white_id, green_id)
    """
    random.seed(seed)
    np.random.seed(seed)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Load robot
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # Load table
    table_id = p.loadURDF("table/table.urdf", TABLE_POS, useFixedBase=True)
    p.changeDynamics(table_id, -1, 
                     lateralFriction=1.2, 
                     rollingFriction=0.01, 
                     spinningFriction=0.01)

    t_min, t_max = p.getAABB(table_id)
    table_top_z = float(t_max[2])

    # Spawn cubes in reachable workspace
    # Use realistic positions that are sometimes easy, sometimes challenging
    white_xy = (float(np.random.uniform(0.58, 0.75)), 
                float(np.random.uniform(-0.14, 0.14)))
    green_xy = (float(np.random.uniform(0.58, 0.75)), 
                float(np.random.uniform(-0.14, 0.14)))

    white_id = spawn_cube_on_table(white_xy, table_top_z, [1, 1, 1, 1], "white")
    green_id = spawn_cube_on_table(green_xy, table_top_z, [0, 1, 0, 1], "green")

    # Setup perception and controller
    object_map = {
        "table": table_id,
        "the_table": table_id,
        "white_cube": white_id,
        "green_cube": green_id,
    }

    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    ctrl = RobotController(robot_id, 9, 10, perception, parser)

    # Configure controller
    ctrl.table_drop_xy = DROP_XY.copy()
    ctrl.table_top_z = table_top_z

    return ctrl, table_id, white_id, green_id


def build_attempt_list(valid, top_k: int, explore_extra: int):
    """
    Build a list of candidates to attempt, balancing exploitation and exploration.
    
    Args:
        valid: List of (score, xyz) tuples sorted best->worst
        top_k: Number of top candidates to try (exploitation)
        explore_extra: Number of random candidates to try (exploration)
    
    Returns:
        (attempt_list, K) where K is the number of top-k candidates
    """
    K = min(top_k, len(valid))
    attempt_list = valid[:K].copy()

    # Add random exploration candidates for data diversity
    remaining = valid[K:]
    if explore_extra > 0 and len(remaining) > 0:
        take = min(explore_extra, len(remaining))
        idxs = np.random.choice(len(remaining), size=take, replace=False)
        for i in idxs:
            attempt_list.append(remaining[int(i)])

    return attempt_list, K


def repick_object(ctrl: RobotController, object_name: str) -> bool:
    """
    Try to repick an object after a failed placement.
    
    Args:
        ctrl: Robot controller
        object_name: Name of object to pick
    
    Returns:
        True if successfully repicked, False otherwise
    """
    ctrl.go_home()
    ctrl.wait(0.2)

    for _ in range(REPICK_TRIES):
        ok, _ = ctrl.pick_object(object_name)
        if ok:
            return True
        ctrl.go_home()
        ctrl.wait(0.2)
    return False


def run_stacking_episode(ep: int, seed: int, log_path: str,
                         top_k: int = TOP_K,
                         explore_extra: int = EXPLORE_EXTRA,
                         continue_after_success: bool = True):
    """
    Run a single stacking episode: pick green cube, try to stack on white cube.
    
    This collects data for the ML model by attempting multiple placements
    and logging the features + outcomes.
    
    Args:
        ep: Episode number
        seed: Random seed
        log_path: Path to JSONL log file
        top_k: Number of top candidates to try
        explore_extra: Number of random candidates to try
        continue_after_success: If True, keep trying candidates even after success
    
    Returns:
        True if at least one placement succeeded
    """
    ctrl, table_id, white_id, green_id = setup_world(seed)
    ctrl.log_path = log_path
    ctrl.set_episode(ep, seed)

    # Let physics settle
    ctrl.wait(1.0)

    # Pick the green cube
    ok_pick, _ = ctrl.pick_object("green_cube")
    if not ok_pick:
        print(f"[EP {ep}] Failed to pick green cube")
        return False

    # Generate stacking candidates
    cands = ctrl.generate_stack_candidates(ctrl.held_object_id, white_id, 
                                          grid=GRID_SIZE, step=GRID_STEP)

    # Score each candidate with heuristic (IK/FK error)
    scored = [(ctrl.score_candidate_heuristic(c), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    valid = [(s, c) for (s, c) in scored if s > -1e8]

    if not valid:
        ctrl._log_attempt({
            "action": "place",
            "object": "green_cube",
            "target": "white_cube",
            "features": {
                "candidate_meta": {
                    "reason": "no_reachable_candidates",
                    "num_candidates": len(cands)
                }
            },
            "success": False,
            "fail_type": "no_reachable_candidates",
            "msg": "No reachable candidates",
        })
        print(f"[EP {ep}] No reachable candidates for stacking")
        return False

    # Build attempt list (top-k + random exploration)
    attempt_list, K = build_attempt_list(valid, top_k=top_k, explore_extra=explore_extra)

    episode_success = False
    attempts_made = 0

    print(f"[EP {ep}] Attempting {len(attempt_list)} placements (top-{K} + {len(attempt_list)-K} random)")

    for rank, (score, cand) in enumerate(attempt_list):
        if attempts_made >= MAX_PLACE_ATTEMPTS:
            break
        attempts_made += 1

        # Rich metadata for ML training
        meta = {
            "num_candidates": int(len(cands)),
            "num_valid": int(len(valid)),
            "attempt_rank": int(rank),
            "heuristic_score": float(score),
            "attempt_pool": "random" if rank >= K else "topK",
            "top_scores": [float(s) for (s, _) in valid[:min(8, len(valid))]],
            "top_xyz": [c.tolist() for (_, c) in valid[:min(8, len(valid))]],
        }

        ok_place, _ = ctrl.place_object("white_cube", 
                                       desired_obj_pos=cand, 
                                       candidate_meta=meta)

        if ok_place:
            episode_success = True
            print(f"  [Attempt {rank+1}/{len(attempt_list)}] Success (score={score:.6f})")
            if not continue_after_success:
                return True
        else:
            print(f"  [Attempt {rank+1}/{len(attempt_list)}] Failed (score={score:.6f})")

        # Repick for next attempt (object likely dropped)
        if rank != len(attempt_list) - 1:
            repicked = repick_object(ctrl, "green_cube")
            if not repicked:
                print(f"  [EP {ep}] Failed to repick, stopping attempts")
                break

    return episode_success


def run_table_placement_episode(ep: int, seed: int, log_path: str,
                                top_k: int = TOP_K,
                                explore_extra: int = EXPLORE_EXTRA):
    """
    Run a table placement episode: pick green cube, place on table.
    
    This provides easier examples for the ML model to learn from.
    
    Args:
        ep: Episode number
        seed: Random seed
        log_path: Path to JSONL log file
        top_k: Number of top candidates to try
        explore_extra: Number of random candidates to try
    
    Returns:
        True if placement succeeded
    """
    ctrl, table_id, white_id, green_id = setup_world(seed)
    ctrl.log_path = log_path
    ctrl.set_episode(ep, seed)

    ctrl.wait(1.0)

    # Pick green cube
    ok_pick, _ = ctrl.pick_object("green_cube")
    if not ok_pick:
        print(f"[EP {ep}] Failed to pick green cube")
        return False

    # Generate table placement candidates
    cands = ctrl.generate_table_candidates(ctrl.held_object_id, table_id, 
                                          num=RADIAL_SAMPLES)

    # Score candidates
    scored = [(ctrl.score_candidate_heuristic(c), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    valid = [(s, c) for (s, c) in scored if s > -1e8]

    if not valid:
        print(f"[EP {ep}] No reachable candidates for table placement")
        return False

    # For table placement, just try the best candidate
    score, cand = valid[0]
    
    meta = {
        "num_candidates": int(len(cands)),
        "num_valid": int(len(valid)),
        "attempt_rank": 0,
        "heuristic_score": float(score),
        "attempt_pool": "topK",
        "placement_type": "table",
    }

    ok_place, _ = ctrl.place_object("the_table", 
                                   desired_obj_pos=cand, 
                                   candidate_meta=meta)

    if ok_place:
        print(f"[EP {ep}] Table placement succeeded")
    else:
        print(f"[EP {ep}] Table placement failed")

    return ok_place


def main(n_episodes: int = 100,
         base_seed: int = 1000,
         stacking_ratio: float = 0.7,
         top_k: int = TOP_K,
         explore_extra: int = EXPLORE_EXTRA,
         continue_after_success: bool = True,
         use_gui: bool = False):
    """
    Main data collection loop.
    
    Args:
        n_episodes: Total number of episodes to run
        base_seed: Starting seed (each episode gets base_seed + ep)
        stacking_ratio: Fraction of episodes that are stacking (rest are table placement)
        top_k: Number of top candidates to try per episode
        explore_extra: Number of random candidates for exploration
        continue_after_success: Keep trying candidates after first success (for more data)
        use_gui: Show PyBullet GUI (useful for debugging)
    """
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RUNS_DIR, f"attempts_{run_tag}.jsonl")
    
    print("=" * 70)
    print("Self-Supervised Data Collection for Reachability Prediction")
    print("=" * 70)
    print(f"Episodes:          {n_episodes}")
    print(f"Stacking ratio:    {stacking_ratio:.1%}")
    print(f"Top-K candidates:  {top_k}")
    print(f"Explore extra:     {explore_extra}")
    print(f"Continue after OK: {continue_after_success}")
    print(f"Log file:          {log_path}")
    print("=" * 70)

    mode = p.GUI if use_gui else p.DIRECT
    p.connect(mode)

    successes = 0
    stacking_successes = 0
    stacking_attempts = 0
    table_successes = 0
    table_attempts = 0

    start_time = time.time()

    for ep in range(n_episodes):
        seed = base_seed + ep

        # Mix stacking and table placement episodes
        is_stacking = random.random() < stacking_ratio

        if is_stacking:
            stacking_attempts += 1
            ok = run_stacking_episode(ep, seed, log_path,
                                     top_k=top_k,
                                     explore_extra=explore_extra,
                                     continue_after_success=continue_after_success)
            if ok:
                stacking_successes += 1
                successes += 1
        else:
            table_attempts += 1
            ok = run_table_placement_episode(ep, seed, log_path,
                                            top_k=top_k,
                                            explore_extra=explore_extra)
            if ok:
                table_successes += 1
                successes += 1

        # Progress report
        if (ep + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (ep + 1) / elapsed
            eta = (n_episodes - ep - 1) / rate if rate > 0 else 0
            
            print(f"\n[{ep+1}/{n_episodes}] Progress Report:")
            print(f"  Overall success:  {successes}/{ep+1} ({100*successes/(ep+1):.1f}%)")
            if stacking_attempts > 0:
                print(f"  Stacking success: {stacking_successes}/{stacking_attempts} ({100*stacking_successes/stacking_attempts:.1f}%)")
            if table_attempts > 0:
                print(f"  Table success:    {table_successes}/{table_attempts} ({100*table_successes/table_attempts:.1f}%)")
            print(f"  Rate: {rate:.1f} ep/s | ETA: {eta:.0f}s")

    p.disconnect()

    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Data Collection Complete!")
    print("=" * 70)
    print(f"Total episodes:    {n_episodes}")
    print(f"Total successes:   {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Stacking:          {stacking_successes}/{stacking_attempts} ({100*stacking_successes/stacking_attempts:.1f}%)")
    print(f"Table placement:   {table_successes}/{table_attempts} ({100*table_successes/table_attempts:.1f}%)")
    print(f"Time elapsed:      {elapsed:.1f}s ({elapsed/n_episodes:.2f}s per episode)")
    print(f"Dataset saved to:  {log_path}")
    print("=" * 70)
    print("\nNext step: Run `python train_place_model.py` to train the ML model!")


if __name__ == "__main__":
    # Quick GUI sanity check (10 episodes with visualization)
    # main(n_episodes=10, base_seed=1000, stacking_ratio=0.7, 
    #      top_k=3, explore_extra=2, continue_after_success=True, use_gui=True)

    # SINGLE-TRY data collection (matches deployment conditions!)
    # This fixes the train/test mismatch identified in ablation studies
    print("\n🔧 SINGLE-TRY MODE: Collecting data that matches deployment conditions")
    print("   - top_k=1: Only try the best candidate (no backup tries)")
    print("   - explore_extra=0: No random exploration")
    print("   - This should give ~60% success rate (matching deployment)\n")
    
    main(n_episodes=100, base_seed=2000, stacking_ratio=0.7,
         top_k=1, explore_extra=0, continue_after_success=False, use_gui=False)
    
    # Old multi-try mode (for reference - this caused the train/test mismatch)
    # main(n_episodes=100, base_seed=1000, stacking_ratio=0.7,
    #      top_k=5, explore_extra=4, continue_after_success=True, use_gui=False)

