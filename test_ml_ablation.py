"""
Ablation Study: Compare Heuristic-Only vs. ML-Guided Candidate Selection

This script tests whether adding ML improves success rate beyond the 
heuristic-only baseline (which already has motion planning fixes).

Configuration:
- Baseline: Heuristic-only candidate selection (current system)
- Treatment: ML-guided candidate selection

Both use the SAME motion planner (with all fixes from commit c057532).
"""

import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import joblib
from robot_controller import RobotController
from language_parser import CommandParser
from perception import PerceptionModule

# Configuration
USE_GUI = False
TRIALS_PER_CONFIG = 10
USE_ML_MODEL = True  # Toggle this to switch between heuristic and ML

def setup_scenario(seed):
    """Create a fresh simulation with robot + table + cubes."""
    if p.isConnected():
        p.resetSimulation()
    else:
        p.connect(p.GUI if USE_GUI else p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", [1.25, 0, 0], useFixedBase=True)
    
    # Get table height
    t_min, t_max = p.getAABB(table_id)
    table_z = t_max[2]
    
    # Random positions on table
    np.random.seed(seed)
    green_pos = [0.65 + np.random.uniform(-0.05, 0.05), 
                 np.random.uniform(-0.15, 0.15), 
                 table_z + 0.03]
    white_pos = [0.65 + np.random.uniform(-0.05, 0.05), 
                 np.random.uniform(-0.15, 0.15), 
                 table_z + 0.03]
    
    green_id = p.loadURDF("cube_small.urdf", green_pos, useFixedBase=False)
    p.changeVisualShape(green_id, -1, rgbaColor=[0, 1, 0, 1])
    
    white_id = p.loadURDF("cube_small.urdf", white_pos, useFixedBase=False)
    p.changeVisualShape(white_id, -1, rgbaColor=[1, 1, 1, 1])
    
    # Settle
    for _ in range(50):
        p.stepSimulation()
    
    object_map = {
        "white_cube": white_id,
        "green_cube": green_id,
        "table": table_id,
        "the_table": table_id
    }
    
    perception = PerceptionModule(robot_id, object_map)
    parser = CommandParser()
    controller = RobotController(robot_id, 9, 10, perception, parser)
    controller.table_top_z = float(table_z)
    
    return controller

def generate_candidates_with_scores(controller, target_name):
    """
    Generate placement candidates and score them with both heuristic and ML.
    
    Returns: list of (xyz, heuristic_score, ml_score) tuples
    """
    target_id = controller.perception.object_map.get(target_name)
    if target_id is None:
        return []
    
    # Get target info
    is_table = target_name in ("table", "the_table")
    
    if controller.held_object_id is None:
        return []
    
    held_id = controller.held_object_id
    o_min, o_max = p.getAABB(held_id)
    obj_half_h = 0.5 * float(o_max[2] - o_min[2])
    
    if is_table:
        # Table placement candidates (radial grid)
        target_xy = np.array(controller.table_drop_xy, dtype=float)
        target_top_z = controller.table_top_z
        clearance = 0.010
        
        candidates = []
        for r in [0.0, 0.04, 0.08, 0.12]:
            for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                angle = np.radians(angle_deg)
                x = target_xy[0] + r * np.cos(angle)
                y = target_xy[1] + r * np.sin(angle)
                z = target_top_z + obj_half_h + clearance
                candidates.append([x, y, z])
    else:
        # Stacking candidates (dense grid)
        target_xy, target_top_z = controller._stack_target_xy_topz(target_id)
        clearance = 0.016
        
        candidates = []
        for dx in [-0.024, -0.012, 0.0, 0.012, 0.024]:
            for dy in [-0.024, -0.012, 0.0, 0.012, 0.024]:
                x = target_xy[0] + dx
                y = target_xy[1] + dy
                z = target_top_z + obj_half_h + clearance
                candidates.append([x, y, z])
    
    # Score each candidate
    scored_candidates = []
    for xyz in candidates:
        # Heuristic score (IK->FK error)
        ok, reason = controller.is_reachable(xyz, use_down_orientation=False)
        if not ok:
            continue  # Skip unreachable candidates
        
        # Compute FK error as heuristic score
        jt = controller._ik(xyz, use_down_orientation=False)
        fk_error = controller._fk_error_for_joints(jt, xyz)
        heuristic_score = -fk_error  # Negative so higher is better
        
        # ML score (if model available)
        ml_score = 0.0
        if USE_ML_MODEL and os.path.exists("models/place_rf.joblib"):
            model_data = joblib.load("models/place_rf.joblib")
            # Model is saved as dict with metadata
            if isinstance(model_data, dict):
                model = model_data['model']
            else:
                model = model_data
            
            features = [
                xyz[0],  # des_x
                xyz[1],  # des_y
                xyz[2],  # des_z
                heuristic_score,  # heuristic_score
                target_top_z,  # target_top_z
                obj_half_h,  # obj_half_h
                clearance  # clearance
            ]
            ml_score = model.predict_proba([features])[0][1]  # Probability of success
        
        scored_candidates.append((xyz, heuristic_score, ml_score))
    
    return scored_candidates

def select_best_candidate(scored_candidates, use_ml=False):
    """
    Select best candidate based on heuristic or ML score.
    
    Args:
        scored_candidates: list of (xyz, heuristic_score, ml_score)
        use_ml: if True, use ml_score; if False, use heuristic_score
    
    Returns:
        best_xyz or None
    """
    if not scored_candidates:
        return None
    
    if use_ml:
        # Sort by ML score (highest first)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
    else:
        # Sort by heuristic score (highest first, i.e., lowest FK error)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return scored_candidates[0][0]  # Return xyz of best candidate

def run_trial(trial_num, seed, use_ml=False):
    """Run a single trial with specified configuration."""
    try:
        controller = setup_scenario(seed)
        controller.wait(0.5)
        
        # Pick green cube
        success, msg = controller.pick_object("green_cube")
        if not success:
            print(f"  Trial {trial_num}: Pick failed - {msg}")
            return False
        
        # Generate and score candidates
        scored_candidates = generate_candidates_with_scores(controller, "white_cube")
        if not scored_candidates:
            print(f"  Trial {trial_num}: No valid candidates")
            return False
        
        # Select best candidate
        best_xyz = select_best_candidate(scored_candidates, use_ml=use_ml)
        if best_xyz is None:
            print(f"  Trial {trial_num}: No candidate selected")
            return False
        
        # Place using selected candidate
        success, msg = controller.place_object("white_cube", desired_obj_pos=best_xyz)
        
        if success:
            print(f"  Trial {trial_num}: Success")
            return True
        else:
            print(f"  Trial {trial_num}: Fail - {msg}")
            return False
            
    except Exception as e:
        print(f"  Trial {trial_num}: Error - {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if p.isConnected():
            p.disconnect()

def main():
    print("=" * 60)
    print("ABLATION STUDY: Heuristic vs. ML Candidate Selection")
    print("=" * 60)
    print()
    
    # Check if ML model exists
    ml_model_exists = os.path.exists("models/place_rf.joblib")
    print(f"ML Model Available: {ml_model_exists}")
    if not ml_model_exists:
        print("WARNING: ML model not found. Run train_place_model.py first!")
        print("Running heuristic-only baseline...")
        print()
    
    # Test 1: Heuristic-Only (Baseline)
    print(f"\n[BASELINE] Heuristic-Only Selection")
    print(f"Running {TRIALS_PER_CONFIG} trials...")
    print("-" * 60)
    
    heuristic_successes = 0
    for i in range(TRIALS_PER_CONFIG):
        seed = 2000 + i
        success = run_trial(i+1, seed, use_ml=False)
        if success:
            heuristic_successes += 1
    
    heuristic_rate = heuristic_successes / TRIALS_PER_CONFIG
    print(f"\nHeuristic-Only Success Rate: {heuristic_successes}/{TRIALS_PER_CONFIG} = {heuristic_rate:.1%}")
    
    # Test 2: ML-Guided (Treatment)
    if ml_model_exists:
        print(f"\n[TREATMENT] ML-Guided Selection")
        print(f"Running {TRIALS_PER_CONFIG} trials...")
        print("-" * 60)
        
        ml_successes = 0
        for i in range(TRIALS_PER_CONFIG):
            seed = 2000 + i  # Same seeds for fair comparison
            success = run_trial(i+1, seed, use_ml=True)
            if success:
                ml_successes += 1
        
        ml_rate = ml_successes / TRIALS_PER_CONFIG
        print(f"\nML-Guided Success Rate: {ml_successes}/{TRIALS_PER_CONFIG} = {ml_rate:.1%}")
        
        # Analysis
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Heuristic-Only: {heuristic_rate:.1%}")
        print(f"ML-Guided:      {ml_rate:.1%}")
        
        improvement = ml_rate - heuristic_rate
        if improvement > 0:
            pct_improvement = (improvement / heuristic_rate) * 100
            print(f"\nML Improvement: +{improvement:.1%} ({pct_improvement:+.1f}% relative)")
        elif improvement < 0:
            print(f"\nML Degradation: {improvement:.1%} (ML performs worse!)")
        else:
            print(f"\nNo difference (both perform equally)")
        
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        if improvement > 0.05:  # >5 percentage points
            print("✅ ML provides meaningful improvement over heuristic-only")
        elif improvement > 0:
            print("⚠️  ML provides small improvement (may not be statistically significant)")
        elif improvement < 0:
            print("❌ ML performs worse than heuristic-only (possible overfitting)")
        else:
            print("➖ ML and heuristic perform equally")
    else:
        print("\nSkipping ML test (model not found)")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("This ablation study isolates the ML contribution by comparing:")
    print("- SAME motion planner (with all fixes)")
    print("- SAME candidate generation")
    print("- DIFFERENT selection strategy (heuristic vs. ML)")
    print("\nAny difference in success rate is attributable to ML.")
    print("=" * 60)

if __name__ == "__main__":
    main()
