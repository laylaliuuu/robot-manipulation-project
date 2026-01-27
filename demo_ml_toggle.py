"""
Demo: Compare Heuristic vs. ML Mode

This script demonstrates the difference between heuristic-only and ML-guided
candidate selection. You can toggle between modes to see how they perform.

Usage:
    python demo_ml_toggle.py --mode heuristic  # Use heuristic-only
    python demo_ml_toggle.py --mode ml         # Use ML-guided
    python demo_ml_toggle.py --mode both       # Compare both (default)
"""

import argparse
import numpy as np
import pybullet as p
import pybullet_data
from robot_controller import RobotController
from perception import PerceptionModule
from language_parser import CommandParser


def setup_scene(seed=1000, use_gui=False):
    """Create a test scene with robot + table + cubes."""
    mode = p.GUI if use_gui else p.DIRECT
    if not p.isConnected():
        p.connect(mode)
    
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load robot and table
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", [1.25, 0, 0], useFixedBase=True)
    
    # Get table height
    t_min, t_max = p.getAABB(table_id)
    table_z = t_max[2]
    
    # Spawn cubes
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
    
    # Setup controller
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


def run_trial(mode="heuristic", seed=1000, use_gui=False):
    """Run a single trial with specified mode."""
    print(f"\n{'='*60}")
    print(f"Running trial with {mode.upper()} mode (seed={seed})")
    print(f"{'='*60}")
    
    controller = setup_scene(seed, use_gui)
    controller.wait(0.5)
    
    # Enable ML mode if requested
    if mode == "ml":
        success = controller.enable_ml_mode(True)
        if not success:
            print("ERROR: ML model not available!")
            return None
    else:
        controller.enable_ml_mode(False)
    
    # Pick green cube
    print("\n[1/2] Picking green cube...")
    success, msg = controller.pick_object("green_cube")
    if not success:
        print(f"  ❌ Pick failed: {msg}")
        return False
    print("  ✓ Pick succeeded")
    
    # Place on white cube
    print("\n[2/2] Placing on white cube...")
    success, msg = controller.place_object("white_cube")
    
    if success:
        print(f"  ✓ Place succeeded!")
        print(f"  Message: {msg}")
        return True
    else:
        print(f"  ❌ Place failed: {msg}")
        return False


def compare_modes(n_trials=5, seed_start=2000):
    """Compare heuristic vs ML over multiple trials."""
    print("\n" + "="*60)
    print("COMPARISON: Heuristic vs. ML Mode")
    print("="*60)
    
    heuristic_results = []
    ml_results = []
    
    for i in range(n_trials):
        seed = seed_start + i
        
        # Test heuristic
        print(f"\n--- Trial {i+1}/{n_trials} ---")
        result = run_trial("heuristic", seed, use_gui=False)
        heuristic_results.append(result)
        
        # Reset and test ML
        if p.isConnected():
            p.disconnect()
        
        result = run_trial("ml", seed, use_gui=False)
        ml_results.append(result)
        
        if p.isConnected():
            p.disconnect()
    
    # Summary
    heur_success = sum(1 for r in heuristic_results if r)
    ml_success = sum(1 for r in ml_results if r)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Heuristic-Only: {heur_success}/{n_trials} = {heur_success/n_trials:.1%}")
    print(f"ML-Guided:      {ml_success}/{n_trials} = {ml_success/n_trials:.1%}")
    
    if ml_success > heur_success:
        print(f"\n✓ ML wins by {ml_success - heur_success} trials!")
    elif ml_success < heur_success:
        print(f"\n✗ Heuristic wins by {heur_success - ml_success} trials")
    else:
        print(f"\n= Tie!")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Demo ML mode toggle")
    parser.add_argument("--mode", choices=["heuristic", "ml", "both"], 
                       default="both",
                       help="Mode to run: heuristic, ml, or both (comparison)")
    parser.add_argument("--trials", type=int, default=5,
                       help="Number of trials for comparison mode")
    parser.add_argument("--seed", type=int, default=2000,
                       help="Random seed")
    parser.add_argument("--gui", action="store_true",
                       help="Show PyBullet GUI")
    
    args = parser.parse_args()
    
    if args.mode == "both":
        compare_modes(n_trials=args.trials, seed_start=args.seed)
    else:
        result = run_trial(args.mode, args.seed, args.gui)
        if result:
            print("\n✓ Trial succeeded!")
        else:
            print("\n✗ Trial failed")
    
    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    main()
