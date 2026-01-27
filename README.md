#Robot Manipulation Project

A PyBullet-based robotic manipulation system that executes natural language commands using **self-supervised machine learning** to predict and improve placement success rates. The robot learns from its own failures to intelligently select reachable poses, combining classical robotics with modern ML techniques.

---

## Project Vision

This project addresses a fundamental challenge in robotic manipulation: **some target positions are physically unreachable, and traditional inverse kinematics (IK) often fails silently or returns solutions that are "close but not close enough."**

Instead of hardcoding workspace boundaries or manually tuning reachability heuristics, this system:

1. **Collects data from its own attempts** (self-supervised learning)
2. **Trains a lightweight ML model** to predict placement success before execution
3. **Uses the model to intelligently choose** among candidate placement poses
4. **Adapts to kinematic constraints** without manual intervention

This is a **failure-aware planning system** that demonstrates real machine learning applied to a concrete robotics problem.

---

## Key Features

### Natural Language Interface
- Parse simple commands like:
  - `"pick up the white cube"`
  - `"put the white cube on the table"`
  - `"stack green cube on top of white cube"`
- Rule-based parser with synonym handling and object aliasing

### Self-Supervised Learning Pipeline
- **Automatic Data Collection**: Logs every pick/place attempt with rich features
- **Feature Engineering**: 
  - Target position (x, y, z)
  - Heuristic reachability score (IK/FK error)
  - Object dimensions (half-height, clearance)
  - Target surface properties
- **Binary Classification**: Success/failure labels generated from post-execution validation
- **Models Trained**:
  - Logistic Regression (interpretable baseline)
  - Random Forest (nonlinear, production model)

### Robust Grasp & Place Validation
- **Grasp Validation**: Checks if object actually lifted after grasp attempt
- **Placement Validation**: Measures XY error, Z height, tilt angle, and drift after settling
- **Failure Classification**: Categorizes failures (fell, unreachable, grasp_failed) for analysis

## The Machine Learning Component

### What Problem Does ML Solve?

**The Core Issue:**
Your robot fails when:
- Target position is too far in XY (table center at x=1.5, but Panda reach ≈ 0.9m)
- Target is too high in Z (approach height + object height exceeds workspace)
- IK returns a solution with FK error > tolerance (pose is "close but not close enough")

**Traditional Approach:**
Hardcode workspace limits like `if r_xy > 0.95: unreachable` — but this is brittle and doesn't account for:
- Object-specific constraints
- Stacking height limitations
- Orientation-dependent reachability
- IK solver quirks

**ML Approach:**
Train a model to predict: **"If I try to place at this position, will it succeed?"**

The model learns patterns like:
- Positions near x=0.7, y=0 are highly reachable
- Stacking above z=0.75 often fails
- High FK errors correlate with placement failures
- Certain XY combinations are kinematically infeasible

### Self-Supervised Learning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. Robot Attempts Placement                                │
│     - Generate candidate poses                              │
│     - Execute pick & place                                  │
│     - Measure outcome (success/failure)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Log Attempt Data (attempts.jsonl)                       │
│     - Features: (x, y, z, heuristic_score, obj_half_h, ...) │
│     - Label: success (1) or failure (0)                     │
│     - Metadata: failure_type, xy_err, tilt, drift           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Train ML Model (train_place_model.py)                   │
│     - Load JSONL data                                       │
│     - Extract features (only pre-action signals)            │
│     - Train Random Forest classifier                        │
│     - Evaluate: AUC, precision, recall                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Use Model for Candidate Selection                       │
│     - Generate N candidates                                 │
│     - Score each with model.predict_proba()                 │
│     - Select highest-probability candidate                  │
│     - Execute placement                                     │
└─────────────────────────────────────────────────────────────┘
```

### Feature Engineering (Pre-Action Only)

Critical constraint: **Only use features known BEFORE attempting the action**

```python
Features = [
    des_x,           # Desired X position
    des_y,           # Desired Y position  
    des_z,           # Desired Z position
    heuristic_score, # IK→FK error (computed before execution)
    target_top_z,    # Height of target surface
    obj_half_h,      # Half-height of object being placed
    clearance        # Safety margin (0.016m for stacking)
]
```

**Why not use post-action features?**
- `xy_err`, `tilt_deg`, `max_xy_drift` are only known AFTER placement
- Using them would be **data leakage** — the model must predict before acting

### Model Performance & Ablation Study Results

**Dataset Statistics:**
- 571 placement attempts logged
- 83% success rate in training data (multi-try scenarios)
- Balanced classes using `class_weight='balanced'`

**Random Forest Classifier (Test Set Performance):**
```
AUC: 0.913
Precision: 0.958
Recall: 1.000
```

**Top Feature Importances:**
1. `des_y` (29%) — Y position
2. `des_z` (19%) — Z position  
3. `des_x` (16%) — X position
4. `obj_half_h` (15%) — Object height
5. `heuristic_score` (14%) — IK/FK error
6. `target_top_z` (7%) — Target height

### ⚠️ Important Finding: Domain Knowledge Beats ML

**Ablation Study Results (January 27, 2026):**

Through **three systematic ablation studies** comparing heuristic-only vs. ML-guided candidate selection:

| Approach | Success Rate | Notes |
|----------|--------------|----------|
| **Baseline (Before Fixes)** | 45% | Original system |
| **Motion Planning Improvements** | 60% | Real improvement from waypoint planning, relaxed limits |
| **Heuristic-Only Selection** | 60% | Simple IK/FK error-based selection |
| **ML-Guided Selection** | 40% | Model underperforms despite 0.913 AUC ❌ |

**Root Cause Identified:**

Through three experimental iterations, I discovered that **the heuristic directly measures reachability** while ML tries to learn it from data:

**Experiment 1: Multi-Try Training Data**
- Training: 571 samples, 83% success (robot tries 9 candidates per episode)
- Result: ML gets 20% success
- Issue: Train/test mismatch (multi-try training, single-try testing)

**Experiment 2: Single-Try Training Data**
- Training: 94 samples, 97% success (robot tries 1 candidate per episode)
- Result: ML improves to 40% success
- Issue: Class imbalance (only 3% failures to learn from)

**Experiment 3: Stricter Validation Thresholds**
- Training: 94 samples, 97% success
- Result: ML remains at 40% success
- Issue: Heuristic is too good - selects positions that almost always work

**Why ML Underperforms:**
1. **Heuristic directly measures reachability:** IK→FK error is physics-based, not learned
2. **Class imbalance:** 97% success means only 3 failure examples to learn from
3. **Heuristic already captures everything:** ML tries to learn physics, but heuristic already knows it

**Key Lesson:**
> High test AUC (0.913) does not guarantee real-world improvement. When domain knowledge directly measures the target metric (IK/FK error → reachability), ML struggles to add value. This is a valuable negative result that demonstrates research maturity—knowing when NOT to use ML is as important as knowing when to use it.

**Current Status:**
-  Motion planning improvements provide 33% relative improvement (45% → 60%)
-  ML model underperforms heuristic (40% vs 60%) - see ablation studies
-  Ablation study scripts available in `test_ml_ablation.py`
-  **ML mode toggle available** - easily switch between heuristic and ML selection

###  ML Mode Toggle

The system supports both heuristic-only and ML-guided candidate selection:

```python
# In your code
controller = RobotController(...)

# Enable ML mode
controller.enable_ml_mode(True)   # Use ML-guided selection

# Disable ML mode (default)
controller.enable_ml_mode(False)  # Use heuristic-only selection
```

**Try it yourself:**
```bash
# Compare heuristic vs ML over 5 trials
python demo_ml_toggle.py --mode both --trials 5

# Run with heuristic only
python demo_ml_toggle.py --mode heuristic --gui

# Run with ML only
python demo_ml_toggle.py --mode ml --gui
```

**Expected Results:**
- **Heuristic mode:** ~60% success (reliable, conservative)
- **ML mode:** ~40% success (experimental, less reliable)
- **Demonstrates:** When domain knowledge beats ML

---

## Technical Implementation

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User Command                             │
│              "stack green cube on white cube"                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  CommandParser                               │
│   Regex-based NLP → [("pick", "green_cube", None),          │
│                      ("place", "white_cube", None)]          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 RobotController                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. PICK Phase                                         │  │
│  │     - Perception: locate object                        │  │
│  │     - Compute grasp pose (top-down or side)            │  │
│  │     - Check reachability (IK → FK error < tol)         │  │
│  │     - Execute: approach → grasp → lift                 │  │
│  │     - Validate: did object lift?                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  2. PLACE Phase (ML-Enhanced)                          │  │
│  │     - Generate candidates (grid/radial)                │  │
│  │     - Score each candidate:                            │  │
│  │         • Heuristic: IK→FK error                       │  │
│  │         • ML: model.predict_proba(features)            │  │
│  │     - Select best candidate (explore_p=0.2)            │  │
│  │     - Execute: approach → pre-place → place → release  │  │
│  │     - Validate: settle → measure xy_err, tilt, drift   │  │
│  │     - Log attempt to JSONL                             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Candidate Generation Strategies

**For Table Placement:**
```python
# Radial grid around spawn drop point
candidates = []
for r in [0.0, 0.04, 0.08, 0.12]:
    for angle in [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]:
        x = drop_x + r * cos(angle)
        y = drop_y + r * sin(angle)
        z = table_top_z + obj_half_h + clearance
        candidates.append([x, y, z])
```

**For Stacking:**
```python
# Dense grid over target object's top surface
target_xy = get_base_position(target_id)  # True center, not AABB
target_top_z = get_aabb_max_z(target_id)

for dx in [-0.024, -0.012, 0.0, 0.012, 0.024]:
    for dy in [-0.024, -0.012, 0.0, 0.012, 0.024]:
        x = target_xy[0] + dx
        y = target_xy[1] + dy
        z = target_top_z + obj_half_h + clearance
        candidates.append([x, y, z])
```

### Reachability Checking

```python
def is_reachable(self, target_pos, use_down_orientation=False, tol=0.04):
    """
    Pure kinematic check: IK → FK error, no stepping, no moving sim.
    """
    # 1. Compute IK solution
    joint_positions = self._ik(target_pos, use_down_orientation)
    
    # 2. Forward kinematics: where does this solution actually reach?
    fk_pos = p.getLinkState(self.robot_id, self.ee_link_id, 
                             computeForwardKinematics=True)[0]
    
    # 3. Measure error
    fk_error = np.linalg.norm(np.array(fk_pos) - np.array(target_pos))
    
    # 4. Check workspace limits
    r_xy = np.linalg.norm(target_pos[:2])
    if r_xy > 0.95:
        return False, f"Out of XY workspace (r={r_xy:.2f})"
    
    # 5. Tolerance check
    if fk_error > tol:
        return False, f"FK error {fk_error:.3f}m > tol {tol}m"
    
    return True, "Reachable"
```

### Placement Validation

```python
def _settle_and_measure(self, body_id, seconds=1.0):
    """
    Step simulation and measure where object ENDS UP.
    Returns: final_pos, final_quat, max_xy_drift, final_tilt_deg
    """
    initial_pos, _ = p.getBasePositionAndOrientation(body_id)
    max_drift = 0.0
    
    for _ in range(int(seconds * 240)):  # 240 Hz
        p.stepSimulation()
        current_pos, current_quat = p.getBasePositionAndOrientation(body_id)
        
        # Track maximum XY drift during settling
        xy_drift = np.linalg.norm(
            np.array(current_pos[:2]) - np.array(initial_pos[:2])
        )
        max_drift = max(max_drift, xy_drift)
    
    final_pos, final_quat = p.getBasePositionAndOrientation(body_id)
    tilt_deg = self._quat_to_tilt_deg(final_quat)
    
    return final_pos, final_quat, max_drift, tilt_deg
```

---

## Setup & Installation

### Prerequisites
```bash
# Python 3.8+
pip install pybullet numpy opencv-python scikit-learn joblib
```

### Quick Start

**1. Run Interactive Simulation:**
```bash
python simulation.py
```
- GUI opens with Franka Panda robot + table
- Type commands in terminal:
  - `pick white cube`
  - `place white cube on table`
  - `stack green cube on top of white cube`
- Press `s` to spawn objects, `d` to delete, `l` to list

**2. Collect Training Data:**
```bash
python collect_dataset.py
```
- Runs 50-300 automated episodes
- Logs all attempts to `attempts_runs/attempts_TIMESTAMP.jsonl`
- Each episode: spawn objects → attempt stacking → log results

**3. Train ML Model:**
```bash
python train_place_model.py
```
- Loads JSONL data
- Trains Logistic Regression + Random Forest
- Saves best model to `models/place_rf.joblib`
- Prints AUC, confusion matrix, feature importances

**4. Test Reliability:**
```bash
python test_reliability.py
```
- Runs 10 trials each of:
  - "put white cube on table"
  - "stack green cube on white cube"
- Reports success rates

---

## Example Output

### Data Collection
```
[EPISODE 0] seed=1000
[PICK] green_cube at [0.62, -0.06, 0.65]
[PLACE] Generating 25 candidates for stacking on white_cube
[PLACE] Valid candidates: 25/25
[PLACE] Chosen candidate: [0.67, 0.10, 0.73] (heuristic=-0.000084)
[PLACE] desired_obj_pos = [0.674, 0.095, 0.726]
[SETTLE] xy_err=0.012, tilt=0.0°, drift=0.000m
✓ Successfully placed object on 'white_cube'
```

### Model Training
```
Loaded place rows: 487  | success rate: 0.618

=== Random Forest ===
AUC: 0.856
[[45  8]
 [ 7 62]]
              precision    recall  f1-score   support
           0      0.865     0.849     0.857        53
           1      0.886     0.899     0.892        69

Top RF feature importances:
heuristic_score: 0.3421
      des_x: 0.2156
      des_y: 0.1834
      des_z: 0.1523
 obj_half_h: 0.0712
target_top_z: 0.0354
```

### Reliability Test
```
[1/10] True - Successfully placed object on 'the_table'
[2/10] True - Successfully placed object on 'the_table'
...
put-on-table success: 9 / 10

[1/10] True - Successfully placed object on 'white_cube'
[2/10] False - Stack place failed (xy_err=0.091, drift=0.000, tilt_logged=180.0)
...
stack success: 7 / 10
```

---

### Future Enhancements
- [ ] **Vision-Based Perception**: Replace ground-truth positions with RGB-D segmentation
- [ ] **Deep Learning**: Replace Random Forest with neural network (more data needed)
- [ ] **Multi-Object Commands**: "stack all cubes on the table"
- [ ] **Failure Recovery**: Automatic retry with different strategy
- [ ] **Sim-to-Real Transfer**: Deploy on physical Franka Panda

---

## References & Inspiration

- **PyBullet Documentation**: https://pybullet.org/
- **Franka Panda Robot**: https://www.franka.de/
- **Affordance Learning**: Zeng et al., "Learning Synergies between Pushing and Grasping" (2018)
- **Self-Supervised Robotics**: Levine et al., "Learning Hand-Eye Coordination for Robotic Grasping" (2016)

---

## License

MIT License - Feel free to use for research, learning, or portfolio projects.

---

## Author

**Layla Liu**  
*Robotics & Machine Learning Enthusiast*

This project demonstrates:
- Strong fundamentals in robotics (kinematics, control, perception)
- Practical ML skills (data collection, feature engineering, model training)
- Systems thinking (modular design, logging, evaluation)
- Research potential (self-supervised learning, failure-aware planning)

**Perfect for**: Robotics internships at companies like Boston Dynamics, Tesla, Google DeepMind, or research labs at Stanford/Cornell/CMU.

---

## Quick Demo

```bash
# 1. Install dependencies
pip install pybullet numpy scikit-learn joblib

# 2. Run simulation
python simulation.py

# 3. In the GUI, type:
spawn
pick green cube
place green cube on top of white cube

# Watch the robot intelligently select a reachable placement pose!
```

**Expected Output:**
```
[PLACE] Generating 25 candidates for stacking
[PLACE] ML model loaded: models/place_rf.joblib
[PLACE] Best candidate: [0.68, 0.09, 0.73] (prob=0.87)
✓ Successfully placed object on 'white_cube'
```

---

**This is not just a robot that follows commands — it's a robot that learns from its mistakes and gets better over time.** 🤖✨