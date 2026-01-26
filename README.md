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

### Model Performance

**Dataset Statistics:**
- 200+ placement attempts logged
- ~60% success rate (realistic for challenging stacking tasks)
- Balanced classes using `class_weight='balanced'`

**Random Forest Classifier:**
```
AUC: 0.85+
Precision: 0.75-0.80
Recall: 0.70-0.80
```

**Top Feature Importances:**
1. `heuristic_score` (IK/FK error) — strongest predictor
2. `des_x`, `des_y` — workspace position matters
3. `des_z` — height is critical for reachability
4. `obj_half_h` — taller objects harder to place

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