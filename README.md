# Diffusion Policy Hackathon Plan

End-to-end pipeline on SO-ARM 101 — 3 to 5 day self-hackathon — prepared for Bingjian Huang

## Overall Strategy

Given the 2060's 6 GB VRAM will almost certainly bottleneck a real diffusion-policy run, smoke-test locally to confirm the pipeline is wired up end-to-end, then rent an A10/A100 for roughly $5-10 of cloud spend for the actual training run.

**Learning emphasis matters more than shipping.** Each phase has a *learning goal* — the concept to walk away owning, not just the artifact produced. Spend ten minutes at the end of each phase journaling what surprised you; that's where the real value compounds.

**Critical path:** Setup → Calibration → Demos → Dataset sanity → Smoke train → Cloud train → Real-robot eval → Iterate.

## Environment Quick-Reference

| Item | Value |
|---|---|
| Conda env | `lerobot` |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu130 |
| LeRobot | 0.5.2 |
| Leader port | COM6 (`my_leader`) |
| Follower port | COM5 (`my_follower`) |

Full setup notes → [setup.md](setup.md)

### Risk hot spots (where hackathons usually die)

- Inconsistent lighting / camera placement across demo sessions → policy doesn't generalize.
- Action-space mismatch (joint vs. EE, absolute vs. delta) → train loss looks fine, real rollouts fail.
- No eval harness → can't tell if changes are helping.
- Underestimating demo collection time. Rule of thumb: ~1 min per demo with reset, so 50 demos = ~1.5h of arm time, plus ~2x that for setup and redos.

---

## Phase 0 — Environment Setup

**Goal.** Reproducible LeRobot install on this PC, both arms talking, both cameras streaming.

**Learning goal.** *The hardware/software contract — which abstractions LeRobot owns vs. what you control.*

**Time:** 3-4h

### TODO

- [ ] Decide OS path. **Recommendation:** dual-boot Ubuntu 22.04. WSL2 + usbipd-win works for cameras, but Dynamixel/Feetech serial-bus passthrough is flaky and burns ~2h debugging. Native Windows is worse. *(0.5-2h depending on path)*
- [ ] Clone LeRobot, install via uv or conda, install SO-ARM 101 extras (`pip install -e ".[feetech]"`). Check the docs page for SO-ARM 101 — CLI was reorganized recently. *(0.5h)*
- [x] Run `python -m lerobot.find_port` for both leader and follower buses; record the port IDs. *(0.25h)*
  - Follower arm: `COM5`
  - Leader arm: `COM6`
- [ ] Calibrate both arms (`lerobot-calibrate`). Even though hardware is "ready," calibration is per-PC and per-cable. *(0.5h)*
- [ ] Mount cameras: 1 fixed overhead/side + 1 wrist cam if available. **Tape camera positions to the table** — if they move between sessions, your dataset is poisoned. *(0.5-1h)*
- [ ] Run `lerobot-teleoperate` with cameras enabled. Verify smooth tracking and clean image streams. *(0.25h)*

  ```bash
  lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=COM5 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=COM6 \
    --teleop.id=my_leader \
    --display_data=true
  ```

### Deliverables

- Teleop video: 30s of you driving the follower with the leader, both cam feeds visible.
- ~~A `setup.md` in your repo listing exact versions, port IDs, calibration notes — future-you will thank you.~~ **Done** — see [setup.md](setup.md).

---

## Phase 1 — Task Definition & Demo Collection

**Goal.** 50 high-quality demonstrations of the pick-and-place task.

**Learning goal.** *Why data quality dominates everything else in BC. You already know this from Meta — but doing it yourself for a small task makes the lesson visceral.*

**Time:** 5-7h

### TODO

- [x] **Write a one-paragraph task spec** before recording anything. *(0.25h)*

  > Pick a 3cm red plastic cube from anywhere in a 15x15cm zone (bounded by transparent tape) on the left, place it inside a white square box on the right. Cube initial orientation arbitrary. Episode ends when cube is stationary in target zone.
- [ ] Tape down: workspace bounds, initial cube zone, target zone. Mark camera positions. *(0.25h)*
- [ ] Record 5 throwaway practice demos with `lerobot-record` to develop your own teleop muscle memory. **Do not use these for training.** *(0.5h)*
- [ ] Record 50 real demos. Vary cube initial pose across the full zone. **Reset between every demo with consistent lighting.** Aim for ~20s episodes. *(2-3h including breaks — fatigue kills demo quality)*

  ```bash
  lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM5 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=COM6 \
    --teleop.id=my_leader \
    --display_data=true \
    --dataset.repo_id=HALDijkstraaa/so101_pickplace \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick the cube and place it in the white box" \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5 \
    --dataset.fps=30 \
    --dataset.push_to_hub=true \
    --dataset.private=true
  ```

- [ ] Visualize the uploaded dataset at [https://huggingface.co/spaces/lerobot/visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset). *(0.5h)*
- [ ] Sanity checks: action histograms not clipped at limits, no zero-length episodes, no episodes where you fumbled and recovered (those teach the policy bad recovery habits — re-record). *(0.5h)*
- [ ] Optional but worth it: record 10 extra demos with deliberately varied lighting / minor distractors. Compare downstream eval with vs. without these later. *(1h)*

### Troubleshooting

**`urllib.error.URLError: <urlopen error [WinError 10054] An existing connection was forcibly closed by the remote host>`**
This is an IPv6 issue on Windows. Fix: disable IPv6 on the Wi-Fi adapter in Network Adapter Settings.
See [huggingface_hub#2043](https://github.com/huggingface/huggingface_hub/issues/2043) for details.

### Deliverables

- HuggingFace dataset on disk with 50 (or 60) episodes: [HALDijkstraaa/so101_pickplace_20260501_134521](https://huggingface.co/datasets/HALDijkstraaa/so101_pickplace_20260501_134521)
- Visualization screenshots showing diverse cube positions across the zone.
- Action statistics printout.

---

## Phase 2 — Smoke Test & Cloud Training

**Goal.** Diffusion policy training to convergence on cloud, with confidence the pipeline isn't silently broken.

**Learning goal.** *Diffusion policy's three knobs — observation horizon, prediction horizon, action execution horizon — and what each controls in practice.*

**Time:** 4-5h active (training itself runs in background)

### TODO

- [ ] **Local smoke run on 2060:** `lerobot-train` with `policy=diffusion`, `batch_size=8`, `steps=2000`. Verify loss decreases, checkpoints save, no OOM. If OOM at bs=8, drop image resolution to 96x96. *(1h)*
- [ ] Read the policy config you're using. Specifically know your: `n_obs_steps`, `horizon`, `n_action_steps`, action space (joint vs. EE), normalization mode. *(0.5h — this is the conceptual crux)*
- [ ] Spin up cloud GPU. **Recommendation:** RunPod or Lambda — A10 (24 GB) at ~$0.75/hr is enough for SO-ARM 101 with default config. A100 only if you're impatient. *(0.5h first time)*
- [ ] Sync dataset to cloud (`rsync` or `huggingface-cli upload` to a private repo). *(0.25h)*
- [ ] Launch full training: 100k steps, batch size 8. **Use Weights & Biases logging** — don't fly blind. *(setup 0.5h, train runs in background)*

  ```powershell
  lerobot-train `
    --dataset.repo_id=HALDijkstraaa/so101_pickplace `
    --dataset.root="$env:USERPROFILE\.cache\huggingface\lerobot\HALDijkstraaa\so101_pickplace_20260501_134521" `
    --policy.type=act `
    --output_dir=outputs/train/act_so101_pickplace `
    --job_name=act_so101_pickplace `
    --policy.device=cuda `
    --batch_size=8 `
    --steps=100000 `
    --policy.push_to_hub=false `
    --wandb.enable=true
  ```
- [ ] While training runs: build the eval harness (Phase 3 prep). Don't waste the wall clock.
- [ ] Pull the best checkpoint back to local. *(0.25h)*

### Deliverables

- Loss curve screenshot showing convergence (`train/l1_loss`, 100k steps, ACT on RTX 3060):

  ![W&B loss curve](outputs/train/act_so101_pickplace/W%26B%20Chart%205_2_2026%2C%207_23_50%20PM.png)
- Trained `.safetensors` checkpoint on local PC.
- A note in your journal answering: "If I doubled `n_action_steps`, what would change about robot behavior?"

### Decision gate

> Before paying for cloud: if the local 2000-step smoke run takes >45 min, the 2060 is hopeless for the real run — go cloud. If it takes <15 min, you *could* try a shorter local run (50k steps, ~10-12h overnight), but cloud is still faster and barely costs anything.

---

## Phase 3 — Real-Robot Evaluation

**Goal.** Honest performance number on N>=20 trials, with characterized failure modes.

**Learning goal.** *The sim-real (or train-deploy) gap is real even without sim. Distribution shift between demo and rollout is where BC bleeds.*

**Time:** 4-5h

### TODO

- [ ] Run rollout on the real robot using `lerobot-rollout`, log per-trial success + failure mode + cube initial pose. *(1-1.5h)*

  ```powershell
  lerobot-rollout `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=my_follower `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --robot.max_relative_target=10 `
    --display_data=true `
    --policy.path=outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model `
    --policy.device=cuda `
    --strategy.type=sentry `
    --inference.type=sync `
    --dataset.repo_id=HALDijkstraaa/rollout_act_so101_pickplace `
    --dataset.root="$env:USERPROFILE\.cache\huggingface\lerobot\HALDijkstraaa\so101_pickplace" `
    --dataset.single_task="Pick the cube and place it in the white box" `
    --dataset.num_episodes=1 `
    --dataset.episode_time_s=25 `
    --dataset.reset_time_s=10 `
    --dataset.fps=30 `
    --dataset.push_to_hub=false
  ```

  > **Note:** Use `--policy.path` (not `--policy.pretrained_path`) so the checkpoint's `config.json` is loaded automatically with the correct input features.
- [ ] Define success criterion *before* running evals (e.g., "cube fully inside target square within 30s"). *(0.1h, but critical for honesty)*
- [ ] Run 20 trials with cube positions sampled across the training zone. Record video of every trial — you'll want to re-watch failures. *(1.5h)*
- [ ] Run 10 trials with cube positions slightly *outside* the trained zone. This tells you the generalization story. *(0.75h)*
- [ ] Categorize failures: approach errors? grasp errors? trajectory drift? releases too early? Don't just report a success rate — the type of failure tells you what to fix. *(0.5h)*

### Observations

**First rollout worked on the first try.** Lighting changed at different times of day — adjusted camera config and it recovered.

**What generalizes well:**
- Positions: reliably picks up cubes from the 5 training locations; many positions *inside* the boundary also work.
- Cube face: trained with "simulator" face on top only, but the policy handles other faces on top as well.

**What doesn't generalize:**
- Orientation: rotating the cube to different horizontal directions causes frequent failure — expected, since only one orientation was collected.
- Position: anywhere outside the taped boundary fails — the policy cannot extrapolate.

**Key insight — no recovery data:** When the arm misses the cube and starts moving toward the box empty-handed, it has no idea how to recover because it never saw that situation in training. Need to collect deliberate "recovery from failure" demos in the next iteration.

**The three failure modes map onto the three classic limitations of behavioral cloning:**

1. **Demo distribution coverage** (orientation failure) — the policy only knows what you showed it. Unseen orientations are out-of-distribution by definition.
2. **Geometric extrapolation** (out-of-zone positions) — neural nets interpolate well but extrapolate poorly. The boundary of the tape is roughly the boundary of the policy's competence.
3. **State distribution shift / compounding errors** (recovery failure) — a small mistake puts the robot in a state it has never seen; errors compound rather than self-correct, and the policy spirals further off-distribution with each step.

Failure mode (3) is the hardest to internalize from reading alone. Feeling it firsthand — watching the arm confidently move to the box with nothing in its gripper — is the visceral version of the lesson.

### Deliverables

- Success-rate table (in-distribution vs. shifted).
- Failure-mode breakdown.
- ~3 video clips of representative failures.
- One-sentence hypothesis for the top failure mode.

---

## Phase 4 — One Iteration Cycle

**Goal.** Prove you can close the loop — diagnose a failure, intervene, measure improvement.

**Learning goal.** *This is the actual research skill. Anyone can run a recipe once.*

**Time:** 4-6h

### TODO

- [ ] **If failure is data coverage** → use DAgger to collect on-the-fly correction demos, retrain, re-eval. *(4-5h)*

  ```powershell
  $env:HF_HUB_OFFLINE=1; lerobot-rollout `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=my_follower `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --robot.max_relative_target=10 `
    --display_data=true `
    --policy.path=outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model `
    --policy.device=cuda `
    --teleop.type=so101_leader `
    --teleop.port=COM6 `
    --teleop.id=my_leader `
    --strategy.type=dagger `
    --strategy.record_autonomous=false `
    --strategy.input_device=keyboard `
    --strategy.keyboard.pause_resume=space `
    --strategy.keyboard.correction=c `
    --strategy.keyboard.upload=u `
    --inference.type=sync `
    --dataset.repo_id=HALDijkstraaa/rollout_act_so101_pickplace_dagger_test `
    --dataset.root="$env:USERPROFILE\.cache\huggingface\lerobot\HALDijkstraaa\rollout_act_so101_pickplace_dagger_test_20260503_110246" `
    --dataset.single_task="Pick the cube and place it in the white box" `
    --dataset.num_episodes=20 `
    --dataset.episode_time_s=30 `
    --dataset.reset_time_s=15 `
    --dataset.fps=30 `
    --dataset.push_to_hub=false `
    --resume=true
  ```

  **Keyboard workflow:**
  1. Watch the policy run autonomously
  2. Press `SPACE` to pause the policy when it's about to fail
  3. Move the leader arm to roughly the same position as the follower
  4. Press `C` to start recording the correction
  5. Guide the arm through the correct motion with the leader
  6. Press `C` again to stop recording
  7. Press `SPACE` to resume the policy

- [ ] **If failure is action-space related** → swap joint/EE action space or absolute/delta, retrain, re-eval. *(3-4h)*
- [ ] **If failure is camera / visual** → add wrist cam (if not already), retrain, re-eval. *(4-6h)*
- [ ] **Dataset merge + retrain on combined data** — strip `intervention` feature from DAgger dataset, merge with original, retrain.

  **Step 1 — remove `intervention` feature from DAgger dataset:**

  ```powershell
  lerobot-edit-dataset `
    --repo_id=HALDijkstraaa/rollout_act_so101_pickplace_dagger_test `
    --root="$env:USERPROFILE\.cache\huggingface\lerobot\HALDijkstraaa\rollout_act_so101_pickplace_dagger_test_20260503_110246" `
    --new_repo_id=HALDijkstraaa/rollout_act_so101_pickplace_dagger_cleaned `
    --new_root=outputs/datasets/dagger_cleaned `
    --operation.type=remove_feature `
    --operation.feature_names="['intervention']" `
    --push_to_hub=false
  ```

  **Step 2 — merge cleaned DAgger + original (50 + 21 = 71 episodes):**

  ```powershell
  lerobot-edit-dataset `
    --new_repo_id=HALDijkstraaa/so101_pickplace_merged `
    --new_root=outputs/datasets/so101_pickplace_merged `
    --operation.type=merge `
    --operation.repo_ids="['HALDijkstraaa/so101_pickplace', 'HALDijkstraaa/rollout_act_so101_pickplace_dagger_cleaned']" `
    --operation.roots="['$env:USERPROFILE\.cache\huggingface\lerobot\HALDijkstraaa\so101_pickplace_20260501_134521', 'outputs/datasets/dagger_cleaned']" `
    --push_to_hub=false
  ```

  **Step 3 — retrain ACT on merged dataset:**

  ```powershell
  lerobot-train `
    --dataset.repo_id=HALDijkstraaa/so101_pickplace_merged `
    --dataset.root=outputs/datasets/so101_pickplace_merged `
    --policy.type=act `
    --output_dir=outputs/train/act_so101_pickplace_v2 `
    --job_name=act_so101_pickplace_v2 `
    --policy.device=cuda `
    --batch_size=8 `
    --steps=60000 `
    --policy.push_to_hub=false `
    --wandb.enable=true
  ```

- [ ] Document before/after numbers and what you learned. *(0.5h)*

### Observations

**DAgger data added:** 8 samples for new positions, 4 for orientations, 8 for recovery from failed pickup (21 episodes total).

**Overall result: v2 performed worse than v1.**

**What worked:**
- Generalized to two new positions at the intersections of major training positions.
- Learned partial recovery behavior — when it fails to pick up, it now returns to the cube faster instead of resetting to home position first. This gives it more chances to reattempt.

**What didn't work:**
- Orientation samples backfired. Showing the robot multiple grasp angles for different orientations confused it about the correct way to grasp in the standard case. It now fails pickups where plenty of clean data existed before.

**Summary:** Found that adding small amounts of intervention data via HG-DAgger had heterogeneous effects: 8 recovery samples in previously-uncovered states improved performance, but 4 orientation samples — covering visual states already in the dataset but with inconsistent action labels — degraded performance overall. This is consistent with the multimodal action distribution problem that motivates approaches like IWR and Diffusion Policy, and with the broader finding that BC is more sensitive to data consistency than data quantity at small scales.

**Key lesson — small noisy data can pollute clean data.** A few samples of high-variance behavior (many different grasp angles) can override the consistent signal from 50 well-collected demos. This is the data-quality-over-quantity principle from Phase 1 showing up in a different form: it's not just about quantity, it's about *consistency*. If you add DAgger data for a new behavior, you need enough samples to form a reliable signal — not just 4.

### Deliverables

- Second eval table showing the delta.
- A short writeup (~200 words) explaining the change and result — useful for future-you and shareable.
- Loss curve for v2 (merged dataset: 50 original + 21 DAgger episodes, ACT, ~60k steps):

  ![W&B loss curve v2](outputs/train/act_so101_pickplace_v2/W%26B%20Chart%205_3_2026%2C%206_57_03%20PM.png)

---

## Stretch Phase — Stacking or Push-T

**Goal.** Generalize the pipeline to a second task with minimal new infrastructure.

**Time:** 6-8h if you have a Day 4-5

The pipeline is the same; only the task spec, demos, and config change. Stack-two-cubes is a natural progression because it's roughly 2x the horizon and tests longer-context behavior. Push-T needs a printed/foam T plus a flat workspace — slightly more setup, but it's the canonical DP benchmark and gives you a direct read on whether your setup matches the literature.

**Recommendation:** stacking over Push-T for a hackathon — existing fixtures mostly transfer.

---

## Total Time Budget

| Phase | Active hours | Wall-clock |
|---|---|---|
| Phase 0 — Setup | 3-4 | 3-4h |
| Phase 1 — Demos | 5-7 | 5-7h |
| Phase 2 — Train | 4-5 active | +3-5h compute in background |
| Phase 3 — Eval | 4-5 | 4-5h |
| Phase 4 — Iterate | 4-6 | 4-6h (incl. retrain) |
| **Core total** | **20-27** | **~3 days** |
| Stretch | 6-8 | +1 day |

*Comfortable in 4 days at 7-8h/day, with Day 5 as buffer or stretch.*

---

## A Few Things to Watch For

- **LeRobot is moving fast.** CLI commands have changed names recently. Pin your commit hash in `setup.md` and use that exact version end-to-end.
- **Don't optimize before measuring.** You'll be tempted to tune DP hyperparameters before running evals. Don't. Default config first, eval, *then* iterate.
- **Stop demo collection when bored, not when hitting a number.** Tired demos are bad demos. 40 fresh demos beat 60 fatigued ones.