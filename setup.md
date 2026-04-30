# Setup Notes — SO-ARM 101 / LeRobot

## Environment

| Item | Value |
|---|---|
| Conda env | `lerobot` |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu130 |
| LeRobot | 0.5.2 (commit `b5f65e5`) |

## Port IDs

| Arm | Port | ID used in config |
|---|---|---|
| Leader | COM6 | `my_leader` |
| Follower | COM5 | `my_follower` |

## Cameras

| Camera | Mount | ID / path |
|---|---|---|
| Camera 1 | TBD | TBD |
| Camera 2 | TBD | TBD |

*(Fill in after Step 10 — lerobot-teleoperate with cameras enabled)*

## Calibration

```bash
lerobot-calibrate \
  --robot-path lerobot/configs/robot/so101.yaml \
  --robot-overrides '~motors' \
    motors.leader_arm.port=COM6 \
    motors.follower_arm.port=COM5
```

## Notes

- Calibration is per-PC and per-cable — redo if you switch machines or USB hubs.
- Pin this LeRobot version end-to-end; CLI commands change between releases.
