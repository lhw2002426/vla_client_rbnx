# vla_client_rbnx

VLA (Vision-Language-Action) client skill — LLM-callable entry point for end-to-end neural network manipulation.

## Overview

This skill implements a closed-loop VLA control pipeline:

```
Camera topics ─► observe ─► VLA Server (inference) ─► safety filter ─► /arm/pos_cmd ─► piper_ctl ─► arm
                    ▲                                                                          │
                    └──────────────────── joint_states feedback ◄───────────────────────────────┘
```

**Key architecture decision**: Actions are sent **directly to piper_ctl** (`/arm/pos_cmd`), bypassing MoveIt entirely. VLA outputs are already planned trajectories — re-planning through MoveIt would destroy learned behavior and add 3-5s latency per step.

## Setup

```bash
# Clone
git clone https://github.com/lhw2002426/vla_client_rbnx.git

# Build
bash scripts/build.sh

# Requires (on target robot):
# - ROS humble + rclpy
# - vla_server_rbnx running (provides inference)
# - OrbbecSDK_rbnx running (provides camera images)
# - piper_ctl_rbnx running (accepts /arm/pos_cmd)
```

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `vla_server_url` | `http://localhost:8777` | Direct URL fallback |
| `use_atlas_discovery` | `true` | Discover vla_server via atlas |
| `full_image_topic` | `/camera/color/image_raw` | Global camera |
| `wrist_image_topic` | `/wrist_camera/color/image_raw` | Wrist camera |
| `joint_states_topic` | `/arm/joint_states_single` | Joint feedback (proprio for VLA) |
| `end_pose_topic` | `/arm/end_pose` | End-effector pose feedback (Cartesian baseline) |
| `pos_cmd_topic` | `/arm/pos_cmd` | Action output (direct to piper_ctl) |
| `image_resize` | `[256, 256]` | Resize before sending to VLA |
| `action_hz` | `10.0` | Control frequency |
| `timeout_s` | `60.0` | Default execution timeout |
| `enable_safety_filter` | `true` | Joint limits + rate limiting |
| `joint_limits_low/high` | `[-2.618]*6` | Joint angle bounds (rad) |
| `max_delta_per_step` | `[0.1]*6` | Max joint change per step (rad) |
| `gripper_range` | `[0.0, 1.0]` | Gripper value bounds |

## Safety

Since MoveIt is bypassed, the internal safety filter provides:
1. **Joint limits hard clip** — prevents exceeding physical joint bounds
2. **Rate limiting** — prevents single-step jumps that could damage hardware
3. **Gripper range clip** — keeps gripper commands in valid range

Emergency reset: if available, calls `manipulation/reset` (MoveIt joint-space) to park the arm safely.

## Contracts

| Contract ID | Mode | Description |
|-------------|------|-------------|
| `robonix/skill/vla/driver` | rpc | Lifecycle |
| `robonix/skill/vla/execute` | rpc | VLA execution (user_invocable — LLM can call directly) |

## vs pick_skill_rbnx

| | pick_skill | vla_client |
|---|---|---|
| Method | YOLO + geometry + MoveIt planning | End-to-end neural VLA |
| Control | MoveIt (3-5s/step) | Direct pos_cmd (10Hz) |
| Safety | MoveIt collision detection | Internal filter |
| Use case | Simple pick-and-place | Complex language-guided manipulation |
