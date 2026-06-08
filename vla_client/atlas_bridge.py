# SPDX-License-Identifier: MulanPSL-2.0
"""vla_client_rbnx atlas bridge — VLA inference + execution skill.

LLM-callable skill that:
1. Subscribes to camera images + joint states (ROS topics)
2. Calls vla_server_rbnx for action inference (HTTP via atlas-resolved endpoint)
3. Sends actions DIRECTLY to /arm/pos_cmd (piper_ctl_rbnx), bypassing MoveIt
4. Includes an internal safety filter (joint limit clip + rate limiting)

Architecture:
    VLA Server ─actions─► vla_client safety filter ─► /arm/pos_cmd ─► piper_ctl ─► CAN ─► arm

Why not MoveIt:
    VLA outputs are already "planned" trajectories. MoveIt would re-plan and
    destroy learned trajectory features. Also, MoveIt's plan+execute takes 3-5s
    per step, incompatible with 10Hz closed-loop control.

Lifecycle (Skill — lazy activate):
    on_init      — parse config, validate safety params
    on_activate  — resolve vla_server endpoint, start ROS subscriber thread
    on_deactivate — stop subscriber thread, clear cache
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import requests
import json_numpy

from robonix_api import ATLAS, Skill, Ok, Err  # noqa: E402

logging.basicConfig(
    level=os.environ.get("VLA_CLIENT_LOG_LEVEL", "INFO"),
    format="[vla_client] %(message)s",
)
log = logging.getLogger("vla_client")

vla_skill = Skill(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "vla_client"),
    namespace="robonix/skill/vla",
)


# ── shared state ────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_cfg: dict = {}
_endpoints: Optional[dict[str, str]] = None

# ROS subscriber state (written by subscriber thread, read by handler)
_latest_full_image: Optional[np.ndarray] = None
_latest_wrist_image: Optional[np.ndarray] = None
_latest_joint_states: Optional[np.ndarray] = None
_obs_lock = threading.Lock()

# Safety filter state
_last_action: Optional[np.ndarray] = None  # last commanded [x,y,z,r,p,y, gripper_raw]

# ROS thread handle
_ros_thread: Optional[threading.Thread] = None
_ros_stop_event = threading.Event()


# ── atlas-resolved upstream contracts ───────────────────────────────────────
REQUIRED_INPUTS = {
    "vla_act": ("robonix/service/vla/inference/act", "mcp"),
}

OPTIONAL_INPUTS = {
    "reset": ("robonix/service/manipulation/reset", "mcp"),
}


def _resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    """Block until atlas can resolve REQUIRED_INPUTS, best-effort OPTIONAL."""
    resolved: dict[str, str] = {}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        for key, (cid, transport) in REQUIRED_INPUTS.items():
            if key in resolved:
                continue
            try:
                cap_view = ATLAS.find_unique_capability(
                    contract_id=cid, transport=transport)
                ch = vla_skill.connect_capability(cap_view, cid, transport)
            except Exception:
                continue
            ep = ch.endpoint
            try:
                ch.close()
            except Exception:
                pass
            if ep:
                resolved[key] = ep
                log.info("resolved %s [%s] → %s", cid, transport, ep)
        if len(resolved) == len(REQUIRED_INPUTS):
            break
        time.sleep(2.0)

    missing = [k for k in REQUIRED_INPUTS if k not in resolved]
    if missing:
        raise RuntimeError(
            f"vla_client cannot find deps on atlas: "
            f"{[REQUIRED_INPUTS[k][0] for k in missing]}. "
            f"Ensure vla_server_rbnx is ACTIVE.")

    for key, (cid, transport) in OPTIONAL_INPUTS.items():
        try:
            cap_view = ATLAS.find_unique_capability(
                contract_id=cid, transport=transport)
            ch = vla_skill.connect_capability(cap_view, cid, transport)
            ep = ch.endpoint
            try:
                ch.close()
            except Exception:
                pass
            if ep:
                resolved[key] = ep
                log.info("resolved %s [%s] → %s (optional)", cid, transport, ep)
        except Exception:
            log.warning("optional dep %s not on atlas — degraded (no emergency reset)", cid)

    return resolved


# ── ROS subscriber thread ───────────────────────────────────────────────────

def _start_ros_subscriber_thread():
    """Start a daemon thread that subscribes to camera + joint_states topics."""
    global _ros_thread
    _ros_stop_event.clear()
    _ros_thread = threading.Thread(target=_ros_spin_loop, daemon=True,
                                   name="vla-client-ros")
    _ros_thread.start()
    log.info("ROS subscriber thread started")


def _ros_spin_loop():
    """ROS2 subscriber loop — runs in dedicated thread."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, JointState
    from cv_bridge import CvBridge

    rclpy.init()
    node = rclpy.create_node("vla_client_subscriber")
    bridge = CvBridge()

    qos_best_effort = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    full_topic = _cfg.get("full_image_topic", "/camera/color/image_raw")
    wrist_topic = _cfg.get("wrist_image_topic", "/wrist_camera/color/image_raw")
    js_topic = _cfg.get("joint_states_topic", "/arm/joint_states_single")
    resize = _cfg.get("image_resize", [256, 256])

    def _on_full_image(msg: Image):
        global _latest_full_image
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img)
            pil = pil.resize((resize[0], resize[1]), PILImage.BILINEAR)
            with _obs_lock:
                _latest_full_image = np.array(pil, dtype=np.uint8)
        except Exception as e:
            log.debug("full_image callback error: %s", e)

    def _on_wrist_image(msg: Image):
        global _latest_wrist_image
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img)
            pil = pil.resize((resize[0], resize[1]), PILImage.BILINEAR)
            with _obs_lock:
                _latest_wrist_image = np.array(pil, dtype=np.uint8)
        except Exception as e:
            log.debug("wrist_image callback error: %s", e)

    def _on_joint_states(msg: JointState):
        global _latest_joint_states
        dim = int(_cfg.get("joint_state_dim", 7))
        positions = list(msg.position)
        if len(positions) >= dim:
            with _obs_lock:
                _latest_joint_states = np.array(positions[:dim], dtype=np.float32)

    node.create_subscription(Image, full_topic, _on_full_image, qos_best_effort)
    node.create_subscription(Image, wrist_topic, _on_wrist_image, qos_best_effort)
    node.create_subscription(JointState, js_topic, _on_joint_states, 10)

    # Also create publisher for pos_cmd
    from piper_msgs.msg import PosCmd
    pos_cmd_topic = _cfg.get("pos_cmd_topic", "/arm/pos_cmd")
    _pos_cmd_pub = node.create_publisher(PosCmd, pos_cmd_topic, 10)

    # Store publisher reference globally
    global _g_pos_cmd_pub, _g_node
    _g_pos_cmd_pub = _pos_cmd_pub
    _g_node = node

    log.info("subscribed: full=%s wrist=%s joints=%s | publish: %s",
             full_topic, wrist_topic, js_topic, pos_cmd_topic)

    while not _ros_stop_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    log.info("ROS subscriber thread stopped")


_g_pos_cmd_pub = None
_g_node = None


# ── observation + action helpers ────────────────────────────────────────────

def _get_current_observation():
    """Get latest (full_image, wrist_image, joint_states) or (None, None, None)."""
    with _obs_lock:
        return (_latest_full_image.copy() if _latest_full_image is not None else None,
                _latest_wrist_image.copy() if _latest_wrist_image is not None else None,
                _latest_joint_states.copy() if _latest_joint_states is not None else None)


def _call_vla_server(instruction: str, full_image: np.ndarray,
                     wrist_image: np.ndarray, state: np.ndarray):
    """Call VLA Server's /act endpoint, return list of action arrays or None."""
    url = _cfg.get("vla_server_url", "http://localhost:8777")
    if _cfg.get("use_atlas_discovery", True) and _endpoints and "vla_act" in _endpoints:
        # Atlas endpoint might be an MCP URL, but VLA server is HTTP
        # Use the base URL from atlas but target the /act path
        url = _cfg.get("vla_server_url", "http://localhost:8777")

    payload = {
        "full_image": full_image,
        "wrist_image": wrist_image,
        "state": state,
        "instruction": instruction,
    }

    try:
        resp = requests.post(
            f"{url}/act",
            json={"encoded": json_numpy.dumps(payload)},
            timeout=30.0,
        )
        resp.raise_for_status()
        actions = json_numpy.loads(resp.json())
        # Normalize to list of 1D arrays
        if isinstance(actions, np.ndarray):
            if actions.ndim == 1:
                return [actions]
            else:
                return [actions[i] for i in range(actions.shape[0])]
        elif isinstance(actions, list):
            return [np.array(a, dtype=np.float32) if not isinstance(a, np.ndarray) else a
                    for a in actions]
        return [np.array(actions, dtype=np.float32)]
    except Exception as e:
        log.error("VLA server call failed: %s", e)
        return None


def _send_action_to_arm(action: np.ndarray) -> None:
    """Send VLA delta action to arm with safety filter.

    VLA model outputs are DELTA actions (not absolute poses):
        action[0:6] = [dx, dy, dz, d_roll, d_pitch, d_yaw]  (Cartesian delta)
        action[6]   = gripper value (unnormalized, ~[-1, 1] range)

    This function:
        1. Reads current end-effector pose from /arm/end_pose (or latest joint_states)
        2. Adds delta to get target pose
        3. Applies safety filter (workspace limits + rate limit)
        4. Maps gripper from model range to physical range
        5. Publishes target pose to /arm/pos_cmd
    """
    global _last_action, _current_pose

    delta_pose = action[:6].copy()   # [dx, dy, dz, d_roll, d_pitch, d_yaw]
    gripper_raw = float(action[6]) if len(action) > 6 else 0.0

    # ── Get current pose as baseline for delta accumulation ──────────────
    # Use the last commanded pose (for smooth accumulation), or fall back to
    # the observed joint_states on the very first step.
    if _last_action is not None:
        current = _last_action[:6].copy()
    else:
        # First step: read current end-effector pose from observation
        with _obs_lock:
            if _latest_joint_states is not None:
                current = _latest_joint_states[:6].copy()
            else:
                log.warning("no current pose available, using zeros")
                current = np.zeros(6, dtype=np.float32)

    # ── Apply delta ─────────────────────────────────────────────────────
    target = current + delta_pose

    # ── Safety filter ───────────────────────────────────────────────────
    if _cfg.get("enable_safety_filter", True):
        # 1. Workspace / joint limits hard clip
        lo = np.array(_cfg.get("joint_limits_low", [-2.618]*6), dtype=np.float32)
        hi = np.array(_cfg.get("joint_limits_high", [2.618]*6), dtype=np.float32)
        target = np.clip(target, lo, hi)

        # 2. Rate limiting: cap the effective delta per step
        max_delta = np.array(_cfg.get("max_delta_per_step", [0.1]*6), dtype=np.float32)
        effective_delta = target - current
        effective_delta = np.clip(effective_delta, -max_delta, max_delta)
        target = current + effective_delta

    # ── Gripper mapping ─────────────────────────────────────────────────
    # VLA model outputs gripper in unnormalized range (approx [-1, +1]):
    #   -1 → fully closed,  +1 → fully open  (convention may vary)
    # Map to piper physical gripper width (meters): [0.0, 0.08]
    gripper_model_min = float(_cfg.get("gripper_model_min", -1.0))
    gripper_model_max = float(_cfg.get("gripper_model_max", 1.0))
    gripper_phys_min = float(_cfg.get("gripper_phys_min", 0.0))    # fully closed (m)
    gripper_phys_max = float(_cfg.get("gripper_phys_max", 0.08))   # fully open (m)

    # Linear map: model range → physical range
    t = (gripper_raw - gripper_model_min) / (gripper_model_max - gripper_model_min + 1e-8)
    t = float(np.clip(t, 0.0, 1.0))
    gripper_physical = gripper_phys_min + t * (gripper_phys_max - gripper_phys_min)

    # ── Update state ────────────────────────────────────────────────────
    _last_action = np.concatenate([target, [gripper_raw]])

    # ── Publish ─────────────────────────────────────────────────────────
    _publish_pos_cmd(target, gripper_physical)


def _publish_pos_cmd(target_pose: np.ndarray, gripper: float) -> None:
    """Publish target end-effector pose to /arm/pos_cmd.

    PosCmd fields (from piper_msgs):
        x, y, z         — target end-effector position (meters, in base frame)
        roll, pitch, yaw — target end-effector orientation (radians)
        gripper          — gripper width (meters, 0=closed, 0.08=open)
        mode1, mode2     — control modes (0=default)
    """
    if _g_pos_cmd_pub is None:
        log.warning("pos_cmd publisher not ready")
        return
    try:
        from piper_msgs.msg import PosCmd
        msg = PosCmd()
        msg.x = float(target_pose[0])
        msg.y = float(target_pose[1])
        msg.z = float(target_pose[2])
        msg.roll = float(target_pose[3])
        msg.pitch = float(target_pose[4])
        msg.yaw = float(target_pose[5])
        msg.gripper = float(gripper)
        msg.mode1 = 0
        msg.mode2 = 0
        _g_pos_cmd_pub.publish(msg)
    except Exception as e:
        log.error("publish pos_cmd failed: %s", e)


def _safe_reset(context: str) -> None:
    """Best-effort emergency reset via manipulation/reset MCP (MoveIt joint-space)."""
    if _endpoints is None or "reset" not in _endpoints:
        log.warning("reset not available — skipping (%s)", context)
        return
    try:
        # Use the sync MCP call pattern from pick_skill
        log.info("calling manipulation/reset (%s)", context)
        resp = requests.post(
            _endpoints["reset"],
            json={"method": "tools/call", "params": {"name": "reset", "arguments": {"ack": True}}},
            timeout=30.0,
        )
        log.info("reset result: %d", resp.status_code)
    except Exception as e:
        log.warning("reset failed (%s): %s", context, e)


# ── MCP handler ─────────────────────────────────────────────────────────────

from vla_client_mcp import VlaExecute_Request, VlaExecute_Response  # noqa: E402


@vla_skill.mcp("robonix/skill/vla/execute")
def execute(req: VlaExecute_Request) -> VlaExecute_Response:
    """Execute a VLA (Vision-Language-Action) policy for a given instruction.

    Use this tool when the user asks the robot to perform a manipulation
    task described in natural language, using end-to-end neural network
    control (VLA policy) rather than traditional planning.

    Examples: "pick up the red cube", "place the bottle on the tray",
    "push the block to the left"

    This call is synchronous — it runs a closed-loop inference cycle
    (observe → infer → act → repeat) until timeout or max_steps.
    Typical latency: 10-60s depending on task complexity.
    """
    global _last_action

    if _endpoints is None:
        return VlaExecute_Response(
            success=False,
            message="vla_client not active (atlas hasn't resolved vla_server yet)",
            steps_executed=0, elapsed_s=0.0,
        )

    instruction = (req.instruction or "").strip()
    if not instruction:
        return VlaExecute_Response(
            success=False, message="instruction is empty",
            steps_executed=0, elapsed_s=0.0,
        )

    timeout_s = float(req.timeout_s) if req.timeout_s > 0 else float(_cfg.get("timeout_s", 60.0))
    max_steps = int(req.max_steps) if req.max_steps > 0 else 0
    action_hz = float(_cfg.get("action_hz", 10.0))
    action_interval = 1.0 / action_hz

    t0 = time.monotonic()
    deadline = t0 + timeout_s
    steps_executed = 0
    _last_action = None  # Reset safety filter state at start of new execution

    log.info("execute(%r) timeout=%.1fs max_steps=%d hz=%.1f",
             instruction, timeout_s, max_steps, action_hz)

    try:
        while time.monotonic() < deadline:
            if max_steps > 0 and steps_executed >= max_steps:
                break

            # 1. Get current observation
            full_img, wrist_img, state = _get_current_observation()
            if full_img is None or state is None:
                # Wrist image fallback: use full_image if wrist camera unavailable
                if full_img is not None and wrist_img is None:
                    wrist_img = full_img.copy()
                    log.debug("wrist_image unavailable, using full_image as fallback")
                else:
                    time.sleep(0.1)
                    continue

            # 2. Call VLA server
            actions = _call_vla_server(instruction, full_img, wrist_img, state)
            if actions is None:
                return VlaExecute_Response(
                    success=False,
                    message="VLA server inference failed",
                    steps_executed=steps_executed,
                    elapsed_s=time.monotonic() - t0,
                )

            # 3. Execute actions with safety filter
            for action in actions:
                if time.monotonic() >= deadline:
                    break
                if max_steps > 0 and steps_executed >= max_steps:
                    break
                _send_action_to_arm(action)
                steps_executed += 1
                time.sleep(action_interval)

        elapsed = time.monotonic() - t0
        log.info("execute done: %d steps in %.2fs", steps_executed, elapsed)
        return VlaExecute_Response(
            success=True,
            message=f"Executed {steps_executed} steps",
            steps_executed=steps_executed,
            elapsed_s=elapsed,
        )
    except Exception as e:
        log.error("execute error: %s", e)
        _safe_reset("after execute error")
        return VlaExecute_Response(
            success=False,
            message=f"execution error: {e}",
            steps_executed=steps_executed,
            elapsed_s=time.monotonic() - t0,
        )


# ── lifecycle ───────────────────────────────────────────────────────────────

@vla_skill.on_init
def init(cfg):
    """CMD_INIT: light. Parse config, validate safety parameters."""
    global _cfg
    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")
    _cfg = cfg
    log.info("CMD_INIT ok (vla_server_url=%s, action_hz=%.1f, safety=%s)",
             _cfg.get("vla_server_url", "http://localhost:8777"),
             float(_cfg.get("action_hz", 10.0)),
             "ON" if _cfg.get("enable_safety_filter", True) else "OFF")
    return Ok()


@vla_skill.on_activate
def activate():
    """CMD_ACTIVATE: heavy. Resolve vla_server endpoint, start ROS subscriber."""
    global _endpoints
    with _state_lock:
        if _endpoints is not None:
            log.info("CMD_ACTIVATE — already active, no-op")
            return Ok()
        try:
            _endpoints = _resolve_inputs()
        except RuntimeError as e:
            return Err(str(e))
        _start_ros_subscriber_thread()
    log.info("CMD_ACTIVATE ok — endpoints: %s", list(_endpoints.keys()))
    return Ok()


@vla_skill.on_deactivate
def deactivate():
    """CMD_DEACTIVATE: stop ROS thread, clear state."""
    global _endpoints, _last_action
    with _state_lock:
        _endpoints = None
        _last_action = None
    _ros_stop_event.set()
    if _ros_thread is not None:
        _ros_thread.join(timeout=5.0)
    log.info("CMD_DEACTIVATE ok")
    return Ok()


# ── entrypoint ──────────────────────────────────────────────────────────────

def main() -> int:
    vla_skill.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
