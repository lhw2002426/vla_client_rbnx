# SPDX-License-Identifier: MulanPSL-2.0
"""vla_client_rbnx atlas bridge — VLA inference + JOINT-SPACE execution skill.

LLM-callable skill that:
1. Subscribes to camera images + joint states (ROS topics)
2. Calls vla_server_rbnx for action inference (HTTP via atlas-resolved endpoint)
3. **De-normalizes [-1, 1] actions back to absolute joint targets**
4. Sends joint commands DIRECTLY to /arm/joint_states (piper_ctl_rbnx joint cb)
5. Includes an internal safety filter (hardware joint-limit clip + rate limit)

Architecture:
    VLA Server ─[-1,1] actions─► vla_client de-norm + safety ─►
        sensor_msgs/JointState on /arm/joint_states ─► piper_ctl ─► CAN ─► arm

Action space (matches piper_grasp_2cam dataset builder):
    action[0:6] — joint1..joint6 absolute angles
    action[6]   — gripper width
    All seven dims live in normalized [-1, 1] space and are de-normalized
    using ACTION_MIN / ACTION_MAX (the dataset's per-channel min/max).

Unit conventions (CRITICAL):
    Dataset / VLA training space — uses piper SDK raw units:
        joints  : 0.001° (so divide by 1000 → degrees, then * π/180 → rad)
                  Equivalent factor used by piper_ctl: 1° = 1000 SDK units,
                  rad = SDK / 57324.840764  (= 1000 * 180/π).
        gripper : "0.001 mm" units * 2 (the SDK uses µm, then piper_ctl
                  multiplies by `gripper_val_mutiple=2` again before
                  publishing to CAN). On the JointState write side we go:
                  meters → SDK_mm * 2; on the read side piper_ctl sees
                  position[6] in METERS and multiplies by 1e6 then by 2.
                  So: SDK_units = meters * 2_000_000  ⇔  meters = SDK / 2e6.
                  Dataset gripper range [-3000, 88500] (SDK units) maps to
                  meters [-0.0015, 0.04425].

Why JOINT space and not Cartesian:
    The training dataset records absolute joint angles + gripper, NOT
    Cartesian end-effector poses. Sending those values to /arm/pos_cmd
    (which expects [x,y,z,roll,pitch,yaw,gripper] in meters/rad) would be
    a category error and drive the arm into limits / singularities.

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
from typing import Optional

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
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "openvla_client"),
    namespace="robonix/skill/vla",
)


# ── unit-conversion constants ───────────────────────────────────────────────
# Matches piper_ctrl_single_node.joint_callback:
#   factor = 57324.840764   # 1000 * 180/π  (rad → SDK 0.001°)
#   joint7 (gripper): SDK_units = position_in_m * 1e6 * gripper_val_mutiple(=2)
SDK_PER_RAD = 57324.840764
SDK_PER_M_GRIPPER = 2.0e6   # piper_ctl multiplies meters by 1e6 then by 2

# ── shared state ────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_cfg: dict = {}
_endpoints: Optional[dict[str, str]] = None

# ROS subscriber state (written by subscriber thread, read by handler)
_latest_full_image: Optional[np.ndarray] = None
_latest_wrist_image: Optional[np.ndarray] = None
_latest_joint_states: Optional[np.ndarray] = None       # joint angles (proprio for VLA)
_obs_lock = threading.Lock()

# Safety filter state — last commanded joint vector in DATASET (SDK) units
_last_action_sdk: Optional[np.ndarray] = None

# ROS thread handle
_ros_thread: Optional[threading.Thread] = None
_ros_stop_event = threading.Event()


# ── atlas-resolved upstream contracts ───────────────────────────────────────
# VLA server is connected directly via HTTP (config.vla_server_url),
# so NO required atlas inputs. Only optional ones for post-grasp reset etc.
REQUIRED_INPUTS = {}

OPTIONAL_INPUTS = {
    "reset": ("robonix/service/manipulation/reset", "mcp"),
}


def _resolve_inputs(deadline_s: float = 60.0) -> dict[str, str]:
    """Best-effort resolve OPTIONAL_INPUTS on atlas. VLA server is
    accessed directly via HTTP URL from config, not through atlas."""
    resolved: dict[str, str] = {}

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
    """ROS2 subscriber loop — runs in dedicated thread.

    Subscribes:
        full_image, wrist_image  — camera RGB streams (resized to model input)
        joint_states_single      — current joint angles (proprio for VLA)

    Publishes:
        sensor_msgs/JointState on /arm/joint_states  — joint command for piper_ctl
    """
    log.info("[ROS-THREAD] importing rclpy...")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    log.info("[ROS-THREAD] importing sensor_msgs...")
    from sensor_msgs.msg import Image, JointState
    log.info("[ROS-THREAD] importing cv_bridge...")
    from cv_bridge import CvBridge

    log.info("[ROS-THREAD] calling rclpy.init()...")
    rclpy.init()
    log.info("[ROS-THREAD] creating node...")
    node = rclpy.create_node("vla_client_subscriber")
    log.info("[ROS-THREAD] creating CvBridge...")
    bridge = CvBridge()
    log.info("[ROS-THREAD] init done, setting up subscriptions...")

    qos_best_effort = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    full_topic = _cfg.get("full_image_topic", "/camera/color/image_raw")
    wrist_topic = _cfg.get("wrist_image_topic", "/wrist_camera/color/image_raw")
    js_topic = _cfg.get("joint_states_topic", "/arm/joint_states_single")
    joint_cmd_topic = _cfg.get("joint_cmd_topic", "/arm/joint_states")
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
        """Latch current joint angles as proprioceptive input for VLA.

        IMPORTANT: piper_ctl publishes joints in **radians** (and gripper in
        **meters**). The VLA model was trained on **dataset SDK units**
        (0.001° for joints, "2*µm" for gripper). We convert here so the
        proprio handed to the server matches training distribution exactly.
        """
        global _latest_joint_states
        dim = int(_cfg.get("joint_state_dim", 7))
        positions = list(msg.position)
        if len(positions) < dim:
            return
        # Convert ROS units → dataset SDK units (matches builder's raw range)
        sdk = np.zeros(dim, dtype=np.float32)
        n_joints = min(6, dim)
        for i in range(n_joints):
            sdk[i] = float(positions[i]) * SDK_PER_RAD          # rad → 0.001°
        if dim >= 7 and len(positions) >= 7:
            sdk[6] = float(positions[6]) * SDK_PER_M_GRIPPER    # m   → SDK gripper
        with _obs_lock:
            _latest_joint_states = sdk

    node.create_subscription(Image, full_topic, _on_full_image, qos_best_effort)
    node.create_subscription(Image, wrist_topic, _on_wrist_image, qos_best_effort)
    node.create_subscription(JointState, js_topic, _on_joint_states, 10)

    # Publisher: joint-space command to piper_ctl
    _joint_cmd_pub = node.create_publisher(JointState, joint_cmd_topic, 10)

    global _g_joint_cmd_pub, _g_node
    _g_joint_cmd_pub = _joint_cmd_pub
    _g_node = node

    log.info("subscribed: full=%s wrist=%s joints=%s | publish: %s",
             full_topic, wrist_topic, js_topic, joint_cmd_topic)
    log.info("[ROS-THREAD] entering spin loop...")

    while not _ros_stop_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    log.info("ROS subscriber thread stopped")


_g_joint_cmd_pub = None
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


# ── action de-normalization + safety ────────────────────────────────────────

def _denormalize_action(action_norm: np.ndarray) -> np.ndarray:
    """Map normalized [-1, 1] action → dataset SDK-unit absolute joint vector.

    Inverse of dataset builder's:
        norm = 2 * (raw - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) - 1
    so:
        raw  = 0.5 * (norm + 1) * (ACTION_MAX - ACTION_MIN) + ACTION_MIN

    Returns 7-vector in **dataset SDK units**:
        [j1, j2, j3, j4, j5, j6]  in 0.001°
        [j7]                       in "2 * µm" (piper SDK gripper)
    """
    a_min = np.array(_cfg["action_min"], dtype=np.float64)
    a_max = np.array(_cfg["action_max"], dtype=np.float64)
    a = np.asarray(action_norm, dtype=np.float64).flatten()
    if a.shape[0] != a_min.shape[0]:
        # Truncate / pad defensively
        n = min(a.shape[0], a_min.shape[0])
        out = np.zeros(a_min.shape[0], dtype=np.float64)
        out[:n] = a[:n]
        a = out
    raw = 0.5 * (a + 1.0) * (a_max - a_min) + a_min
    return raw.astype(np.float32)


def _apply_safety(target_sdk: np.ndarray) -> np.ndarray:
    """Clip joint command to hardware limits and bound the per-step delta.

    Operates entirely in SDK units (0.001° for joints, SDK gripper for j7).
    """
    global _last_action_sdk
    target = target_sdk.astype(np.float32).copy()

    if not _cfg.get("enable_safety_filter", True):
        return target

    # 1) Hardware joint-limit clip
    hw_min = np.array(_cfg["hw_action_min"], dtype=np.float32)
    hw_max = np.array(_cfg["hw_action_max"], dtype=np.float32)
    target = np.clip(target, hw_min, hw_max)

    # 2) Rate limiting (max change per step, in SDK units)
    if _last_action_sdk is not None:
        max_delta = np.array(_cfg["max_delta_per_step_sdk"], dtype=np.float32)
        delta = target - _last_action_sdk
        delta = np.clip(delta, -max_delta, max_delta)
        target = _last_action_sdk + delta

    return target


def _send_action_to_arm(action_norm: np.ndarray) -> None:
    """De-normalize a [-1,1] VLA action and publish to /arm/joint_states."""
    global _last_action_sdk

    # 1) De-normalize → SDK units
    target_sdk = _denormalize_action(action_norm)

    # 2) Safety filter (clip + rate limit)
    target_sdk = _apply_safety(target_sdk)

    # 3) Convert SDK units → ROS units expected by piper_ctl.joint_callback
    #    joints[0..5] : 0.001°  → rad
    #    joint[6]     : "2*µm" → meters    (piper_ctl will * 1e6 * 2 again)
    target_ros = np.zeros(7, dtype=np.float64)
    target_ros[:6] = target_sdk[:6] / SDK_PER_RAD
    target_ros[6]  = float(target_sdk[6]) / SDK_PER_M_GRIPPER

    # 4) Publish JointState
    _publish_joint_state(target_ros)

    # 5) Latch state for next-step rate limiting
    _last_action_sdk = target_sdk


def _publish_joint_state(positions_ros: np.ndarray) -> None:
    """Publish sensor_msgs/JointState to /arm/joint_states.

    Field layout matches what piper_ctrl_single_node.joint_callback expects:
        name     = ['joint1'..'joint6', 'gripper']
        position = [j1..j6 (rad), gripper (m)]
        velocity = [0]*6 + [vel_pct]   # SDK velocity %, 1-100; 0 = use default 30
        effort   = [0]*6 + [grip_eff]  # gripper effort 0.5-3
    """
    if _g_joint_cmd_pub is None:
        log.warning("joint_states publisher not ready")
        return
    try:
        from sensor_msgs.msg import JointState
        msg = JointState()
        if _g_node is not None:
            msg.header.stamp = _g_node.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        msg.position = [float(x) for x in positions_ros]

        vel_pct = float(_cfg.get("joint_velocity_pct", 30.0))
        # piper_ctl reads velocity[6] as "all-axis %"; non-zero triggers MotionCtrl_2
        msg.velocity = [0.0] * 6 + [vel_pct]

        grip_eff = float(_cfg.get("gripper_effort", 1.0))
        msg.effort = [0.0] * 6 + [grip_eff]

        _g_joint_cmd_pub.publish(msg)
    except Exception as e:
        log.error("publish joint_states failed: %s", e)


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
    global _last_action_sdk

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
    _last_action_sdk = None  # Reset safety filter state at start of new execution

    log.info("execute(%r) timeout=%.1fs max_steps=%d hz=%.1f",
             instruction, timeout_s, max_steps, action_hz)

    try:
        while time.monotonic() < deadline:
            if max_steps > 0 and steps_executed >= max_steps:
                break

            # 1. Get current observation
            full_img, wrist_img, state = _get_current_observation()
            if full_img is None or state is None:
                # Required obs not yet available — wait and retry.
                time.sleep(0.1)
                continue
            if wrist_img is None:
                # Optional: fall back to full_image if wrist camera not running.
                wrist_img = full_img.copy()
                log.debug("wrist_image unavailable — using full_image as fallback")

            # 2. Call VLA server (state is in SDK units, matching training)
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

    # Sanity-check that de-normalization constants are present and well-shaped.
    for key in ("action_min", "action_max", "hw_action_min",
                "hw_action_max", "max_delta_per_step_sdk"):
        v = _cfg.get(key)
        if v is None or len(v) != 7:
            return Err(f"config.{key} must be a length-7 list (got {v!r})")

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
    global _endpoints, _last_action_sdk
    with _state_lock:
        _endpoints = None
        _last_action_sdk = None
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
