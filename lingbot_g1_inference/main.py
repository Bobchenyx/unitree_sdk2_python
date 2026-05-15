"""Main inference loop: cloud LingBot-VA policy → G1 arms + grippers.

Precondition: the robot must already be in 'ai' motion mode, set via the
Unitree app before starting this script — this code does not switch modes.

Pipeline:
  1. DDS setup, start arm + gripper controller threads, then camera client.
  2. Move arms to INIT_POSE_READY; close then open grippers.
  3. STANDBY: hold the ready pose until the operator presses Enter.
  4. Switch arm kp from the stiff init value to the softer inference value.
  5. Connect to the cloud WebSocket server, send reset({prompt}).
  6. Loop per chunk:
       a. infer({obs, prompt}) → action [16, 2, 16].
       b. Iterate frames × sub_steps, dispatch targets at 30 Hz.
       c. Every CAPTURE_EVERY sub-steps snap a keyframe — 8 per chunk,
          except the first chunk where frame 0 is skipped → 4 keyframes.
       d. compute_kv_cache: hand the keyframes + executed action back so
          the server's KV cache reflects reality.
       e. Request the next chunk and repeat.

Usage:
    python -m lingbot_g1_inference.main \\
        --iface enp0s31f6 \\
        --server-host 1.2.3.4 \\
        --server-port 29056 \\
        --prompt "pick up the pink object and place it on the blue cross mark"
"""

import argparse
import logging
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from .arm_controller import ArmController, INIT_POSE_READY
from .gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from .camera_client import CameraClient
from .policy_client import PolicyClient


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("lingbot_g1.main")


# Cadence:
#   frame_chunk_size = 2 latent frames per chunk
#   action_per_frame = 16 sub-steps per latent frame  → 32 sub-steps per chunk
#   sub-step rate    = ~30 Hz (33.3 ms each)
#   capture cadence  = every 4 sub-steps → 8 keyframes per chunk
CAPTURE_EVERY = 4
FRAME_CHUNK = 2
SUBSTEPS_PER_FRAME = 16

# Action tensor layout — channels along axis 0 of action[16, FRAME_CHUNK, SUBSTEPS_PER_FRAME]:
#   [0:7]   = left arm  (order matches arm_controller.ARM_JOINTS[0:7]:
#                        L pitch/roll/yaw/elbow/wristR/wristP/wristY)
#   [7:14]  = right arm (mirrored, matches ARM_JOINTS[7:14])
#   [14]    = left gripper  (rad in [GRIPPER_MIN, GRIPPER_MAX])
#   [15]    = right gripper
ARM_CHANNELS = slice(0, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15


def log_chunk_summary(chunk_id, action):
    """One-line summary of the per-joint range for a chunk so we can see what
    the model is asking for before it streams to the robot."""
    arm = action[ARM_CHANNELS]
    grip_l = action[LEFT_GRIPPER_CHANNEL]
    grip_r = action[RIGHT_GRIPPER_CHANNEL]
    arm_min = arm.min(axis=(1, 2))
    arm_max = arm.max(axis=(1, 2))
    log.info(f"[chunk {chunk_id}] arm joint ranges (rad):")
    joint_names = [
        "L_pitch", "L_roll ", "L_yaw  ", "L_elbow", "L_wrR  ", "L_wrP  ", "L_wrY  ",
        "R_pitch", "R_roll ", "R_yaw  ", "R_elbow", "R_wrR  ", "R_wrP  ", "R_wrY  ",
    ]
    for i, name in enumerate(joint_names):
        log.info(f"   {name}  min={arm_min[i]:+.3f}  max={arm_max[i]:+.3f}")
    log.info(f"[chunk {chunk_id}] gripper L: [{grip_l.min():.2f}, {grip_l.max():.2f}] "
             f"R: [{grip_r.min():.2f}, {grip_r.max():.2f}]")


def execute_chunk(action, arm_ctrl, grip_ctrl, cam_client, is_first_chunk,
                  substep_dt=1.0 / 30.0, verbose_substeps=True):
    """Dispatch one [16, 2, 16] action chunk to arms+grippers at 30 Hz.

    Returns the captured keyframes — 8 per chunk, except the first chunk
    where frame 0 is skipped → 4. Each keyframe is the camera-only image
    dict (no prompt key), ready to ship back as part of compute_kv_cache.

    For the first chunk only, skip frame 0's sub-steps (the model treats
    them as 'observe the start position').
    """
    assert action.shape == (16, FRAME_CHUNK, SUBSTEPS_PER_FRAME), \
        f"Unexpected action shape {action.shape}"

    keyframes = []
    start_frame = 1 if is_first_chunk else 0

    for t in range(start_frame, FRAME_CHUNK):
        for f in range(SUBSTEPS_PER_FRAME):
            tic = time.time()

            cmd = action[:, t, f]
            arm_q = cmd[ARM_CHANNELS].astype(np.float64)
            l_grip = float(np.clip(cmd[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX))
            r_grip = float(np.clip(cmd[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX))

            if verbose_substeps:
                # Pretty-print the target the controller will see this tick.
                # Format: L_arm(7 floats) | R_arm(7 floats) | L_grip R_grip
                l_arm_str = " ".join(f"{x:+.3f}" for x in arm_q[:7])
                r_arm_str = " ".join(f"{x:+.3f}" for x in arm_q[7:])
                log.info(f"  t={t} f={f:2d} | L=[{l_arm_str}] R=[{r_arm_str}] "
                         f"| g=({l_grip:.2f},{r_grip:.2f})")

            arm_ctrl.set_arm_target(arm_q)
            grip_ctrl.set_targets(l_grip, r_grip)

            # Snap a keyframe at the end of each CAPTURE_EVERY-step window
            # (f=3,7,11,15 — late in the window so motion has had time to settle).
            if (f + 1) % CAPTURE_EVERY == 0:
                try:
                    keyframes.append(cam_client.get_obs_images())
                except Exception as e:
                    log.warning(f"Camera capture failed at frame={t} sub={f}: {e}")

            elapsed = time.time() - tic
            sleep = substep_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    return keyframes


# ---------- pipeline stages ----------

def _setup_arm(args) -> ArmController:
    log.info("Starting ArmController")
    arm = ArmController(publish_hz=50.0, velocity_limit=args.velocity_limit)
    arm.start()
    return arm


def _setup_gripper() -> GripperController:
    log.info("Starting GripperController")
    grip = GripperController(publish_hz=200.0)
    grip.start()
    return grip


def _setup_camera(args) -> CameraClient:
    log.info("Starting CameraClient")
    return CameraClient(host=args.image_server)


def _initialize_pose(arm, grip, args) -> None:
    """Slow ramp to INIT_POSE_READY, then close→open grippers, then settle."""
    log.info(f"Moving arms to ready pose over {args.init_duration:.1f}s "
             f"(velocity_limit={args.velocity_limit} rad/s)")
    arm.move_to_pose(INIT_POSE_READY,
                     duration=args.init_duration,
                     velocity_limit=args.velocity_limit)
    half = args.init_duration / 2
    log.info(f"Closing grippers to {GRIPPER_MIN} over {half:.1f}s")
    grip.move_to_targets(GRIPPER_MIN, GRIPPER_MIN, duration=half)
    log.info(f"Opening grippers to ({args.init_gripper_left}, {args.init_gripper_right}) "
             f"over {half:.1f}s")
    grip.move_to_targets(args.init_gripper_left,
                         args.init_gripper_right,
                         duration=half)
    log.info("Init complete.")
    time.sleep(args.settle_duration)
    log.info("Arms settled at ready pose.")


def _wait_for_operator(args) -> None:
    """Block on Enter so the operator can stage the scene. No-op with --auto-start.

    Arm + gripper publish threads keep streaming the current target during the
    wait, so the robot stays locked at INIT_POSE_READY.
    """
    if args.auto_start:
        return
    log.info("===============================================================")
    log.info("STANDBY: arms locked at ready pose.")
    log.info("Set up the scene (place pink object, blue cross mark, etc.)")
    log.info("Press [Enter] to connect to the policy server and start.")
    log.info("Press [Ctrl+C] at any time to abort safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — proceeding without prompt")


def _connect_policy(args) -> PolicyClient:
    log.info(f"Connecting to policy server ws://{args.server_host}:{args.server_port}")
    policy = PolicyClient(host=args.server_host, port=args.server_port)
    log.info(f"Server metadata: {policy.get_server_metadata()}")
    return policy


def _request_action(policy, cam, args) -> np.ndarray:
    """Send a fresh obs + prompt to the policy and return the action tensor."""
    result = policy.infer({
        "obs": cam.get_obs(args.prompt),
        "prompt": args.prompt,
        "video_guidance_scale": args.video_guidance,
        "action_guidance_scale": args.action_guidance,
    })
    return result["action"]


def _run_inference_loop(arm, grip, cam, policy, args) -> None:
    log.info(f"Reset with prompt: {args.prompt!r}")
    policy.reset(args.prompt)

    chunk_id = 0
    log.info(f"[chunk {chunk_id}] sending initial obs")
    action = _request_action(policy, cam, args)
    log.info(f"[chunk {chunk_id}] got action {action.shape}")
    log_chunk_summary(chunk_id, action)

    while chunk_id < args.max_chunks:
        keyframes = execute_chunk(action, arm, grip, cam,
                                  is_first_chunk=(chunk_id == 0),
                                  substep_dt=1.0 / args.substep_hz,
                                  verbose_substeps=not args.quiet_substeps)

        # Hand the just-executed action + keyframes back so the server's
        # KV cache reflects reality before the next infer().
        policy.infer({
            "obs": keyframes,
            "compute_kv_cache": True,
            "state": action,
        })

        chunk_id += 1
        if chunk_id >= args.max_chunks:
            break

        # Request next chunk. Note: after the first chunk, the model uses
        # its KV cache as context — the obs field isn't strictly needed for
        # autoregression but we send a fresh capture in case the server
        # uses it for sanity logging.
        log.info(f"[chunk {chunk_id}] requesting next action")
        action = _request_action(policy, cam, args)
        log.info(f"[chunk {chunk_id}] got action {action.shape}")
        log_chunk_summary(chunk_id, action)


def _cleanup(arm, grip, cam, policy) -> None:
    log.info("Shutting down — releasing arm_sdk")
    # Stop publish loop FIRST so disable_arm_sdk has exclusive access to
    # self.cmd / self.pub — otherwise the two threads race on CRC.
    arm.stop()
    try:
        arm.disable_arm_sdk()
    except Exception as e:
        log.warning(f"disable_arm_sdk failed: {e}")
    if grip is not None:
        grip.stop()
    if cam is not None:
        cam.close()
    if policy is not None:
        policy.close()


def run(args):
    log.info(f"Initializing DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)

    # arm.start() is outside the try: if it fails, arm_sdk was never enabled
    # so no cleanup is needed. Everything after this point is wrapped.
    arm = _setup_arm(args)

    grip = None
    cam = None
    policy = None
    try:
        grip = _setup_gripper()
        cam = _setup_camera(args)
        _initialize_pose(arm, grip, args)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        policy = _connect_policy(args)
        _run_inference_loop(arm, grip, cam, policy, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--server-host", required=True, help="Cloud GPU server hostname or IP")
    p.add_argument("--server-port", type=int, default=29056, help="Cloud server WebSocket port")
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt", default="pick up the pink object and place it on the blue cross mark")
    p.add_argument("--max-chunks", type=int, default=20,
                   help="How many action chunks to execute before stopping")
    p.add_argument("--video-guidance", type=float, default=5.0)
    p.add_argument("--action-guidance", type=float, default=1.0)

    # ---- Safety / motion limits ----
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="rad/s velocity cap on the per-tick motion clamp, applied "
                        "for both the init move and model-driven motion (default 8.0)")
    p.add_argument("--inference-kp-arm", type=float, default=80.0,
                   help="kp applied to shoulder/elbow joints once inference starts "
                        "(default 80; init/standby uses the controller's default kp_arm=150)")
    p.add_argument("--init-duration", type=float, default=5.0,
                   help="Seconds to take moving from current pose to ready pose")
    p.add_argument("--settle-duration", type=float, default=2.0,
                   help="Seconds to wait after init move so the arm settles at "
                        "the target pose before entering standby (default 2.0)")
    p.add_argument("--init-gripper-left", type=float, default=5.0,
                   help="Open position after the close→open init sequence (0..5.4, default 5.0)")
    p.add_argument("--init-gripper-right", type=float, default=5.0,
                   help="Open position after the close→open init sequence (0..5.4, default 5.0)")
    p.add_argument("--substep-hz", type=float, default=30.0,
                   help="Action sub-step dispatch rate (default 30 Hz). Lower this "
                        "to slow down the whole chunk if motion still looks too fast.")
    p.add_argument("--quiet-substeps", action="store_true",
                   help="Suppress per-substep target printing (only chunk summaries).")
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start inference immediately.")
    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()
