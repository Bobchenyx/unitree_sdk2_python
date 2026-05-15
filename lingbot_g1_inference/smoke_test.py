"""Minimal cloud round-trip test — no robot, no cameras.

Sends a synthetic 3-camera observation to the cloud LingBot-VA server and prints
the action chunk shape. Use this to validate connectivity, server health, and
the policy-client contract before involving the physical robot.

Usage:
    python -m lingbot_g1_inference.smoke_test \\
        --server-host 1.2.3.4 --server-port 29056
"""

import argparse
import logging

import numpy as np

from .policy_client import PolicyClient


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("lingbot_g1.smoke")


def fake_obs(prompt: str) -> dict:
    img = lambda: (np.random.rand(256, 320, 3) * 255).astype(np.uint8)
    return {
        "observation.images.cam_left_high":   img(),
        "observation.images.cam_left_wrist":  img(),
        "observation.images.cam_right_wrist": img(),
        "task": prompt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=29056)
    p.add_argument("--prompt", default="pick up the pink object and place it on the blue cross mark")
    args = p.parse_args()

    policy = PolicyClient(host=args.server_host, port=args.server_port)
    log.info(f"Connected. Server metadata: {policy.get_server_metadata()}")

    log.info(f"Sending reset({args.prompt!r})")
    policy.reset(args.prompt)

    log.info("Sending first infer() with synthetic obs")
    result = policy.infer({
        "obs": fake_obs(args.prompt),
        "prompt": args.prompt,
        "video_guidance_scale": 5.0,
        "action_guidance_scale": 1.0,
    })

    if "action" not in result:
        log.error(f"Server didn't return an action: keys={list(result.keys())}")
        return 1
    action = result["action"]
    log.info(f"action shape: {action.shape}, dtype: {action.dtype}")
    log.info(f"action min/max: {action.min():.3f} / {action.max():.3f}")

    # Slice into the channels we care about
    arm = action[:14]
    grip = action[14:16]
    log.info(f"arm joints range: [{arm.min():.3f}, {arm.max():.3f}] rad")
    log.info(f"gripper range:    [{grip.min():.3f}, {grip.max():.3f}] (expect 0..5.4)")

    # First step the robot would execute (frame=1, sub=0 — skip frame=0 on first chunk)
    first_step = action[:, 1, 0]
    log.info(f"first step to execute: {first_step}")

    policy.close()
    log.info("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
