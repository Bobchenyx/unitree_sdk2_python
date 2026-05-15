# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

`lingbot_g1_inference/` is a runtime inference client that drives a Unitree G1 humanoid's arms and grippers from action chunks produced by a remote LingBot-VA policy server. It is an untracked subdirectory layered on top of the upstream `unitree_sdk2_python` repo and is **not** part of the published `unitree_sdk2py` package.

The parent repo is the Unitree Python SDK (`unitree_sdk2py`, CycloneDDS-backed). This package depends on it but is otherwise self-contained.

## Commands

Install deps (from repo root, **not** from here):

```bash
cd ..                      # /home/yixiao/Workspace/dev/unitree_sdk2_python
pip3 install -e .          # installs unitree_sdk2py
# If cyclonedds isn't found, build it (see ../README.md) and:
# export CYCLONEDDS_HOME=~/cyclonedds/install
```

Extra runtime deps that are NOT in `setup.py`: `websockets`, `msgpack`, `teleimager` (the G1 image-server client). Install ad-hoc as needed.

Run the full pipeline (must be run as a module so relative imports resolve):

```bash
python -m lingbot_g1_inference.main \
    --iface enp0s31f6 \
    --server-host <cloud-ip> \
    --server-port 29056 \
    --prompt "pick up the pink object and place it on the blue cross mark"
```

CWD must be the repo root (`unitree_sdk2_python/`), not this directory.

Validate cloud connectivity without touching the robot:

```bash
python -m lingbot_g1_inference.smoke_test --server-host <cloud-ip> --server-port 29056
```

There are no tests, linters, or build configs in this package.

## Architecture

The inference pipeline is three asynchronous loops glued together in `main.py`:

```
Cloud LingBot-VA server  ──(WebSocket + msgpack_numpy)──►  PolicyClient
                                                              │
                                                              ▼ action [16,2,16]
                                                          execute_chunk()
                                                          ├─► ArmController  ──(DDS rt/arm_sdk @ 50 Hz)──► G1 arms+body lock
                                                          ├─► GripperController ──(DDS rt/dex1/{l,r}/cmd @ 200 Hz)──► Dex1 grippers
                                                          └─► CameraClient.get_obs() ◄──(ZMQ teleimager)── G1 PC2 image server
```

**Action tensor layout.** Each chunk is shape `(16, FRAME_CHUNK=2, SUBSTEPS_PER_FRAME=16)`. The named constants live at the top of `main.py`:
- `ARM_CHANNELS = slice(0, 14)` — 14 arm joints, ordered L{pitch,roll,yaw,elbow,wristR,wristP,wristY} then R{same}. This order is hard-coded in `arm_controller.ARM_JOINTS` and the model is trained to match it — do not reorder.
- `LEFT_GRIPPER_CHANNEL = 14`, `RIGHT_GRIPPER_CHANNEL = 15` — gripper q in `[GRIPPER_MIN=0, GRIPPER_MAX=5.4]` rad.

**Chunk execution cadence.** 30 Hz sub-step dispatch → one chunk = 2 × 16 sub-steps ≈ 1.07 s. Keyframes are captured every `CAPTURE_EVERY=4` sub-steps and shipped back to the server in a `compute_kv_cache` call so the server's autoregressive context tracks reality. **8 keyframes per chunk** for chunks 1+, but **only 4 for chunk 0** — the first chunk's frame 0 is skipped (the model treats it as "observe the start position"), so only frame 1's four `f=3,7,11,15` captures fire.

**ArmController (`arm_controller.py`).** Publishes a `LowCmd_` on `rt/arm_sdk` at 50 Hz. To take control of arm joints from the locomotion service, it sets `motor_cmd[29].q = 1.0` (the `kNotUsedJoint` slot doubles as the arm_sdk handover weight). The same `LowCmd_` also pins legs (`LEG_JOINTS` 0–11) and waist (`WAIST_JOINTS` 12–14) at their startup pose with `kp_body_lock=300`, so locomotion stays balanced while arms are driven. Per-tick velocity is clamped via `_clip_target` against a single fixed `velocity_limit` (set at construction, not changed at runtime — `set_velocity_limit` is kept for callers but the inference pipeline doesn't use it). On exit, `disable_arm_sdk()` ramps the weight back to 0 to hand control to the loco service. **kp switching pattern**: `kp_arm` defaults to **150** (stiffer hold, less gravity sag during the standby wait); `arm.set_arm_kp(80)` is called once Enter is pressed (just before connecting to the policy server) to drop into the softer kp the model was trained against. Required precondition: robot is in **ai** motion mode and standing — `main.py` does not switch modes, operator sets it via the Unitree app.

**GripperController (`gripper_controller.py`).** Publishes `MotorCmds_` on `rt/dex1/left/cmd` and `rt/dex1/right/cmd` at 200 Hz with a `DELTA_GRIPPER_CMD=0.18 rad` per-tick rate cap (the publish thread, not the user, enforces this). State is read back from `rt/dex1/{left,right}/state`.

**CameraClient (`camera_client.py`).** Uses `teleimager.ImageClient` (host defaults to `192.168.123.164`, the G1 PC2). The head camera may be binocular; if so, only its **left half** is taken to match the training format. All three streams are converted BGR→RGB and resized to **256×320** (height × width) — these dimensions must match the model's expected input.

**PolicyClient (`policy_client.py`).** Synchronous `websockets.sync.client` connection with `ping_interval=None` so long inference calls (multi-second GPU work) don't trip the keepalive. The wire format is msgpack with NumPy support (`msgpack_numpy.py`, mirrors the server-side module byte-for-byte — keep them in sync).

**Startup sequence.** `arm.start()` → `move_to_pose(INIT_POSE_READY)` (5 s ramp at the operating vlim, no separate "init vlim") → grippers do a `close→open` sequence (each half of `--init-duration`) so the operator can see the dex1 has joined the SDK → `--settle-duration` pause → `input("")` standby (skipped with `--auto-start`) → `arm.set_arm_kp(args.inference_kp_arm)` (default 80) → connect to policy server. The arm and gripper publish threads keep streaming the current target during standby so the robot stays locked at `INIT_POSE_READY`.

**Shutdown safety.** A single `try`/`finally` block in `run()` wraps everything from gripper init onwards. On any exit path (normal completion, exception, or Ctrl+C — including during the `input("")` standby), the finally:
1. Calls `arm.stop()` **first** — this blocks until the publish thread has joined, so step 2 has exclusive access to `self.cmd`.
2. Calls `arm.disable_arm_sdk()` — ramps `motor_cmd[29].q` 1→0 over 1 s, returning arm authority to the loco service.
3. Calls `grip.stop()`, `cam.close()`, `policy.close()` conditionally (each may be `None` if the failure happened before that resource was constructed).
Order matters: doing `disable_arm_sdk` before `arm.stop()` would race the publish thread for CRC, occasionally publishing a malformed message.

**SDK channel choice.** G1 uses `unitree_sdk2py.idl.unitree_hg` (NOT `unitree_go`, which is for Go2/B2/H1). This is documented in `../example/g1/readme.md` and is easy to get wrong. The Dex1 grippers in `gripper_controller.py` are an exception — they use the older `unitree_go` IDL because dex1 is a separate accessory; do not "fix" this to match the arms.
