"""G1 arm controller for LingBot-VA inference.

Publishes 14 arm joint targets to rt/arm_sdk (motion-mode arm_sdk topic) at 50 Hz
with body-lock and velocity clamping. Adapted from
xr_teleoprate_copy/teleop/robot_control/robot_arm.py (G1_29_ArmController) but
stripped to the inference-only essentials.

Key responsibilities:
- Lock the legs and waist with high kp (so the locomotion service has authority
  cancelled for those joints — robot must be in ai mode standing on the ground)
- Stream the 14 arm joint targets at a fixed rate
- Rate-limit per-tick motion via velocity_limit (rad/s)
- Set motor_cmd[29].q = 1.0 to enable arm_sdk
- Compute CRC every tick
"""

import threading
import time
from enum import IntEnum

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


kTopicArmSDK = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"


class G1JointIndex(IntEnum):
    # Legs
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    # Waist
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    # Special: q=1 enables arm_sdk, q=0 returns control to locomotion service
    kNotUsedJoint = 29


# Joint indices in order matching the model output channels 0..13:
# out[0..6] -> left arm, out[7..13] -> right arm
ARM_JOINTS = [
    G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,   G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,     G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
    G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]
WAIST_JOINTS = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]
LEG_JOINTS = list(range(0, 12))  # joints 0..11


# Conservative joint limits (radians) for the G1 29-DoF arm, in the same order
# as ARM_JOINTS. Slightly tighter than the hardware limits to keep a margin.
# Order: [L pitch, L roll, L yaw, L elbow, L wristR, L wristP, L wristY,
#         R pitch, R roll, R yaw, R elbow, R wristR, R wristP, R wristY]
ARM_JOINT_MIN = np.array([
    -2.8,  -0.4, -2.4, -0.5, -1.9, -1.5, -1.5,   # left
    -2.8,  -2.2, -2.4, -0.5, -1.9, -1.5, -1.5,   # right
], dtype=np.float64)
ARM_JOINT_MAX = np.array([
     1.4,   2.2,  2.4,  2.9,  1.9,  1.5,  1.5,   # left
     1.4,   0.4,  2.4,  2.9,  1.9,  1.5,  1.5,   # right
], dtype=np.float64)

# A safe ready pose: arms slightly forward with elbows bent ~90° — palms facing
# each other in front of the robot. Good starting point for manipulation tasks.
# Order matches ARM_JOINTS. Tune to taste.
PI_2 = 1.5707963267948966
# INIT_POSE_READY = np.zeros(14)
INIT_POSE_READY = np.array([
     0.0,   0.0,  0.0,  -0.3,  0.0,  0.0,  0.0,   # left
     0.0,   0.0,  0.0,  -0.3,  0.0,  0.0,  0.0,   # right
], dtype=np.float64)

class ArmController:
    def __init__(self,
                 publish_hz: float = 50.0,
                 velocity_limit: float = 20.0,
                 kp_arm: float = 150.0,
                 kd_arm: float = 3.0,
                 kp_wrist: float = 40.0,
                 kd_wrist: float = 1.5,
                 kp_body_lock: float = 300.0,
                 kd_body_lock: float = 3.0):
        self.publish_hz = publish_hz
        self.control_dt = 1.0 / publish_hz
        self.velocity_limit = velocity_limit

        self.kp_arm = kp_arm
        self.kd_arm = kd_arm
        self.kp_wrist = kp_wrist
        self.kd_wrist = kd_wrist
        self.kp_body_lock = kp_body_lock
        self.kd_body_lock = kd_body_lock

        self.crc = CRC()
        self.cmd = unitree_hg_msg_dds__LowCmd_()

        self.pub = ChannelPublisher(kTopicArmSDK, LowCmd_)
        self.pub.Init()

        self.sub = ChannelSubscriber(kTopicLowState, LowState_)
        self.sub.Init()

        self._state_lock = threading.Lock()
        self._latest_state = None
        self._state_ready = threading.Event()

        self._target_lock = threading.Lock()
        self._q_target = None  # set on first state update

        self._stop = threading.Event()
        self._sub_thread = threading.Thread(target=self._subscribe_loop, daemon=True)
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)

    # ---------- public API ----------

    def start(self):
        self._sub_thread.start()
        if not self._state_ready.wait(timeout=10.0):
            raise RuntimeError("Timed out waiting for rt/lowstate")
        # initialize cmd with current state so the lock pose matches reality
        self._init_cmd_from_state()
        self._pub_thread.start()

    def stop(self):
        """Signal the subscribe + publish threads to exit and wait for them.

        Blocking until publish thread exits is important: callers may want to
        write to self.cmd / self.pub themselves after stop() (e.g.
        disable_arm_sdk), and racing against an active publish loop on the
        same cmd object would corrupt the published CRC.
        """
        self._stop.set()
        if self._pub_thread.is_alive():
            self._pub_thread.join(timeout=0.5)
        if self._sub_thread.is_alive():
            self._sub_thread.join(timeout=0.5)

    def get_arm_q(self) -> np.ndarray:
        """Return current 14-DoF arm joint positions."""
        with self._state_lock:
            state = self._latest_state
        if state is None:
            raise RuntimeError("No state received yet")
        return np.array([state.motor_state[j].q for j in ARM_JOINTS], dtype=np.float64)

    def set_arm_target(self, q_target: np.ndarray):
        """Set the 14-DoF arm joint target (radians). Thread-safe. Clipped to
        per-joint position limits."""
        if q_target.shape != (14,):
            raise ValueError(f"Expected shape (14,), got {q_target.shape}")
        clipped = np.clip(q_target, ARM_JOINT_MIN, ARM_JOINT_MAX)
        with self._target_lock:
            self._q_target = clipped.astype(np.float64).copy()

    def set_velocity_limit(self, vlim: float):
        """Update the per-tick velocity clamp at runtime.

        Currently unused in the inference pipeline (vlim is set once at
        construction). Kept for callers that might want to tighten the clamp
        during specific maneuvers.
        """
        self.velocity_limit = vlim

    def set_arm_kp(self, kp_arm: float):
        """Update kp for the 8 shoulder/elbow arm joints (wrist kp unchanged).

        Use to run a stiffer kp during init/standby — which reduces gravity
        sag — and switch to a softer kp for model-driven inference so the
        dynamics during inference match what the model was trained against.
        """
        self.kp_arm = kp_arm
        wrist_set = set([G1JointIndex.LeftWristRoll, G1JointIndex.LeftWristPitch,
                         G1JointIndex.LeftWristYaw,  G1JointIndex.RightWristRoll,
                         G1JointIndex.RightWristPitch, G1JointIndex.RightWristYaw])
        for j in ARM_JOINTS:
            if j not in wrist_set:
                self.cmd.motor_cmd[j].kp = kp_arm

    def move_to_pose(self, target_q: np.ndarray, duration: float = 4.0,
                     velocity_limit: float = 2.0):
        """Smoothly interpolate from current arm pose to target_q over `duration`
        seconds. Blocks until done. The publish thread keeps streaming at 50 Hz;
        this method just rewrites the target every tick to make the trajectory
        slow and predictable.

        velocity_limit overrides self.velocity_limit for the duration of the
        move (restored on exit). Pass the same value as the controller's vlim
        to keep the dynamics uniform — duration alone controls speed."""
        target_q = np.clip(np.asarray(target_q, dtype=np.float64),
                           ARM_JOINT_MIN, ARM_JOINT_MAX)
        start_q = self.get_arm_q().copy()

        saved_vlim = self.velocity_limit
        self.velocity_limit = velocity_limit
        try:
            steps = max(1, int(duration * self.publish_hz))
            for i in range(steps + 1):
                alpha = i / steps
                interp = start_q + alpha * (target_q - start_q)
                self.set_arm_target(interp)
                time.sleep(self.control_dt)
        finally:
            self.velocity_limit = saved_vlim

    def disable_arm_sdk(self):
        """Smoothly ramp motor_cmd[29].q from 1 to 0 to return arm control to
        the locomotion service. Call before exit if the robot is standing."""
        for w in np.linspace(1.0, 0.0, 50):
            self.cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = float(w)
            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            time.sleep(0.02)

    # ---------- internals ----------

    def _subscribe_loop(self):
        while not self._stop.is_set():
            msg = self.sub.Read()
            if msg is not None:
                with self._state_lock:
                    self._latest_state = msg
                if not self._state_ready.is_set():
                    self._state_ready.set()
            time.sleep(0.002)

    def _init_cmd_from_state(self):
        with self._state_lock:
            state = self._latest_state

        # Enable arm_sdk handover (overrides waist/arm authority on the loco service)
        self.cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

        # Lock legs at their current pose with high gains so they stay where
        # the locomotion service was already holding them. The loco service is
        # still active and balancing — we just provide a hold reference.
        for j in LEG_JOINTS:
            self.cmd.motor_cmd[j].mode = 1
            self.cmd.motor_cmd[j].q = state.motor_state[j].q
            self.cmd.motor_cmd[j].dq = 0.0
            self.cmd.motor_cmd[j].tau = 0.0
            self.cmd.motor_cmd[j].kp = self.kp_body_lock
            self.cmd.motor_cmd[j].kd = self.kd_body_lock

        # Rigidly clamp the waist at startup pose — keeps the torso upright so
        # arm motion doesn't trigger the balance controller to bend forward.
        for j in WAIST_JOINTS:
            self.cmd.motor_cmd[j].mode = 1
            self.cmd.motor_cmd[j].q = state.motor_state[j].q
            self.cmd.motor_cmd[j].dq = 0.0
            self.cmd.motor_cmd[j].tau = 0.0
            self.cmd.motor_cmd[j].kp = self.kp_body_lock
            self.cmd.motor_cmd[j].kd = self.kd_body_lock

        # Arm joints: split kp/kd between shoulder-elbow and wrist
        wrist_set = set([G1JointIndex.LeftWristRoll, G1JointIndex.LeftWristPitch,
                         G1JointIndex.LeftWristYaw,  G1JointIndex.RightWristRoll,
                         G1JointIndex.RightWristPitch, G1JointIndex.RightWristYaw])
        for j in ARM_JOINTS:
            self.cmd.motor_cmd[j].mode = 1
            self.cmd.motor_cmd[j].q = state.motor_state[j].q
            self.cmd.motor_cmd[j].dq = 0.0
            self.cmd.motor_cmd[j].tau = 0.0
            if j in wrist_set:
                self.cmd.motor_cmd[j].kp = self.kp_wrist
                self.cmd.motor_cmd[j].kd = self.kd_wrist
            else:
                self.cmd.motor_cmd[j].kp = self.kp_arm
                self.cmd.motor_cmd[j].kd = self.kd_arm

        # Seed q_target with current arm pose so the first publish doesn't jump
        with self._target_lock:
            self._q_target = np.array([state.motor_state[j].q for j in ARM_JOINTS],
                                      dtype=np.float64)

    def _clip_target(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        delta = q_target - q_current
        motion_scale = np.max(np.abs(delta)) / (self.velocity_limit * self.control_dt)
        return q_current + delta / max(motion_scale, 1.0)

    def _publish_loop(self):
        while not self._stop.is_set():
            t0 = time.time()

            with self._target_lock:
                q_target = self._q_target.copy()

            q_current = self.get_arm_q()
            q_cmd = self._clip_target(q_target, q_current)

            for idx, j in enumerate(ARM_JOINTS):
                self.cmd.motor_cmd[j].q = float(q_cmd[idx])
                self.cmd.motor_cmd[j].dq = 0.0
                self.cmd.motor_cmd[j].tau = 0.0

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)

            elapsed = time.time() - t0
            sleep = self.control_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
