"""G1 Dex1 gripper controller.

Publishes left/right gripper targets to rt/dex1/{left,right}/cmd at 200 Hz with
±0.18 rad/step rate limit (gripper rail = 0.6 rad/cm, so 0.18 rad ≈ 3 mm/cycle).
Adapted from xr_teleoprate_copy/teleop/robot_control/robot_hand_unitree.py."""

import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

GRIPPER_MIN = 0.0   # fully closed
GRIPPER_MAX = 5.4   # fully open (rail stroke ≈ 9 cm at 0.6 rad/cm)
DELTA_GRIPPER_CMD = 0.18  # rate limit: rad per publish tick


def _make_cmd_msg(kp: float, kd: float) -> MotorCmds_:
    msg = MotorCmds_()
    msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
    msg.cmds[0].q = 0.0
    msg.cmds[0].dq = 0.0
    msg.cmds[0].tau = 0.0
    msg.cmds[0].kp = kp
    msg.cmds[0].kd = kd
    return msg


class GripperController:
    def __init__(self, publish_hz: float = 200.0, kp: float = 5.0, kd: float = 0.05):
        self.publish_hz = publish_hz
        self.control_dt = 1.0 / publish_hz
        self.kp = kp
        self.kd = kd

        self.left_pub = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.left_pub.Init()
        self.right_pub = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.right_pub.Init()

        self.left_sub = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.left_sub.Init()
        self.right_sub = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.right_sub.Init()

        self.left_msg = _make_cmd_msg(self.kp, self.kd)
        self.right_msg = _make_cmd_msg(self.kp, self.kd)

        self._state_lock = threading.Lock()
        self._left_q = 0.0
        self._right_q = 0.0
        self._state_ready = threading.Event()

        self._target_lock = threading.Lock()
        self._left_target = (GRIPPER_MIN + GRIPPER_MAX) / 2.0
        self._right_target = (GRIPPER_MIN + GRIPPER_MAX) / 2.0

        self._stop = threading.Event()
        self._sub_thread = threading.Thread(target=self._subscribe_loop, daemon=True)
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)

    def start(self):
        self._sub_thread.start()
        if not self._state_ready.wait(timeout=10.0):
            raise RuntimeError("Timed out waiting for rt/dex1 state")
        with self._state_lock:
            self._left_target = self._left_q
            self._right_target = self._right_q
        self._pub_thread.start()

    def stop(self):
        self._stop.set()

    def set_targets(self, left: float, right: float):
        left = float(np.clip(left, GRIPPER_MIN, GRIPPER_MAX))
        right = float(np.clip(right, GRIPPER_MIN, GRIPPER_MAX))
        with self._target_lock:
            self._left_target = left
            self._right_target = right

    def move_to_targets(self, left: float, right: float, duration: float = 2.0):
        """Smoothly interpolate gripper targets over `duration` seconds. Blocks
        until done. The publish thread continues at 200 Hz with its built-in
        rate limit; this just steps the target."""
        l_start, r_start = self.get_state()
        l_end = float(np.clip(left, GRIPPER_MIN, GRIPPER_MAX))
        r_end = float(np.clip(right, GRIPPER_MIN, GRIPPER_MAX))

        steps = max(1, int(duration * self.publish_hz))
        for i in range(steps + 1):
            alpha = i / steps
            li = l_start + alpha * (l_end - l_start)
            ri = r_start + alpha * (r_end - r_start)
            self.set_targets(li, ri)
            time.sleep(self.control_dt)

    def get_state(self) -> tuple:
        with self._state_lock:
            return self._left_q, self._right_q

    def _subscribe_loop(self):
        while not self._stop.is_set():
            l = self.left_sub.Read()
            r = self.right_sub.Read()
            if l is not None and r is not None:
                with self._state_lock:
                    self._left_q = l.states[0].q
                    self._right_q = r.states[0].q
                if not self._state_ready.is_set():
                    self._state_ready.set()
            time.sleep(0.002)

    def _publish_loop(self):
        while not self._stop.is_set():
            t0 = time.time()

            with self._target_lock:
                lt, rt = self._left_target, self._right_target
            with self._state_lock:
                lq, rq = self._left_q, self._right_q

            # Rate limit
            l_cmd = float(np.clip(lt, lq - DELTA_GRIPPER_CMD, lq + DELTA_GRIPPER_CMD))
            r_cmd = float(np.clip(rt, rq - DELTA_GRIPPER_CMD, rq + DELTA_GRIPPER_CMD))

            self.left_msg.cmds[0].q = l_cmd
            self.right_msg.cmds[0].q = r_cmd
            self.left_pub.Write(self.left_msg)
            self.right_pub.Write(self.right_msg)

            elapsed = time.time() - t0
            sleep = self.control_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
