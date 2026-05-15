"""Camera wrapper for LingBot-VA inference.

Subscribes to the three G1 cameras via teleimager.ImageClient and returns each
as a 256x320 RGB numpy array (uint8) matching the training format:
- cam_left_high: left half of the binocular head camera, BGR→RGB, resized
- cam_left_wrist:  left wrist camera, BGR→RGB, resized
- cam_right_wrist: right wrist camera, BGR→RGB, resized

Resize: source 480x640 → 256x320 (preserves 4:3 aspect ratio).
"""

import cv2
import numpy as np

try:
    from teleimager import ImageClient
except ImportError:
    # The installed teleimager package may not re-export ImageClient at the
    # package root — import from the submodule directly.
    from teleimager.image_client import ImageClient


CAM_HEIGHT = 256
CAM_WIDTH = 320


class CameraClient:
    def __init__(self, host: str = "192.168.123.164", request_port: int = 60000):
        # request_bgr=True so .bgr is decoded for us
        self._client = ImageClient(host=host, request_port=request_port, request_bgr=True)
        self._cfg = self._client.get_cam_config()

        if not self._cfg["head_camera"]["enable_zmq"]:
            raise RuntimeError("head_camera ZMQ not enabled on image server")
        if not self._cfg["left_wrist_camera"]["enable_zmq"]:
            raise RuntimeError("left_wrist_camera ZMQ not enabled on image server")
        if not self._cfg["right_wrist_camera"]["enable_zmq"]:
            raise RuntimeError("right_wrist_camera ZMQ not enabled on image server")

        self._head_binocular = bool(self._cfg["head_camera"].get("binocular", False))

    @staticmethod
    def _bgr_to_rgb_resize(bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != CAM_HEIGHT or rgb.shape[1] != CAM_WIDTH:
            rgb = cv2.resize(rgb, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
        return rgb

    def _take_left_half(self, bgr: np.ndarray) -> np.ndarray:
        if not self._head_binocular:
            return bgr
        # Binocular feed is two views concatenated along width; take the left half.
        w = bgr.shape[1]
        return bgr[:, : w // 2, :]

    def get_obs_images(self) -> dict:
        """Return just the three camera frames as a dict (no prompt key).

        Note: head/lw/rw are fetched sequentially — they are a few ms apart in
        time, not a true simultaneous snapshot.
        """
        head = self._client.get_head_frame()
        lw = self._client.get_left_wrist_frame()
        rw = self._client.get_right_wrist_frame()

        if head.bgr is None or lw.bgr is None or rw.bgr is None:
            raise RuntimeError("Got None from one or more cameras")

        head_bgr = self._take_left_half(head.bgr)
        return {
            "observation.images.cam_left_high":   self._bgr_to_rgb_resize(head_bgr),
            "observation.images.cam_left_wrist":  self._bgr_to_rgb_resize(lw.bgr),
            "observation.images.cam_right_wrist": self._bgr_to_rgb_resize(rw.bgr),
        }

    def get_obs(self, prompt: str) -> dict:
        """Full inference-payload obs: images + task prompt."""
        obs = self.get_obs_images()
        obs["task"] = prompt
        return obs

    def close(self):
        self._client.close()
