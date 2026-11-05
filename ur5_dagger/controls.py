"""Nonblocking terminal input, preview and bounded TCP action conversion."""

import os
import sys

from hardware.geometry import compose
from hardware.geometry import difference
from hardware.geometry import limit_norm
from hardware.geometry import vector
import numpy as np


class TerminalKeys:
    def __enter__(self):
        self.saved = None
        if os.name != "nt" and sys.stdin.isatty():
            import termios
            import tty

            self.saved = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        if os.name == "nt":
            import msvcrt

            return msvcrt.getwch().lower() if msvcrt.kbhit() else ""
        import select

        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return ""

    def __exit__(self, *args):
        if self.saved is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.saved)


class Preview:
    def __init__(self, enabled):
        self.cv2 = None
        if enabled:
            import cv2

            self.cv2 = cv2

    def update(self, images, mode, index):
        if self.cv2 is None:
            return ""
        cv2 = self.cv2
        tiles = [cv2.resize(frame.rgb, (320, 240)) for frame in images.values()]
        canvas = np.concatenate(tiles, axis=1)[..., ::-1].copy()
        cv2.putText(
            canvas,
            f"{mode} | frame {index} | H human P policy Space pause S success F failure Q quit",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        cv2.imshow("UR5 DAgger", canvas)
        key = cv2.waitKey(1) & 0xFF
        return chr(key).lower() if key != 255 else ""

    def close(self):
        if self.cv2 is not None:
            self.cv2.destroyAllWindows()


def bounded_action(pose, action, cfg):
    action = vector(action, 7).copy()
    dt = 1 / cfg.dataset.fps
    action[:3] = limit_norm(action[:3], cfg.dagger.max_translation_speed * dt)
    action[3:6] = limit_norm(action[3:6], cfg.dagger.max_rotation_speed * dt)
    # A regressed binary command may slightly overshoot [0,1]. Quantize at the label midpoint.
    action[6] = int(action[6] >= 0.5)
    target = compose(pose, action[:6])
    target[:3] = np.clip(target[:3], cfg.robot.workspace_min, cfg.robot.workspace_max)
    action[:6] = difference(pose, target)
    return target, action


class ControlMode:
    def __init__(self, initial, reset):
        self.mode = initial
        self.reset = reset

    def change(self, mode):
        if mode == self.mode:
            return False
        self.mode = mode
        self.reset()
        print(f"Control -> {mode}", flush=True)
        return True

    def update(self, key, sample, cfg, last_grasp):
        requested = {"h": "human", "p": "policy", " ": "hold"}.get(key, self.mode)
        moved = sample.enabled and np.max(np.abs(sample.axes)) >= cfg.dagger.takeover_threshold
        gripper_pressed = sample.grasp != last_grasp
        if requested == "policy" and cfg.dagger.takeover_on_motion and (moved or gripper_pressed):
            requested = "human"
        return self.change(requested)
