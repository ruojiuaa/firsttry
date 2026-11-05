"""Continuous UR5/SpaceMouse HIL collection into a new LeRobot v2.1 dataset."""

import argparse
from contextlib import ExitStack
import time

from controls import ControlMode
from controls import Preview
from controls import TerminalKeys
from controls import bounded_action
from hardware.cameras import CameraRig
from hardware.geometry import target_from_axes
from hardware.gripper import GripperFeedback
from hardware.robot import UR5Backend
from hardware.teleop import SpaceMouseInput
import numpy as np
from policy_worker import PolicyWorker
from recording import Recorder
from recording import atomic_json
from recording import make_frame
from settings import CAMERAS
from settings import HERE
from settings import load_settings


def observation(cfg, state, gripper, images, grasp):
    return {
        "observation.state": np.r_[state.pose, gripper.qpos].astype(np.float32),
        **{f"observation.images.{name}": images[name].rgb for name in CAMERAS},
        "prompt": cfg.dataset.task,
        "grasp_command": grasp,
    }


def stop_motion(robot, gripper):
    try:
        robot.stop()
    finally:
        gripper.stop()


def run_episode(cfg, robot, gripper, mouse, rig, policy, recorder, preview, *, auto_demo=False):
    policy.reset()
    mouse.start_episode()
    mode = ControlMode("policy" if auto_demo else cfg.dagger.start_mode, policy.reset)
    last_grasp = mouse.grasp
    previous_source = 0
    preview_key = ""
    period = 1 / cfg.dataset.fps
    total_frames = round(cfg.dataset.episode_seconds * cfg.dataset.fps)
    with TerminalKeys() as keys:
        for index in range(total_frames):
            tick = time.monotonic()
            key = keys.read() or preview_key
            if auto_demo:
                # Two interventions with a policy segment between them.
                phase = index / total_frames
                key = "h" if 0.2 <= phase < 0.4 or 0.6 <= phase < 0.8 else "p"
            if key in ("s", "f", "q", "a"):
                return {"s": "success", "f": "failure", "q": "aborted", "a": "aborted"}[key], key == "q"
            mouse.grasp = last_grasp
            sample = mouse.read()
            if auto_demo:
                # Mock axes exercise human motion without triggering automatic takeover in policy sections.
                sample.axes[:] = 0
                sample.grasp = last_grasp
                if key == "h":
                    sample.axes[1] = 0.4
            if key == "g" and mode.mode == "human":
                mouse.toggle_grasp()
                sample.grasp = mouse.grasp
            if mode.update(key, sample, cfg, last_grasp):
                stop_motion(robot, gripper)
                previous_source = 0
            images = rig.read()
            state = robot.get_state(cfg.synchronization.max_robot_age_s)
            feedback = gripper.read()
            if time.monotonic() - feedback.host_time > cfg.gripper.feedback_kwargs.get("feedback_timeout_s", 0.25):
                raise RuntimeError("Gripper feedback is stale")
            source = 0
            action = np.array([0, 0, 0, 0, 0, 0, last_grasp], dtype=np.float64)
            if mode.mode == "human":
                _, delta = target_from_axes(state.pose, sample.axes, cfg, sample.enabled, sample.high_speed)
                action = np.r_[delta, sample.grasp]
                source = 2
            elif mode.mode == "policy":
                try:
                    result = policy.action(observation(cfg, state, feedback, images, last_grasp))
                    if result is not None:
                        action = result
                        source = 1
                except RuntimeError as exc:
                    mode.change("hold")
                    print(f"Policy paused. {exc}. Press H for human control.", flush=True)
            else:
                policy.poll()
            target, action = bounded_action(state.pose, action, cfg)
            if source:
                robot.send_target(target, period, True)
                gripper.update_command(int(action[6]), enabled=True)
            elif previous_source:
                stop_motion(robot, gripper)
            last_grasp = int(action[6])
            previous_source = source
            recorder.add(make_frame(cfg, state, feedback, images, action, target, source, time.monotonic()))
            preview_key = preview.update(images, mode.mode, index)
            elapsed = time.monotonic() - tick
            if cfg.mode == "real" and elapsed > period + cfg.synchronization.max_loop_lateness_s:
                raise RuntimeError(f"Control/recording cycle too slow ({elapsed:.3f}s)")
            time.sleep(max(0, period - elapsed))
    return ("success" if auto_demo else "aborted"), False


def collect(cfg, *, auto_demo=False, show_preview=True, recorder_factory=Recorder):
    mock = cfg.mode == "mock"
    if auto_demo and not mock:
        raise ValueError("--auto-demo is only available with mode=mock")
    if not mock and not (cfg.robot.motion_enabled and cfg.robot.tcp_configured and cfg.robot.workspace_configured):
        raise ValueError("Set motion_enabled, tcp_configured and workspace_configured after checking the real setup")
    with ExitStack() as stack:
        recorder = recorder_factory(cfg)
        stack.callback(recorder.close)
        preview = Preview(show_preview)
        stack.callback(preview.close)
        policy = PolicyWorker(cfg.dagger.policy, mock=mock, fps=cfg.dataset.fps)
        stack.callback(policy.close)
        robot = UR5Backend(cfg.robot, mock=mock)
        stack.callback(robot.disconnect)
        gripper = GripperFeedback(cfg.gripper, mock=mock)
        stack.callback(gripper.disconnect)
        rig = CameraRig(cfg)
        stack.callback(rig.disconnect)
        mouse = SpaceMouseInput(cfg.teleop, mock=mock, gripper_cfg=cfg.gripper)
        stack.callback(mouse.disconnect)
        # External Control hosts the RG2 worker. Connect robot before reading RG2 registers.
        robot.connect()
        gripper.connect()
        mouse.connect()
        rig.connect()
        if hasattr(recorder, "root"):
            atomic_json(recorder.root / "meta/dagger_cameras.json", rig.metadata())
        print("H human | P policy | Space hold | G gripper | S success | F failure | A abort | Q save-abort and quit")
        for episode in range(cfg.dataset.episodes):
            if not auto_demo:
                input(f"Episode {episode + 1}. Put the robot in its start position and press Enter. ")
            try:
                outcome, quit_requested = run_episode(
                    cfg, robot, gripper, mouse, rig, policy, recorder, preview, auto_demo=auto_demo
                )
            except BaseException as exc:
                try:
                    stop_motion(robot, gripper)
                finally:
                    recorder.finish("aborted", f"{type(exc).__name__}: {exc}")
                raise
            finally:
                stop_motion(robot, gripper)
                policy.reset()
            recorder.finish(outcome, "operator_or_time_limit" if outcome == "aborted" else "")
            print(f"Saved episode {episode + 1}, outcome={outcome}", flush=True)
            if quit_requested:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "mock.yaml"))
    parser.add_argument("--output", help="New dataset directory, relative to repository root or absolute")
    parser.add_argument("--auto-demo", action="store_true", help="Mock policy/human/policy transitions")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate config without connecting devices")
    args = parser.parse_args()
    cfg = load_settings(args.config)
    if args.output:
        from pathlib import Path

        from settings import ROOT

        cfg.dataset.root = str((ROOT / Path(args.output).expanduser()).resolve())
    if args.validate:
        print(cfg.model_dump_json(indent=2))
        return
    collect(cfg, auto_demo=args.auto_demo, show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
