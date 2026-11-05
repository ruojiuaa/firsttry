"""CPU regression tests. Run with Python plus numpy/scipy/pydantic/PyYAML."""

import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from collect import collect
from controls import ControlMode
from controls import bounded_action
from hardware.teleop import MouseSample
import numpy as np
from policy_worker import PolicyWorker
from prepare_training import human_segments
from prepare_training import prepare
from recording import Recorder
from settings import CONTRACT
from settings import HERE
from settings import load_settings


class MemoryRecorder:
    def __init__(self, cfg):
        self.frames = []
        self.episodes = []
        self.closed = False

    def add(self, frame):
        self.frames.append(frame)

    def finish(self, outcome, reason=""):
        self.episodes.append((outcome, self.frames))
        self.frames = []

    def close(self):
        self.closed = True


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for subprocess")


def serve_test_policy(port):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    class Policy:
        def infer(self, obs):
            result = np.zeros((10, 7), dtype=np.float32)
            result[:, 6] = obs["grasp_command"]
            return {"actions": result}

    WebsocketPolicyServer(
        Policy(),
        host="127.0.0.1",
        port=port,
        metadata={"ur5_dagger_contract": CONTRACT, "fps": 30},
    ).serve_forever()


class DaggerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_settings(HERE / "mock.yaml")

    def test_config_rejects_unknown_and_nonfinite_values(self):
        from settings import Settings

        raw = self.cfg.model_dump()
        raw["dagger"]["misspelled"] = True
        with self.assertRaises(ValueError):
            Settings.model_validate(raw)
        raw["dagger"].pop("misspelled")
        raw["dagger"]["max_translation_speed"] = float("nan")
        with self.assertRaises(ValueError):
            Settings.model_validate(raw)

    def test_real_config_validates_without_device_access(self):
        cfg = load_settings(HERE / "real.example.yaml")
        self.assertEqual(cfg.robot.control_hz, 125)
        self.assertEqual(cfg.gripper.feedback_kwargs["robot_ip"], cfg.robot.ip)
        self.assertFalse(cfg.robot.motion_enabled)

    def test_takeover_and_zero_motion_remain_human(self):
        resets = []
        mode = ControlMode("policy", lambda: resets.append(True))
        sample = MouseSample(np.array([0.8, 0, 0, 0, 0, 0]), True, 0)
        self.assertTrue(mode.update("", sample, self.cfg, 0))
        sample.axes[:] = 0
        self.assertFalse(mode.update("", sample, self.cfg, 0))
        self.assertEqual(mode.mode, "human")
        self.assertTrue(mode.update("p", sample, self.cfg, 0))
        self.assertEqual(len(resets), 2)

    def test_gripper_button_takes_over_policy(self):
        mode = ControlMode("policy", lambda: None)
        self.assertTrue(mode.update("", MouseSample(np.zeros(6), True, 1), self.cfg, 0))
        self.assertEqual(mode.mode, "human")

    def test_action_clipping_rotation_and_gripper_quantization(self):
        pose = np.array(self.cfg.robot.mock_initial_pose)
        target, action = bounded_action(pose, [1, 1, 1, 1, 1, 1, 0.6], self.cfg)
        self.assertLessEqual(np.linalg.norm(action[:3]), self.cfg.dagger.max_translation_speed / 30 + 1e-12)
        self.assertLessEqual(np.linalg.norm(action[3:6]), self.cfg.dagger.max_rotation_speed / 30 + 1e-12)
        self.assertEqual(action[6], 1)
        self.assertTrue(np.isfinite(target).all())
        _, undershoot = bounded_action(pose, [0, 0, 0, 0, 0, 0, -0.05], self.cfg)
        self.assertEqual(undershoot[6], 0)
        with self.assertRaises(ValueError):
            bounded_action(pose, [0, 0, 0, 0, 0, 0, float("nan")], self.cfg)

    def test_human_segments_do_not_bridge_policy_frames(self):
        self.assertEqual(list(human_segments([False, True, True, False, True])), [(1, 3), (4, 5)])
        self.assertEqual(list(human_segments([])), [])

    def test_late_inference_reply_is_discarded(self):
        cfg = self.cfg.dagger.policy.model_copy(update={"mock_delay_s": 0.15, "max_result_age_s": 5})
        worker = PolicyWorker(cfg, mock=True)
        try:

            def ready():
                worker.poll()
                return worker.ready

            wait_until(ready)
            self.assertIsNone(worker.action({"grasp_command": 0}))
            before = time.monotonic()
            worker.reset()
            self.assertLess(time.monotonic() - before, 0.05)

            def finished():
                worker.poll()
                return worker.pending_at is None

            wait_until(finished)
            self.assertEqual(worker.actions, [])
            self.assertIsNone(worker.action({"grasp_command": 1}))
            wait_until(finished)
            np.testing.assert_equal(worker.action({"grasp_command": 1})[6], 1)
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_continuous_mock_collection_two_interventions_per_episode(self):
        cfg = self.cfg
        cfg.dataset.episode_seconds = 3
        cfg.dataset.episodes = 2
        holder = []

        def factory(config):
            recorder = MemoryRecorder(config)
            holder.append(recorder)
            return recorder

        collect(cfg, auto_demo=True, show_preview=False, recorder_factory=factory)
        recorder = holder[0]
        self.assertTrue(recorder.closed)
        self.assertEqual(len(recorder.episodes), 2)
        for outcome, frames in recorder.episodes:
            self.assertEqual(outcome, "success")
            self.assertEqual(len(frames), 90)
            flags = [bool(frame["complementary_info.intervention"][0]) for frame in frames]
            self.assertEqual(list(human_segments(flags)), [(18, 36), (54, 72)])
            sources = {int(frame["complementary_info.action_source"][0]) for frame in frames}
            self.assertIn(1, sources)
            self.assertIn(2, sources)
            for frame in frames:
                self.assertEqual(frame["observation.state"].shape, (8,))
                self.assertEqual(frame["action"].shape, (7,))

    def test_episode_outcome_commit_and_intervention_count(self):
        class Dataset:
            @classmethod
            def create(cls, **kwargs):
                obj = cls()
                obj.meta = types.SimpleNamespace(total_episodes=0)
                Path(kwargs["root"]).mkdir()
                return obj

            def add_frame(self, frame):
                pass

            def save_episode(self):
                self.meta.total_episodes += 1

            def stop_image_writer(self):
                pass

        module = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
        module.CODEBASE_VERSION = "v2.1"
        module.LeRobotDataset = Dataset
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {module.__name__: module}):
            self.cfg.dataset.root = str(Path(temporary) / "recording")
            recorder = Recorder(self.cfg)
            for flag in [False, True, True, False, True]:
                recorder.add({"complementary_info.intervention": np.array([flag])})
            recorder.finish("failure")
            annotation = json.loads((recorder.root / "meta/dagger/episode_000000.json").read_text())
            self.assertFalse(annotation["success"])
            self.assertEqual(annotation["intervention_count"], 2)
            self.assertTrue(annotation["committed"])
            self.assertEqual(recorder.frame_count, 0)
            recorder.close()

    def test_inference_process_failure_allows_human_control(self):
        worker = PolicyWorker(self.cfg.dagger.policy, mock=True)
        try:
            worker.process.terminate()
            worker.process.join(3)
            worker.poll()
            self.assertIsNotNone(worker.error)
            mode = ControlMode("policy", worker.reset)
            self.assertTrue(mode.update("h", MouseSample(np.zeros(6), True, 0), self.cfg, 0))
            self.assertEqual(mode.mode, "human")
        finally:
            worker.close()


@unittest.skipUnless(importlib.util.find_spec("openpi_client"), "OpenPI client is not installed")
class WebsocketIntegrationTests(unittest.TestCase):
    def test_existing_server_client_and_new_policy_worker(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        server = mp.get_context("spawn").Process(target=serve_test_policy, args=(port,), daemon=True)
        server.start()
        cfg = load_settings(HERE / "mock.yaml").dagger.policy.model_copy(
            update={
                "port": port,
                "timeout_s": 10,
                "max_result_age_s": 10,
            }
        )
        worker = PolicyWorker(cfg, mock=False)
        try:

            def ready():
                worker.poll()
                if worker.error:
                    raise AssertionError(worker.error)
                return worker.ready

            wait_until(ready, timeout=30)
            self.assertIsNone(worker.action({"grasp_command": 1}))

            def finished():
                worker.poll()
                return bool(worker.actions)

            wait_until(finished)
            result = worker.action({"grasp_command": 1})
            self.assertEqual(result.shape, (7,))
            self.assertEqual(result[6], 1)
        finally:
            worker.close()
            server.terminate()
            server.join(3)


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "Pinned LeRobot is not installed")
class LeRobotIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_three_video_streams_can_be_written_and_decoded(self):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        cfg = load_settings(HERE / "mock.yaml")
        cfg.dataset.episodes = 1
        cfg.dataset.episode_seconds = 1
        with tempfile.TemporaryDirectory() as temporary:
            cfg.dataset.root = str(Path(temporary) / "videos")
            collect(cfg, auto_demo=True, show_preview=False)
            videos = list(Path(cfg.dataset.root).glob("videos/**/*.mp4"))
            self.assertEqual(len(videos), 3)
            dataset = LeRobotDataset(cfg.dataset.repo_id, root=Path(cfg.dataset.root), video_backend="pyav")
            self.assertEqual(dataset.meta.total_frames, 30)
            import av

            # Inspect the actual encoded stream independently of torchvision's optional VideoReader.
            for video in videos:
                with av.open(str(video)) as container:
                    frames = list(container.decode(video=0))
                    self.assertEqual(len(frames), 30)
                    self.assertEqual(frames[0].to_ndarray(format="rgb24").shape, (72, 96, 3))

    def test_real_writer_readback_and_human_segment_export(self):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        cfg = load_settings(HERE / "mock.yaml")
        cfg.dataset.episodes = 1
        cfg.dataset.episode_seconds = 2
        cfg.dagger.video = False
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            cfg.dataset.root = str(temporary / "rollout")
            collect(cfg, auto_demo=True, show_preview=False)
            original = LeRobotDataset(cfg.dataset.repo_id, root=Path(cfg.dataset.root))
            self.assertEqual(len(original), 60)
            self.assertEqual(tuple(original[0]["observation.state"].shape), (8,))
            annotation = json.loads((Path(cfg.dataset.root) / "meta/dagger/episode_000000.json").read_text())
            self.assertEqual(annotation["intervention_count"], 2)
            self.assertTrue(annotation["committed"])
            config_path = temporary / "config.yaml"
            import yaml

            config_path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
            output = temporary / "training"
            prepare(
                argparse.Namespace(
                    config=str(config_path),
                    output=str(output),
                    repo_id="local/test_training",
                    demo=[],
                    hil=[cfg.dataset.root],
                    min_frames=10,
                    include_aborted=False,
                )
            )
            training = LeRobotDataset("local/test_training", root=output)
            self.assertEqual(training.meta.total_episodes, 2)
            self.assertEqual(len(training), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
