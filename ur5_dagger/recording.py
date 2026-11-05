"""LeRobot v2.1 frames plus crash-recoverable episode outcome annotations."""

import json
import os
from pathlib import Path

import numpy as np
from settings import CAMERAS
from settings import CONTRACT

STATE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "left_finger_m", "right_finger_m"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grasp"]


def features(cfg):
    def numeric(dtype, names):
        return {"dtype": dtype, "shape": (len(names),), "names": names}

    result = {
        "observation.state": numeric("float32", STATE_NAMES),
        "action": numeric("float32", ACTION_NAMES),
        "complementary_info.intervention": numeric("bool", ["human_control"]),
        "complementary_info.action_source": numeric("int64", ["0_hold_1_policy_2_human"]),
        "complementary_info.motion_enabled": numeric("bool", ["motion_enabled"]),
        "complementary_info.target_pose": numeric("float32", STATE_NAMES[:6]),
        "complementary_info.host_time": numeric("float64", ["monotonic_s"]),
        "complementary_info.robot_host_time": numeric("float64", ["monotonic_s"]),
        "complementary_info.gripper_host_time": numeric("float64", ["monotonic_s"]),
        "complementary_info.camera_host_times": numeric("float64", list(CAMERAS)),
        "complementary_info.camera_frame_numbers": numeric("int64", list(CAMERAS)),
    }
    for camera in cfg.cameras:
        result[f"observation.images.{camera.name}"] = {
            "dtype": "video" if cfg.dagger.video else "image",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def create_dataset(cfg, root, repo_id, dataset_features):
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Use this repository's pinned LeRobot v2.1, got {CODEBASE_VERSION}")

    class CodecDataset(LeRobotDataset):
        def encode_episode_videos(self, episode_index):
            from lerobot.common.datasets.video_utils import encode_video_frames

            paths = {}
            for key in self.meta.video_keys:
                path = self.root / self.meta.get_video_file_path(episode_index, key)
                if not path.exists():
                    images = self._get_image_file_path(episode_index, key, 0).parent
                    encode_video_frames(images, path, self.fps, vcodec=cfg.dataset.vcodec, overwrite=True)
                paths[key] = str(path)
            return paths

    return CodecDataset.create(
        repo_id=repo_id,
        root=root,
        fps=cfg.dataset.fps,
        robot_type="ur5_rg2",
        features=dataset_features,
        use_videos=cfg.dagger.video,
        image_writer_threads=cfg.dataset.image_writer_threads,
    )


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.dataset.root)
        # Each invocation owns a new directory. Existing recordings are never overwritten.
        self.dataset = create_dataset(cfg, self.root, cfg.dataset.repo_id, features(cfg))
        self.frame_count = 0
        self.interventions = 0
        self.previous_human = False
        atomic_json(
            self.root / "meta/dagger_session.json",
            {
                "contract": CONTRACT,
                "configuration": cfg.model_dump(),
                "gripper_state_source": "mock" if cfg.mode == "mock" else "RG2 worker reported target width",
            },
        )

    @property
    def episode_index(self):
        return self.dataset.meta.total_episodes

    def add(self, frame):
        self.dataset.add_frame(frame)
        human = bool(frame["complementary_info.intervention"][0])
        if human and not self.previous_human:
            self.interventions += 1
        self.previous_human = human
        self.frame_count += 1

    def finish(self, outcome, reason=""):
        if not self.frame_count:
            return
        index = self.episode_index
        annotation = {
            "episode_index": index,
            "outcome": outcome,
            "success": {"success": True, "failure": False, "aborted": None}[outcome],
            "reason": reason,
            "length": self.frame_count,
            "intervention_count": self.interventions,
            "autonomous_success": outcome == "success" and self.interventions == 0,
            "committed": False,
        }
        path = self.root / f"meta/dagger/episode_{index:06d}.json"
        # Persist outcome first. A failure during save_episode leaves an explicit pending annotation.
        atomic_json(path, annotation)
        self.dataset.save_episode()
        annotation["committed"] = True
        atomic_json(path, annotation)
        self.frame_count = self.interventions = 0
        self.previous_human = False

    def close(self):
        self.dataset.stop_image_writer()


def make_frame(cfg, state, gripper, images, action, target, source, now):
    return {
        "observation.state": np.r_[state.pose, gripper.qpos].astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "complementary_info.intervention": np.array([source == 2], dtype=bool),
        "complementary_info.action_source": np.array([source], dtype=np.int64),
        "complementary_info.motion_enabled": np.array([source != 0], dtype=bool),
        "complementary_info.target_pose": np.asarray(target, dtype=np.float32),
        "complementary_info.host_time": np.array([now], dtype=np.float64),
        "complementary_info.robot_host_time": np.array([state.host_time], dtype=np.float64),
        "complementary_info.gripper_host_time": np.array([gripper.host_time], dtype=np.float64),
        "complementary_info.camera_host_times": np.array([images[c].host_time for c in CAMERAS]),
        "complementary_info.camera_frame_numbers": np.array([images[c].frame_number for c in CAMERAS], dtype=np.int64),
        **{f"observation.images.{c}": images[c].rgb for c in CAMERAS},
        "task": cfg.dataset.task,
    }
