"""Aggregate complete expert demos and contiguous human intervention segments."""

import argparse
import json

import numpy as np
from recording import atomic_json
from recording import create_dataset
from recording import features
from settings import CAMERAS
from settings import HERE
from settings import ROOT
from settings import load_settings


def human_segments(flags):
    """Keep contiguous spans separate so action chunks never cross a policy span."""
    start = None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            yield start, index
            start = None


def image_uint8(image):
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image


def prepare(args):
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("Use the repository's pinned LeRobot v2.1")
    cfg = load_settings(args.config)
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    selected_features = {
        key: value
        for key, value in features(cfg).items()
        if key in ("observation.state", "action") or key.startswith("observation.images.")
    }
    destination = None
    provenance = []
    skipped = 0
    try:
        for source_kind, paths in (("demo", args.demo), ("hil", args.hil)):
            for source_path in paths:
                root = (ROOT / source_path).resolve()
                info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
                if info["fps"] != cfg.dataset.fps:
                    raise ValueError(f"FPS mismatch in {root}")
                for key, expected in selected_features.items():
                    actual = info["features"].get(key, {})
                    if tuple(actual.get("shape", [])) != tuple(expected["shape"]):
                        raise ValueError(f"Feature shape mismatch for {key} in {root}")
                source = LeRobotDataset(info.get("repo_id", "local/source"), root=root, video_backend="pyav")
                for episode in range(source.meta.total_episodes):
                    start = int(source.episode_data_index["from"][episode])
                    end = int(source.episode_data_index["to"][episode])
                    if source_kind == "hil":
                        annotation = json.loads(
                            (root / f"meta/dagger/episode_{episode:06d}.json").read_text(encoding="utf-8")
                        )
                        if not annotation["committed"]:
                            raise ValueError(f"Uncommitted episode {episode} in {root}")
                        if annotation["outcome"] == "aborted" and not args.include_aborted:
                            skipped += 1
                            continue
                        # Read scalar columns directly, avoiding video decoding for non-expert frames.
                        flags = [
                            bool(np.asarray(value).item())
                            for value in source.hf_dataset[start:end]["complementary_info.intervention"]
                        ]
                        spans = human_segments(flags)
                    else:
                        spans = [(0, end - start)]
                    for first, last in spans:
                        if last - first < args.min_frames:
                            skipped += 1
                            continue
                        if destination is None:
                            destination = create_dataset(cfg, output, args.repo_id, selected_features)
                        for index in range(start + first, start + last):
                            row = source[index]
                            frame = {
                                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                                "action": np.asarray(row["action"], dtype=np.float32),
                                "task": source.meta.tasks[int(row["task_index"])],
                            }
                            for camera in CAMERAS:
                                key = f"observation.images.{camera}"
                                frame[key] = image_uint8(row[key])
                            destination.add_frame(frame)
                        destination.save_episode()
                        provenance.append(
                            {
                                "episode_index": destination.meta.total_episodes - 1,
                                "source": str(root),
                                "source_kind": source_kind,
                                "source_episode": episode,
                                "source_frames": [first, last],
                            }
                        )
                        atomic_json(output / "meta/dagger_training_sources.json", provenance)
        if destination is None:
            raise ValueError("No eligible expert segments; check inputs and --min-frames")
        print(f"Training dataset saved to {output}; episodes={len(provenance)}, skipped_spans={skipped}")
    finally:
        if destination is not None:
            destination.stop_image_writer()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "mock.yaml"))
    parser.add_argument("--demo", action="append", default=[], help="Expert demonstration dataset; repeatable")
    parser.add_argument("--hil", action="append", default=[], help="Collector output dataset; repeatable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="local/ur5_dagger")
    parser.add_argument("--min-frames", type=int, default=10, help="Use at least the model's action horizon")
    parser.add_argument("--include-aborted", action="store_true")
    args = parser.parse_args()
    if not args.demo and not args.hil:
        parser.error("Provide --demo and/or --hil")
    if args.min_frames < 2:
        parser.error("--min-frames must be >=2")
    prepare(args)


if __name__ == "__main__":
    main()
