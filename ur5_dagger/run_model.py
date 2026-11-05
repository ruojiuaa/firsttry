"""Run existing OpenPI stats/train/serve code with the added UR5 configuration."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stats", "train", "serve"))
    parser.add_argument("--repo-id", default="local/ur5_dagger")
    parser.add_argument("--data-home", default="ur5_dagger/output/training")
    parser.add_argument("--checkpoint", help="Trained UR5 checkpoint directory for serve")
    parser.add_argument("--base-params", default="gs://openpi-assets/checkpoints/pi05_base/params")
    parser.add_argument("--exp-name", default="round1")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--assets", default="ur5_dagger/output/assets")
    parser.add_argument("--checkpoints", default="ur5_dagger/output/checkpoints")
    args = parser.parse_args()
    if min(args.horizon, args.fps, args.batch_size, args.steps) <= 0:
        parser.error("horizon, fps, batch-size and steps must be positive")
    if args.mode == "serve" and not args.checkpoint:
        parser.error("serve requires --checkpoint pointing to your UR5 model")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["HF_LEROBOT_HOME"] = str(Path(args.data_home).resolve())
    from model_config import NAME
    from model_config import register

    cfg = register(args)
    if args.mode == "stats":
        from scripts.compute_norm_stats import main as compute_stats

        compute_stats(NAME)
    elif args.mode == "train":
        from scripts.train import main as train

        train(cfg)
    else:
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        policy = create_trained_policy(cfg, args.checkpoint)
        WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
