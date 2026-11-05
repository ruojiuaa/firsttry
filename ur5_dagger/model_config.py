"""UR5 pi0.5 transforms and configuration, registered only in this process."""

import dataclasses
from pathlib import Path

import numpy as np
from prepare_training import image_uint8
from settings import CAMERAS
from settings import CONTRACT

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config
from openpi.training import weight_loaders

NAME = "pi05_ur5_dagger"


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    def __call__(self, data):
        # Two global cameras occupy base/right-wrist slots; the actual wrist uses left-wrist.
        # This mapping must stay identical during every round of training and deployment.
        slots = ("base_0_rgb", "right_wrist_0_rgb", "left_wrist_0_rgb")
        result = {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "image": {
                slot: image_uint8(data[f"observation.images.{camera}"])
                for camera, slot in zip(CAMERAS, slots, strict=True)
            },
            "image_mask": dict.fromkeys(slots, np.True_),
            "prompt": data["prompt"],
        }
        if result["state"].shape != (8,):
            raise ValueError("UR5 state must have eight dimensions")
        if "actions" in data:
            result["actions"] = data["actions"]
        return result


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :7]}


@dataclasses.dataclass(frozen=True)
class UR5Data(config.DataConfigFactory):
    def create(self, assets_dirs: Path, model_config):
        mapping = {
            "observation.state": "observation.state",
            **{f"observation.images.{name}": f"observation.images.{name}" for name in CAMERAS},
            "actions": "action",
            "prompt": "prompt",
        }
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(mapping)]),
            data_transforms=transforms.Group(inputs=[UR5Inputs()], outputs=[UR5Outputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def register(args):
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=args.horizon,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if args.lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if args.lora else "gemma_300m",
    )
    result = config.TrainConfig(
        name=NAME,
        exp_name=args.exp_name,
        model=model,
        data=UR5Data(repo_id=args.repo_id),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.base_params),
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None if args.lora else 0.99,
        wandb_enabled=False,
        resume=args.resume,
        assets_base_dir=str(Path(args.assets).resolve()),
        checkpoint_base_dir=str(Path(args.checkpoints).resolve()),
        policy_metadata={"ur5_dagger_contract": CONTRACT, "fps": args.fps},
    )
    # Existing scripts consult this registry. No repository config file is changed.
    config._CONFIGS_DICT[NAME] = result
    return result
