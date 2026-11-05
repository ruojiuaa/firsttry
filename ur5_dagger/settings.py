"""Configuration shared by the collector, mock run and model wrappers."""

from pathlib import Path
from typing import Literal

from hardware.config import CaptureConfig
from hardware.config import StrictModel
from pydantic import Field
from pydantic import model_validator
import yaml

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CAMERAS = ("agentview_1", "agentview_2", "eye_in_hand")
CONTRACT = "ur5_tcp_base_delta_7d_state8_rgb3_v1"


class PolicyConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    execution_horizon: int = Field(5, ge=1)
    timeout_s: float = Field(5.0, gt=0)
    max_result_age_s: float = Field(0.5, gt=0)
    mock_delay_s: float = Field(0.05, ge=0)


class DaggerConfig(StrictModel):
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    start_mode: Literal["human", "policy"] = "human"
    takeover_on_motion: bool = True
    takeover_threshold: float = Field(0.25, gt=0, le=1)
    max_translation_speed: float = Field(0.05, gt=0, le=0.8)
    max_rotation_speed: float = Field(0.15, gt=0, le=2)
    video: bool = True


class Settings(CaptureConfig):
    dagger: DaggerConfig = Field(default_factory=DaggerConfig)

    @model_validator(mode="after")
    def validate_dagger(self):
        if tuple(c.name for c in self.cameras) != CAMERAS:
            raise ValueError(f"Camera names/order must be {CAMERAS}")
        if self.teleop.frame != "base":
            raise ValueError("This model contract requires base-frame deltas")
        if self.mode == "real" and self.gripper.feedback_plugin != "hardware.rg2:RTDEGripper":
            raise ValueError("Real mode uses the bundled RG2 RTDE register driver")
        if self.mode == "real":
            from hardware.rg2 import RTDEConfig

            rg2 = RTDEConfig.model_validate(self.gripper.feedback_kwargs)
            if rg2.robot_ip != self.robot.ip:
                raise ValueError("Robot and RG2 RTDE IP addresses must match")
        return self


def load_settings(path: str | Path) -> Settings:
    cfg = Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    root = Path(cfg.dataset.root).expanduser()
    cfg.dataset.root = str((ROOT / root).resolve())
    return cfg
