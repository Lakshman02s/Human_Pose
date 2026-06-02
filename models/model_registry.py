from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    default_config: str
    default_checkpoint: str


POSE_MODELS = {
    "rtmpose": ModelSpec(
        name="rtmpose",
        default_config="configs/mmpose_configs/rtmpose-m_8xb256-420e_coco-256x192.py",
        default_checkpoint="checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-74d8ae39_20230228.pth",
    ),
    "vitpose": ModelSpec(
        name="vitpose",
        default_config="configs/mmpose_configs/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py",
        default_checkpoint="checkpoints/vitpose_base_coco_aic_mpii.pth",
    ),
}


DETECTOR_MODEL = ModelSpec(
    name="rtmdet_person",
    default_config="configs/mmpose_configs/rtmdet_nano_320-8xb32_coco-person.py",
    default_checkpoint="checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
)
