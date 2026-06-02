from __future__ import annotations

from pathlib import Path


def export_onnx_command(config: str, checkpoint: str, output_path: str, device: str = "cpu") -> str:
    return (
        "python tools/deploy.py "
        f"{config} {checkpoint} --output {output_path} --device {device} --backend onnxruntime"
    )


def export_tensorrt_command(deploy_cfg: str, model_cfg: str, checkpoint: str, output_dir: str) -> str:
    return (
        "python tools/deploy.py "
        f"{deploy_cfg} {model_cfg} {checkpoint} demo.jpg "
        f"--work-dir {Path(output_dir)} --device cuda:0 --backend tensorrt"
    )
