"""Load released RUAC checkpoints from disk or the Hugging Face Hub."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

DEFAULT_REPO_ID = "HongyouZhou/ruac-sam2.1-hiera-bplus"
DEFAULT_WEIGHTS_FILENAME = "model.safetensors"
DEFAULT_MC_SAMPLES = 20

_INFERENCE_UNUSED_PREFIXES = (
    "style_gcn.",
    "style_attacker.",
    "deform_attacker.",
)


def _download_weights(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Loading RUAC from the Hugging Face Hub requires `pip install 'ruac[hub]'`.") from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading safetensors weights requires `pip install 'ruac[hub]'`.") from exc
        state_dict: Any = load_file(str(path), device="cpu")
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions predating the weights_only argument.
            payload = torch.load(path, map_location="cpu")  # noqa: S614
        state_dict = payload.get("model", payload) if isinstance(payload, Mapping) else payload

    if not isinstance(state_dict, Mapping) or not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in state_dict.items()):
        raise ValueError(f"{path} does not contain a tensor state dict or a checkpoint with a `model` state dict.")
    return state_dict


def load_ruac_model(
    checkpoint_path: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    device: str | torch.device | None = None,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    include_attackers: bool = False,
) -> torch.nn.Module:
    """Build RUAC-SAM2.1-B+ and load released weights.

    Args:
        checkpoint_path: Local ``.safetensors`` or SAM2 training ``.pt`` file.
            When omitted, ``filename`` is downloaded from ``repo_id``.
        repo_id: Hugging Face model repository used when no local path is given.
        filename: Weight filename in the Hugging Face repository.
        device: Destination device. Defaults to CUDA when available, otherwise CPU.
        mc_samples: Monte Carlo samples used for the uncertainty map. The released
            default and main-paper setting is 20.
        include_attackers: Construct and load the training-only style/deformation
            attackers. Leave disabled for normal inference.

    Returns:
        An eval-mode ``SAM2RUACTrain`` model ready for ``SAM2ImagePredictor``.
    """
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be positive, got {mc_samples}")

    weights_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else _download_weights(repo_id, filename)
    if not weights_path.is_file():
        raise FileNotFoundError(f"RUAC checkpoint not found: {weights_path}")

    config_resource = files("ruac.configs").joinpath("sam2_ruac_train.yaml")
    with as_file(config_resource) as config_path:
        config = OmegaConf.load(config_path)
        config.trainer.model.sam_mask_decoder_extra_args.bndl_sample_num = mc_samples
        config.trainer.model.aue_config.enabled = include_attackers
        config.trainer.model.freeze_image_encoder_epochs = 0
        model = instantiate(config.trainer.model)

    state_dict = _load_state_dict(weights_path)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        raise RuntimeError("Checkpoint is missing required RUAC parameters: " + ", ".join(incompatible.missing_keys))

    unexpected = list(incompatible.unexpected_keys)
    if include_attackers:
        disallowed_unexpected = unexpected
    else:
        disallowed_unexpected = [name for name in unexpected if not name.startswith(_INFERENCE_UNUSED_PREFIXES)]
    if disallowed_unexpected:
        raise RuntimeError("Checkpoint contains parameters incompatible with this RUAC build: " + ", ".join(disallowed_unexpected))
    if unexpected:
        logging.info(
            "Ignored %d training-only attacker tensors during inference loading.",
            len(unexpected),
        )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(resolved_device).eval()


def load_ruac_predictor(
    checkpoint_path: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    device: str | torch.device | None = None,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    mask_threshold: float = 0.0,
):
    """Load RUAC and wrap it in the SAM2 image-prediction API."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = load_ruac_model(
        checkpoint_path,
        repo_id=repo_id,
        filename=filename,
        device=device,
        mc_samples=mc_samples,
        include_attackers=False,
    )
    return SAM2ImagePredictor(model, mask_threshold=mask_threshold)


__all__ = [
    "DEFAULT_MC_SAMPLES",
    "DEFAULT_REPO_ID",
    "DEFAULT_WEIGHTS_FILENAME",
    "load_ruac_model",
    "load_ruac_predictor",
]
