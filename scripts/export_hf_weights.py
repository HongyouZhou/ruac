#!/usr/bin/env python3
"""Export the model tensors from a SAM2 training checkpoint to safetensors."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="SAM2/RUAC training checkpoint (.pt)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.safetensors"),
        help="Output path (default: model.safetensors)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Default MC sample count recorded in metadata (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.mc_samples < 1:
        raise ValueError("--mc-samples must be positive")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state_dict, Mapping) or not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in state_dict.items()):
        raise ValueError("Checkpoint does not contain a `model` tensor state dict")

    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}
    source_digest = sha256(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output_path),
        metadata={
            "architecture": "SAM2RUACTrain",
            "base_model": "facebook/sam2.1-hiera-base-plus",
            "default_mc_samples": str(args.mc_samples),
            "format": "pt",
            "source_checkpoint_sha256": source_digest,
        },
    )

    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"Exported {len(tensors)} tensors ({tensor_bytes:,} tensor bytes)")
    print(f"Source SHA-256: {source_digest}")
    print(f"Output: {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"Output SHA-256: {sha256(output_path)}")


if __name__ == "__main__":
    main()
