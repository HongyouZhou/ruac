#!/usr/bin/env python3
"""Run point-prompted RUAC inference and save a mask and uncertainty map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ruac.hub import DEFAULT_REPO_ID, load_ruac_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Local .safetensors or .pt checkpoint")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default=None, help="For example cuda, cuda:1, or cpu")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "LABEL"),
        help="Point prompt; LABEL is 1 (foreground) or 0 (background). Repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ruac_output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = np.array(Image.open(args.image).convert("RGB"), copy=True)
    height, width = image.shape[:2]
    prompts = args.point or [[width / 2.0, height / 2.0, 1.0]]
    point_coords = np.asarray([[x, y] for x, y, _ in prompts], dtype=np.float32)
    point_labels = np.asarray([int(label) for _, _, label in prompts], dtype=np.int32)
    if not np.isin(point_labels, [0, 1]).all():
        raise ValueError("Point labels must be 0 (background) or 1 (foreground)")

    predictor = load_ruac_predictor(
        args.checkpoint,
        repo_id=args.repo_id,
        device=args.device,
        mc_samples=args.mc_samples,
    )
    device_type = predictor.device.type
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ),
    ):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True,
        )

    selected_index = int(np.argmax(scores))
    binary_mask = masks[selected_index] > 0.0
    aux_outputs = predictor.get_last_aux_outputs()
    try:
        uncertainty_candidates = aux_outputs["bndl"]["pixel_uncertainty_sampling"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("RUAC inference did not return a BNDL uncertainty map") from exc
    if uncertainty_candidates.shape[-1] != masks.shape[0]:
        raise RuntimeError(f"RUAC uncertainty channels are not aligned with the returned masks: {uncertainty_candidates.shape[-1]} != {masks.shape[0]}")
    uncertainty_low_res = uncertainty_candidates[:, :, selected_index]
    uncertainty = (
        F.interpolate(
            uncertainty_low_res.float()[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .cpu()
        .numpy()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask.astype(np.uint8) * 255).save(args.output_dir / "mask.png")
    np.save(args.output_dir / "uncertainty.npy", uncertainty)
    uncertainty_png = np.clip(uncertainty / np.log(2.0), 0.0, 1.0)
    Image.fromarray((uncertainty_png * 255).astype(np.uint8)).save(args.output_dir / "uncertainty.png")
    print(f"Selected mask {selected_index} with predicted IoU {float(scores[selected_index]):.4f}")
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
