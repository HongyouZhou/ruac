# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RUAC post-refactor smoke test.

Verifies that the layered refactor (core / adapters / modeling) preserves the
end-to-end behaviour:

    1. Public API surface still imports (core + adapters + modeling.aue).
    2. All four loss classes construct.
    3. ``BNDLLoss`` runs forward + backward on synthetic Weibull outputs and
       produces non-trivial gradients on its leaf tensors.
    4. ``AdversarialPipeline`` runs ``generate_adversarial_samples`` with mock
       attackers and a fake backbone callable; output dict matches the contract.
    5. Hydra ``_target_`` paths in ``configs/sam2_ruac_train.yaml`` resolve to
       the expected modules under ``ruac.core.losses.*`` and
       ``ruac.adapters.sam2.trainer.SAM2RUACTrain``.

Run from project root:

    PYTHONPATH=$PWD:$PWD/docs/ruac:$PWD/sam2:$PWD/BNDL:$PWD/BNDL/BNDL_upload \\
        python docs/ruac/tests/smoke.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import Any

import torch
import yaml


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _fail(msg: str, exc: BaseException) -> None:
    print(f"  [FAIL] {msg}: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)


def step_1_imports() -> None:
    print("[1/5] Import surface")
    try:
        from ruac.adapters.sam2 import AUEModule, BayesianMaskDecoder, SAM2RUACTrain
        from ruac.core.attackers import AdversarialAttacker
        from ruac.core.bndl import BNDLOutputs, pixel_weibull_to_entropy_uncertainty
        from ruac.core.grl import GRL
        from ruac.core.losses import (
            CORE_LOSS_KEY,
            AUELoss,
            BNDLLoss,
            CombinedSAMBNDLLoss,
        )
        from ruac.core.pipeline import AdversarialPipeline
        from ruac.modeling.aue import AUEConfig
    except Exception as exc:  # noqa: BLE001
        _fail("import chain", exc)
        return
    # Exercise each import so ruff doesn't auto-remove them on the next pass.
    assert callable(pixel_weibull_to_entropy_uncertainty)
    assert hasattr(BNDLOutputs, "__dataclass_fields__")
    assert isinstance(GRL(), torch.nn.Module)
    for cls in (AUELoss, BNDLLoss, CombinedSAMBNDLLoss):
        assert callable(cls)
    _ok(f"AUEModule          -> {AUEModule.__module__}")
    _ok(f"SAM2RUACTrain      -> {SAM2RUACTrain.__module__}")
    _ok(f"BayesianMaskDecoder-> {BayesianMaskDecoder.__module__}")
    _ok(f"AdversarialPipeline-> {AdversarialPipeline.__module__}")
    _ok(f"AdversarialAttacker-> {AdversarialAttacker.__module__}")
    _ok(f"AUEConfig          -> {AUEConfig.__module__}")
    _ok(f"CORE_LOSS_KEY      = {CORE_LOSS_KEY!r}")
    _ok("BNDLOutputs / pixel_weibull_to_entropy_uncertainty / GRL / 3 loss classes all importable")


def step_2_loss_construction() -> None:
    print("[2/5] Loss class construction")
    from ruac.core.losses import AUELoss, BNDLLoss, CombinedSAMBNDLLoss

    try:
        bndl = BNDLLoss(kl_weight=1e-11)
        aue = AUELoss(task_weight=1.0, bndl_weight=0.05, cal_weight=0.1)
        combined = CombinedSAMBNDLLoss(sam_loss=None, bndl_loss=bndl, aue_loss=aue)
    except Exception as exc:  # noqa: BLE001
        _fail("loss construction", exc)
        return
    _ok(f"BNDLLoss / AUELoss / CombinedSAMBNDLLoss constructed (combined has {sum(1 for _ in combined.parameters())} params)")


def step_3_bndl_backward() -> None:
    print("[3/5] BNDLLoss forward + backward")
    from ruac.core.losses import CORE_LOSS_KEY, BNDLLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W, C = 2, 8, 8, 32

    wei_lambda = torch.rand(B, H, W, C, device=device, requires_grad=True)
    inv_k = (torch.rand(B, H, W, 1, device=device) * 0.5 + 0.5).detach().requires_grad_(True)
    fake_bndl: dict[str, Any] = {
        "wei_lambda": wei_lambda,
        "inv_k": inv_k,
    }
    fake_outs = [{"multistep_aux_outputs": [{"bndl": fake_bndl}]}]
    fake_targets = torch.rand(B, 1, H, W, device=device)

    loss_fn = BNDLLoss(kl_weight=1.0).to(device)
    try:
        result = loss_fn(fake_outs, fake_targets)
        core = result[CORE_LOSS_KEY]
        if not torch.isfinite(core):
            raise RuntimeError(f"BNDLLoss returned non-finite core_loss: {core}")
        core.backward()
    except Exception as exc:  # noqa: BLE001
        _fail("BNDLLoss forward/backward", exc)
        return
    g_lambda = wei_lambda.grad
    g_inv_k = inv_k.grad
    if g_lambda is None or g_inv_k is None:
        _fail("gradient absent on leaf tensor", RuntimeError("None grad"))
        return
    _ok(f"core_loss = {core.item():.4e}, wei_lambda.grad |norm|={g_lambda.norm().item():.4e}, inv_k.grad |norm|={g_inv_k.norm().item():.4e}")


def step_4_pipeline_with_mock_attacker() -> None:
    print("[4/5] AUEPipeline with mock attackers + GRL gradient")
    import torch.nn as nn

    from ruac.core.grl import GRL
    from ruac.core.pipeline import AdversarialPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H_feat, W_feat = 1, 256, 4, 4
    H_img, W_img = 16, 16

    class MockStyleAttacker(nn.Module):
        mode = "image_level"
        aug_type = "style"

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(3))  # GRL-routed param
            self.grl = GRL()

        def predict_params(self, *, clean_features, pixel_gt, model, img_batch, **kw):
            # GRL routes gradient back to self.weight with flipped sign.
            return self.grl(self.weight).view(1, 1, 3).expand(pixel_gt.shape[0], pixel_gt.shape[1], 3)

        def apply_transform(self, *, img_batch, params, pixel_gt, model, **kw):
            # Add the small per-object shift to the image (tiny scalar) so gradient flows.
            shift = params.mean(dim=(1, 2)).view(-1, 1, 1, 1)
            return img_batch + shift

    def fake_backbone(img, **kw):
        # Trivial backbone: pool to stride-4 then broadcast singleton channel up to C.
        feats = torch.nn.functional.adaptive_avg_pool2d(img, output_size=(H_feat, W_feat))  # [B, 3, h, w]
        feats = feats.mean(dim=1, keepdim=True).expand(-1, C, -1, -1)  # [B, C, h, w], grad-preserving
        return {"backbone_fpn": [feats, feats, feats]}

    attacker = MockStyleAttacker().to(device)
    pipeline = AdversarialPipeline(
        attackers={"style": attacker},
        attack_order=["style"],
        backbone_fn=fake_backbone,
        use_high_res_features=False,
        enable_background=False,
        attack_context=None,
    )

    img = torch.rand(B, 3, H_img, W_img, device=device, requires_grad=True)
    feats = torch.rand(B, C, H_feat, W_feat, device=device)
    pixel_gt = (torch.rand(B, 2, H_img, W_img, device=device) > 0.5).float()

    try:
        out = pipeline.generate_adversarial_samples(
            img_batch=img,
            backbone_features=feats,
            high_res_features=[feats, feats],
            pixel_gt=pixel_gt,
        )
    except Exception as exc:  # noqa: BLE001
        _fail("pipeline forward", exc)
        return

    expected = {"adv_img", "adv_features", "adv_high_res", "adv_pixel_gt", "adv_single_obj_gt", "adv_images_for_attacker", "deform_offsets", "vis_refs"}
    if set(out.keys()) != expected:
        _fail("pipeline output keys mismatch", RuntimeError(f"got {set(out.keys())}, want {expected}"))
        return

    # Drive a backward through pipeline to verify GRL flips attacker grad.
    fake_loss = out["adv_features"].sum() + out["adv_img"].sum()
    try:
        fake_loss.backward()
    except Exception as exc:  # noqa: BLE001
        _fail("pipeline backward", exc)
        return
    if attacker.weight.grad is None:
        _fail("attacker weight has no gradient — GRL did not propagate", RuntimeError("None grad"))
        return
    _ok(f"pipeline forward+backward OK; attacker.weight.grad |norm|={attacker.weight.grad.norm().item():.4e} (GRL routed)")


def step_5_hydra_targets() -> None:
    print("[5/5] Hydra _target_ resolution from yaml")
    yaml_path = "docs/ruac/ruac/configs/sam2_ruac_train.yaml"
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        _fail(f"yaml load {yaml_path}", exc)
        return

    targets = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "_target_" and isinstance(v, str):
                    targets.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cfg)

    expected_under_ruac = {t for t in targets if t.startswith("ruac.")}
    if not expected_under_ruac:
        _fail("no ruac._target_ entries found", RuntimeError("expected at least 1"))
        return

    bad = []
    for t in sorted(expected_under_ruac):
        mod_path, _, attr = t.rpartition(".")
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, attr)
        except Exception as exc:  # noqa: BLE001
            bad.append((t, type(exc).__name__, str(exc)))
            continue
        _ok(f"{t}")

    if bad:
        for t, etype, emsg in bad:
            print(f"  [FAIL] {t}: {etype}: {emsg}")
        sys.exit(1)


def main() -> None:
    print("=" * 70)
    print("RUAC refactor smoke test")
    print("=" * 70)
    step_1_imports()
    step_2_loss_construction()
    step_3_bndl_backward()
    step_4_pipeline_with_mock_attacker()
    step_5_hydra_targets()
    print("=" * 70)
    print("ALL STEPS PASSED")


if __name__ == "__main__":
    main()
