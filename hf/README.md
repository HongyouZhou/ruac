---
license: apache-2.0
library_name: ruac
pipeline_tag: image-segmentation
tags:
  - sam2
  - image-segmentation
  - uncertainty-estimation
  - domain-generalization
  - pytorch
base_model: facebook/sam2.1-hiera-base-plus
paper: https://arxiv.org/abs/2605.10603
code: https://github.com/HongyouZhou/ruac
---

# RUAC for SAM 2.1 Hiera Base Plus

Official checkpoint for **Segment Anything with Robust Uncertainty-Accuracy
Correlation** (ICML 2026). RUAC adds a Weibull Bayesian mask decoder and an
adversarial uncertainty-estimation training branch to SAM 2.1. It produces
prompted segmentation masks together with per-pixel uncertainty maps.

## Model details

- Base model: SAM 2.1 Hiera Base Plus
- Training data: MOSE training split
- Evaluation: zero-shot across 23 out-of-domain datasets
- Default uncertainty estimator: Bernoulli entropy from 20 Weibull Monte Carlo samples
- Weight format: safetensors
- Original training checkpoint SHA-256:
  `4f43fd1bd6ed421dbdb6a4d57b68dabddb0ccb69f8c95de089b859922e299561`
- Released `model.safetensors`: 330,844,812 bytes, SHA-256
  `678dae09c6834504d20e825793d8270f83dbc1c3c2d542ce843247d7154b46e4`

The value 20 is the released default and main setting. Other Monte Carlo sample
counts belong to the parameter-sensitivity study and are not release defaults.

## Installation

```bash
git clone --branch version/019 https://github.com/HongyouZhou/sam2.git
git clone https://github.com/HongyouZhou/BNDL.git
git clone https://github.com/HongyouZhou/ruac.git

python -m pip install -e sam2
python -m pip install -e './ruac[hub]'
export PYTHONPATH=$PWD:$PWD/sam2:$PWD/BNDL:$PWD/BNDL/BNDL_upload:$PYTHONPATH
```

## Load the model

```python
from ruac.hub import load_ruac_predictor

predictor = load_ruac_predictor(
    repo_id="HongyouZhou/ruac-sam2.1-hiera-bplus",
    device="cuda",
    mc_samples=20,
)
```

RUAC is not registered with Transformers `AutoModel`; the loader above builds
the custom Bayesian SAM2 decoder and validates the released state dict.

## Point-prompted image inference

```python
import numpy as np
from PIL import Image

image = np.array(Image.open("image.jpg").convert("RGB"), copy=True)
predictor.set_image(image)
masks, predicted_iou, _ = predictor.predict(
    point_coords=np.array([[640, 360]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int32),
    multimask_output=True,
    return_logits=True,
)

selected = int(np.argmax(predicted_iou))
mask = masks[selected] > 0
aux = predictor.get_last_aux_outputs()
uncertainty = aux["bndl"]["pixel_uncertainty_sampling"][0, :, :, selected]
```

See the complete [model loading guide](https://github.com/HongyouZhou/ruac/blob/main/docs/model_loading.md)
and [runnable inference example](https://github.com/HongyouZhou/ruac/blob/main/examples/image_inference.py).

## Limitations

RUAC follows SAM2's prompt-based segmentation interface. It does not perform
semantic classification, and uncertainty quality can vary with the domain,
prompt choice, and Monte Carlo sample count. The released model was trained on
MOSE and should be evaluated before deployment in safety-critical settings.

## Licenses

The released weights are derived from the Apache-2.0-licensed SAM 2.1
checkpoint and are distributed under Apache 2.0. The RUAC implementation is
MIT licensed. Users must also follow the licenses of the SAM2 and BNDL
dependencies.

## Citation

```bibtex
@inproceedings{ruac2026,
  title         = {Segment Anything with Robust Uncertainty-Accuracy Correlation},
  author        = {Zhou, Hongyou and Toussaint, Marc and Shao, Ling and Ye, Zihan},
  booktitle     = {Proceedings of the 43rd International Conference on Machine Learning},
  year          = {2026},
  eprint        = {2605.10603},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
