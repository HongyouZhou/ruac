# Loading the released RUAC model

The released model is RUAC trained from SAM 2.1 Hiera Base Plus. Segmentation
logits use the analytic Weibull expectation. Pixel uncertainty uses 20 Monte
Carlo samples by default; 20 is the main setting, while other sample counts are
reported only as sensitivity experiments.

## 1. Install

```bash
git clone --branch version/019 https://github.com/HongyouZhou/sam2.git
git clone https://github.com/HongyouZhou/BNDL.git
git clone https://github.com/HongyouZhou/ruac.git

python -m pip install -e sam2
python -m pip install -e './ruac[hub]'
export PYTHONPATH=$PWD:$PWD/sam2:$PWD/BNDL:$PWD/BNDL/BNDL_upload:$PYTHONPATH
```

The loader has been tested with Python 3.12, PyTorch 2.6, SAM2 commit
`63b921d84a295e04f2f30e850bd8880d3792e174`, and BNDL commit
`545fcb24a9b2c0ee897df593c7c274c71d0e2899`.

## 2. Load from Hugging Face

```python
from ruac.hub import load_ruac_predictor

predictor = load_ruac_predictor(
    repo_id="HongyouZhou/ruac-sam2.1-hiera-bplus",
    device="cuda",
    mc_samples=20,
)
```

This downloads `model.safetensors`, constructs the RUAC Bayesian decoder, and
omits the training-only style/deformation attackers. The loader validates that
all inference parameters are present and that any ignored tensors belong only
to those attackers.

To load the original training checkpoint or a locally downloaded safetensors
file, pass its path as the first argument:

```python
predictor = load_ruac_predictor(
    "/path/to/checkpoint.pt",  # or model.safetensors
    device="cuda:0",
    mc_samples=20,
)
```

## 3. Image inference

```python
import numpy as np
from PIL import Image

image = np.array(Image.open("image.jpg").convert("RGB"), copy=True)
predictor.set_image(image)

masks, predicted_iou, low_res_logits = predictor.predict(
    point_coords=np.array([[640, 360]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int32),
    multimask_output=True,
    return_logits=True,
)

selected = int(np.argmax(predicted_iou))
mask = masks[selected] > 0
aux = predictor.get_last_aux_outputs()
uncertainty_candidates = aux["bndl"]["pixel_uncertainty_sampling"][0]
assert uncertainty_candidates.shape[-1] == masks.shape[0]
uncertainty_low_res = uncertainty_candidates[:, :, selected]
```

`uncertainty_low_res` is a low-resolution entropy map. Resize it to the input
image size with bilinear interpolation for visualization. Entropy ranges from
0 to `ln(2)` for the Bernoulli mask prediction.

At inference, RUAC aligns the public BNDL auxiliary channels with SAM2's mask
selection: three uncertainty channels are returned for three multimask
candidates, and one channel is returned for single-mask prediction. The four
internal SAM2 mask-token channels remain available under
`aux["bndl"]["all_mask_tokens"]`. To reproduce the original evaluation's
all-hypothesis uncertainty aggregation, use:

```python
raw_uncertainty = aux["bndl"]["all_mask_tokens"][
    "pixel_uncertainty_sampling"
][0]
uncertainty_all_hypotheses = raw_uncertainty.mean(dim=-1)
```

The runnable example performs the resize and writes `mask.png`,
`uncertainty.png`, and the raw `uncertainty.npy`:

```bash
python ruac/examples/image_inference.py image.jpg \
  --point 640 360 1 \
  --output-dir output
```

## 4. Reproducibility and safety

- Default uncertainty samples: `20`.
- Base model: `facebook/sam2.1-hiera-base-plus`.
- Original training checkpoint SHA-256:
  `4f43fd1bd6ed421dbdb6a4d57b68dabddb0ccb69f8c95de089b859922e299561`.
- Released `model.safetensors` SHA-256:
  `678dae09c6834504d20e825793d8270f83dbc1c3c2d542ce843247d7154b46e4`.
- Prefer `model.safetensors` for inference. Only load `.pt` files from trusted
  sources; PyTorch checkpoints use a pickle-based container.
- The HF model is not a Transformers `AutoModel`. Use `ruac.hub` as shown
  above because RUAC depends on its custom SAM2 Bayesian decoder.

Verify a downloaded release with:

```bash
sha256sum model.safetensors
```

Maintainers can reproduce the safetensors artifact from the original training
checkpoint with:

```bash
python scripts/export_hf_weights.py /path/to/checkpoint.pt \
  --output model.safetensors \
  --mc-samples 20
```
