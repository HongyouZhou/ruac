# RUAC: Robust Uncertainty-Accuracy Correlation for SAM2

Reference implementation for the ICML 2026 paper:

> **Segment Anything with Robust Uncertainty-Accuracy Correlation**
> Hongyou Zhou¹, Marc Toussaint¹, Ling Shao², Zihan Ye²✉
>
> ¹ Learning and Intelligent Systems, Technical University of Berlin, Germany
> ² UCAS-Terminus AI Lab, University of Chinese Academy of Sciences, China
> ✉ Corresponding author: [zihhye@outlook.com](mailto:zihhye@outlook.com)
>
> [![arXiv](https://img.shields.io/badge/arXiv-2605.10603-b31b1b.svg)](https://arxiv.org/abs/2605.10603) &nbsp; [Paper PDF](https://arxiv.org/pdf/2605.10603)

![RUAC overview](assets/banner.png)

*Vanilla SAM2 produces unexplainable masks and confused confidence maps on out-of-domain images. RUAC introduces an adversarial training branch (style + deformation perturbations) and a Bayesian mask decoder, yielding interpretable segmentation with uncertainty maps that track prediction error across 23 OOD domains.*

## Method

- **Uncertainty Estimation.** The mask decoder's hypernet pixel path is replaced by a Bayesian head with Weibull variational posteriors (BNDL, Hu et al., ICLR 2025) over image and mask tokens. Per-pixel uncertainty is computed analytically or by Monte Carlo sampling.
- **Adversarial Uncertainty Estimation.** Style (AdaIN + GCN) and deformation (DG-Font + grid sample) networks generate adversarial images via Gradient Reversal. A symmetric dual-stopgrad calibration loss keeps predicted uncertainty aligned with prediction error.

Trained jointly under a curriculum (clean → +KL → +AUE) on MOSE; evaluated zero-shot on 23 OOD domains.

## Install

```bash
git clone --branch version/019 https://github.com/HongyouZhou/sam2.git
git clone https://github.com/HongyouZhou/BNDL.git
git clone https://github.com/HongyouZhou/ruac.git
python -m pip install -e sam2
python -m pip install -e './ruac[hub]'
export PYTHONPATH=$PWD:$PWD/sam2:$PWD/BNDL:$PWD/BNDL/BNDL_upload:$PYTHONPATH
```

Tested with Python 3.12 and PyTorch 2.x (CUDA 12).

## Pretrained model

The released SAM 2.1 Hiera Base Plus checkpoint is hosted at
[`HongyouZhou/ruac-sam2.1-hiera-bplus`](https://huggingface.co/HongyouZhou/ruac-sam2.1-hiera-bplus).
Load it through RUAC's validated SAM2 adapter:

```python
from ruac.hub import load_ruac_predictor

predictor = load_ruac_predictor(device="cuda", mc_samples=20)
```

The default of 20 Monte Carlo samples is the main release setting. Other sample
counts are sensitivity experiments. RUAC is not a Transformers `AutoModel`;
the custom loader constructs the Bayesian mask decoder and omits training-only
attackers during inference.

For local checkpoints, image inference, uncertainty-map extraction, exact
dependency commits, and checksum verification, see the
**[model loading guide](docs/model_loading.md)**. A runnable example is provided
at [`examples/image_inference.py`](examples/image_inference.py).

## Train

```bash
python sam2/training/train.py --config-dir ruac/configs --config-name sam2_ruac_train
```

Edit `ruac/configs/sam2_ruac_train.yaml` to point at your MOSE data and set `${PROJECT_HOME}` before training. Hyper-parameters match the paper's Appendix table.

## Layout

```
ruac/
├── core/                model-agnostic math (zero `import sam2.*`)
│   ├── grl.py, pipeline.py
│   ├── bndl/uncertainty.py
│   ├── attackers/{base,style,deform}.py
│   └── losses/{bndl,aue,combined}.py
├── adapters/sam2/       SAM2-specific glue (all `import sam2.*` lives here)
│   ├── bayesian_decoder.py
│   ├── trainer.py
│   └── aue_module.py
├── modeling/            shared helpers (style GCN, AUE config + viz)
├── configs/sam2_ruac_train.yaml
└── hub.py               validated local/Hugging Face checkpoint loader
```

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

## License

MIT (see [`LICENSE`](LICENSE)).
