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
git clone https://github.com/HongyouZhou/sam2.git
git clone https://github.com/HongyouZhou/BNDL.git
git clone https://github.com/HongyouZhou/ruac.git
pip install -e sam2 && pip install -e BNDL && pip install -e ruac
export PYTHONPATH=$PWD/sam2:$PWD/BNDL:$PWD/BNDL/BNDL_upload:$PYTHONPATH
```

Tested with Python 3.12 and PyTorch 2.x (CUDA 12).

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
└── configs/sam2_ruac_train.yaml
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
