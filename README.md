# Grad-CAM ROI saliency analysis

This repository reproduces the manuscript's Grad-CAM region-of-interest (ROI)
tables for the SE-AlexNet models.

## Reproduce the paper tables

```bash
pip install -r requirements.txt
python3 analyze_paper_roi.py
```

The script reads `Results/Paper_GradCAM_Input.csv`, validates the 105
paper-scope rows (four conditions for the face-pretrained Location-3/R16 model
and the Full condition for the object-pretrained Location-3/R16 model), and
writes four tables to `Results/`:

| Output | Content |
| --- | --- |
| `Paper_ROI_Intermediate_Data.csv` | Per-image ROI proportions and diagnostic ratio |
| `Paper_ROI_Summary.csv` | N, mean, and SD per measure |
| `Paper_ROI_Statistics.csv` | Face- vs object-pretraining Welch t-tests and pooled-SD Cohen's d |
| `Paper_ROI_Occlusion_Statistics.csv` | Full vs occluded paired comparisons for the face-pretrained model |

The diagnostic ratio is `(eyes + mouth) / (eyes + nose + mouth)`.

## Inputs and model weights

`input_images/` contains the reference stimuli used when the saliency values
were generated. The full model checkpoints and canonical loader are available
from [Hugging Face](https://huggingface.co/JiayuMBao/SE-AlexNet). The committed
CSV is the paper-matched analysis input, so the four manuscript tables can be
reproduced without redistributing checkpoints or raw participant data.

## Public-release scope

The full 72-model archive, generated heatmap and trend-plot collections, and
legacy flat-`.pth` batch scripts are intentionally excluded because they are not
inputs to the manuscript tables or the current public loader. This keeps the
repository limited to the reproducible paper analysis rather than stale or
non-public intermediate material.

Before committing or publishing, run:

```bash
python3 scripts/audit_public_release.py
```

## Related repositories

- Model training and prediction: [SE-AlexNet](https://github.com/JyBmegan/SE-AlexNet)
- Psychometric fitting: [PsychometricFittingCurve](https://github.com/JyBmegan/PsychometricFittingCurve)

## Citation

<!-- Citation to be added -->
