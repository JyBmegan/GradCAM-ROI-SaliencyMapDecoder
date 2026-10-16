# Grad-CAM ROI Saliency Analysis

Grad-CAM heatmaps and region-of-interest (ROI) saliency analysis for the SE-AlexNet paper. This repository reproduces the paper's Grad-CAM results for the AlexNet models: the proportion of saliency in the eyes, nose, and mouth, and the face- vs object-pretraining comparisons.

## Reproduce

```bash
pip install -r requirements.txt
```

### ROI statistics (paper tables)

```bash
python3 analyze_paper_roi.py
```

Reads `Results/GradCAM_Results.csv`, selects the two `SE-FC-L3` reduction-16 models (face- and object-pretrained) on the 21 full-face images, and writes:

| Output | Content |
| --- | --- |
| `Results/Paper_ROI_Intermediate_Data.csv` | Per-image ROI proportions and diagnostic ratio (42 images) |
| `Results/Paper_ROI_Summary.csv` | N, mean, and SD per measure |
| `Results/Paper_ROI_Statistics.csv` | Face vs object Welch t-test and pooled-SD Cohen's d |
| `Results/Paper_ROI_Occlusion_Statistics.csv` | Full vs occluded paired comparisons for the face-pretrained model |

The diagnostic ratio is `(eyes + mouth) / (eyes + nose + mouth)`.

### Heatmaps and full ROI data

This step needs the model weights. Download them into `All_Models_Collection/` (filenames must match the `ModelFile` column), then run:

```bash
python3 universal_GradCAM_pipeline.py
```

This recomputes the Grad-CAM heatmaps, per-ROI saliency, and trend plots for every model, mask condition, and image, and rewrites `Results/GradCAM_Results.csv`.

## Inputs

| Path | Content |
| --- | --- |
| `Results/GradCAM_Results.csv` | 72 models × 4 mask conditions × 21 images (6048 rows) |
| `input_images/` (`Full`, `E`, `N`, `M`) | Full and eye/nose/mouth-occluded faces |
| `All_Models_Collection/` | Model weights from [Hugging Face](https://huggingface.co/JiayuMBao/SE-AlexNet) |

## Formulas

The Grad-CAM localization map for class $c$:

$$
L_{Grad-CAM}^{c}=\mathrm{ReLU}\Big(\sum_k \alpha_k^c A^k\Big),
\qquad
\alpha_k^c=\frac1Z\sum_i\sum_j\frac{\partial y^c}{\partial A_{ij}^k}
$$

The proportion of saliency in region $R$:

$$
PoS_R=\frac{\sum_{(x,y)\in R}L(x,y)}{\sum_{x,y}L(x,y)}
$$

The diagnostic ratio (eyes + mouth over all three ROIs):

$$
\frac{PoS_{eyes}+PoS_{mouth}}{PoS_{eyes}+PoS_{nose}+PoS_{mouth}}
$$

## Related repositories

- Model training and prediction: [SE-AlexNet](https://github.com/JyBmegan/SE-AlexNet)
- Psychometric-curve fitting (PSE): [PsychometricFittingCurve](https://github.com/JyBmegan/PsychometricFittingCurve)

## Citation

<!-- Citation to be added -->
