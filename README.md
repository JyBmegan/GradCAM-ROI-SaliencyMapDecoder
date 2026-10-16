# Grad-CAM ROI Saliency Analysis

面孔刺激、Grad-CAM 流程、逐图 ROI 显著性表，以及论文中面孔/物体预训练 SE-AlexNet 对比的统计分析。

## 复现论文 ROI 统计

```bash
pip install -r requirements.txt
python3 analyze_paper_roi.py
```

脚本读取 `Results/GradCAM_Results.csv`，选取两个 `SE-FC-L3` 压缩比 16 模型的完整面孔图像（各 21 张），计算诊断比率与各 ROI 显著性，并输出：

| 输出 | 内容 |
| --- | --- |
| `Results/Paper_ROI_Intermediate_Data.csv` | 42 张图像的 ROI 数据与诊断比率 |
| `Results/Paper_ROI_Summary.csv` | 各指标的 N、均值、SD |
| `Results/Paper_ROI_Statistics.csv` | 面孔 vs 物体的 Welch t 检验与合并 SD 的 Cohen's d |
| `Results/Paper_ROI_Occlusion_Statistics.csv` | 面孔预训练 `SE-FC-L3`-R16 模型的完整 vs 遮挡配对比较 |

## 重新计算 Grad-CAM 与全部 ROI

把模型权重放入 `All_Models_Collection/`（文件名与 `Results/GradCAM_Results.csv` 中的 `ModelFile` 一致），然后运行：

```bash
python3 universal_GradCAM_pipeline.py
```

## 输入

- `Results/GradCAM_Results.csv`：72 个模型 × 4 个刺激条件 × 21 张图像 = 6048 行。
- `input_images/Full|E|N|M`：完整与眼/鼻/口遮挡的面孔图像。
- `All_Models_Collection/`：模型权重（发布在 [Hugging Face](https://huggingface.co/JiayuMBao/SE-AlexNet)）。

## ROI 定义

图像缩放到 224 × 224 像素，坐标以左上角为原点。

| 区域 | X | Y |
| --- | ---: | ---: |
| 眼 | 52–171 | 70–99 |
| 鼻 | 52–171 | 120–149 |
| 嘴 | 52–171 | 169–199 |

## 公式

类别 $c$ 的 Grad-CAM 图：

$$
L_{Grad-CAM}^{c}=\operatorname{ReLU}\Big(\sum_k \alpha_k^c A^k\Big),
\qquad
\alpha_k^c=\frac1Z\sum_i\sum_j\frac{\partial y^c}{\partial A_{ij}^k}
$$

区域 $R$ 的显著性占比（Proportion of Saliency）：

$$
PoS_R=\frac{\sum_{(x,y)\in R}L(x,y)}{\sum_{x,y}L(x,y)}
$$

诊断比率（眼与嘴占三处 ROI 总显著性之比）：

$$
\frac{PoS_{eyes}+PoS_{mouth}}{PoS_{eyes}+PoS_{nose}+PoS_{mouth}}
$$
