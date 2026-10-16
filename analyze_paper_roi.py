#!/usr/bin/env python3
"""Reproduce the manuscript ROI comparison from per-image Grad-CAM data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel


MODEL_FILES = {
    "FaceBased": "SE-FC-L3_FaceBased_R16.pth",
    "ObjectBased": "SE-FC-L3_ObjectBased_R16.pth",
}
ROI_COLUMNS = ["Eyes_PoS", "Nose_PoS", "Mouth_PoS"]


def pooled_cohens_d(first: np.ndarray, second: np.ndarray) -> float:
    pooled_variance = (
        (len(first) - 1) * np.var(first, ddof=1)
        + (len(second) - 1) * np.var(second, ddof=1)
    ) / (len(first) + len(second) - 2)
    return float((np.mean(first) - np.mean(second)) / np.sqrt(pooled_variance))


def main() -> None:
    repository_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root / "Results" / "GradCAM_Results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "Results",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"ModelFile", "Condition", "ImageID", *ROI_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(data) != 6048 or data["ModelFile"].nunique() != 72:
        raise ValueError(
            "Expected 6048 rows from 72 models, four conditions, and 21 images"
        )
    group_sizes = data.groupby(["ModelFile", "Condition"]).size()
    if not group_sizes.eq(21).all():
        raise ValueError("Every model-condition group must contain exactly 21 rows")
    if data.duplicated(["ModelFile", "Condition", "ImageID"]).any():
        raise ValueError("Duplicate model-condition-image rows detected")
    if not data[ROI_COLUMNS].apply(lambda column: column.between(0, 1).all()).all():
        raise ValueError("ROI saliency proportions must be between 0 and 1")

    selected = data[
        (data["Condition"] == "Full")
        & data["ModelFile"].isin(MODEL_FILES.values())
    ].copy()
    if len(selected) != 42:
        raise ValueError(f"Expected 42 selected rows, found {len(selected)}")

    selected["Pretraining"] = selected["ModelFile"].map(
        {value: key for key, value in MODEL_FILES.items()}
    )
    selected["Diagnostic_Ratio"] = (
        selected["Eyes_PoS"] + selected["Mouth_PoS"]
    ) / selected[ROI_COLUMNS].sum(axis=1)

    by_group = {
        label: selected[selected["Pretraining"] == label]
        .sort_values("ImageID")
        .reset_index(drop=True)
        for label in MODEL_FILES
    }
    for label, group in by_group.items():
        if len(group) != 21 or group["ImageID"].nunique() != 21:
            raise ValueError(f"{label}: expected 21 unique full-face images")
    if not by_group["FaceBased"]["ImageID"].equals(
        by_group["ObjectBased"]["ImageID"]
    ):
        raise ValueError("The two groups must contain the same ImageID values")

    measures = ["Diagnostic_Ratio", *ROI_COLUMNS]
    summary_rows = []
    for label, group in by_group.items():
        for measure in measures:
            values = group[measure].to_numpy(float)
            summary_rows.append(
                {
                    "Pretraining": label,
                    "Measure": measure,
                    "N": len(values),
                    "Mean": float(np.mean(values)),
                    "SD": float(np.std(values, ddof=1)),
                }
            )

    statistic_rows = []
    for measure in measures:
        face = by_group["FaceBased"][measure].to_numpy(float)
        object_based = by_group["ObjectBased"][measure].to_numpy(float)
        test = ttest_ind(face, object_based, equal_var=False)
        statistic_rows.append(
            {
                "Measure": measure,
                "Contrast": "FaceBased - ObjectBased",
                "Test": "Welch two-sample t-test",
                "t": float(test.statistic),
                "df": float(test.df),
                "p_TwoSided": float(test.pvalue),
                "Cohens_d_PooledSD": pooled_cohens_d(face, object_based),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output / "Paper_ROI_Intermediate_Data.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        args.output / "Paper_ROI_Summary.csv", index=False
    )
    pd.DataFrame(statistic_rows).to_csv(
        args.output / "Paper_ROI_Statistics.csv", index=False
    )

    target_model = data[data["ModelFile"] == MODEL_FILES["FaceBased"]].copy()
    full = target_model[target_model["Condition"] == "Full"].sort_values("ImageID")
    occlusion_rows = []
    for condition in ["E", "M", "N"]:
        occluded = target_model[target_model["Condition"] == condition].sort_values(
            "ImageID"
        )
        if len(full) != 21 or len(occluded) != 21:
            raise ValueError(f"{condition}: expected 21 full and 21 occluded images")
        if not full["ImageID"].reset_index(drop=True).equals(
            occluded["ImageID"].reset_index(drop=True)
        ):
            raise ValueError(f"{condition}: ImageID values do not match the full condition")
        for measure in ROI_COLUMNS:
            full_values = full[measure].to_numpy(float)
            occluded_values = occluded[measure].to_numpy(float)
            difference = full_values - occluded_values
            test = ttest_rel(full_values, occluded_values)
            occlusion_rows.append(
                {
                    "OcclusionCondition": condition,
                    "Measure": measure,
                    "N_Pairs": len(difference),
                    "Full_Mean": float(np.mean(full_values)),
                    "Occluded_Mean": float(np.mean(occluded_values)),
                    "MeanDifference_FullMinusOccluded": float(np.mean(difference)),
                    "t": float(test.statistic),
                    "df": len(difference) - 1,
                    "p_TwoSided": float(test.pvalue),
                    "Cohens_dz": float(
                        np.mean(difference) / np.std(difference, ddof=1)
                    ),
                }
            )
    pd.DataFrame(occlusion_rows).to_csv(
        args.output / "Paper_ROI_Occlusion_Statistics.csv", index=False
    )
    print("Validated 6048 rows and 21 selected images per pretraining condition.")
    print(f"Wrote manuscript ROI tables to {args.output}")


if __name__ == "__main__":
    main()
