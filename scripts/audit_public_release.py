#!/usr/bin/env python3
"""Fail fast when non-public or stale material enters the release tree."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.name

POLICIES = {
    "SE-AlexNet": {
        "required": [
            "README.md",
            "requirements.txt",
            "scripts/validate_prediction_archive.py",
            "2_Results/1_Predict21Datapoints",
        ],
        "forbidden_roots": [
            "2_Results/2_GradCAMHeatMap",
            "2_Results/3_GradCAMDatapoints",
            "3_Ablation(Codes)",
            "4_Ablation(Results)",
            "5_FittingCurveAnalysis",
        ],
    },
    "PsychometricFittingCurve": {
        "required": [
            "README.md",
            "requirements.txt",
            "Code/reproduce_psychometric_results.py",
            "Code/Human_PSE_40_Subjects.csv",
            "Data/human-data-9point.csv",
            "Results/Model_Psychometric_Results.csv",
        ],
        "forbidden_roots": ["Ref_Results", "Data/Human_Fitting_Parameters.xlsx"],
    },
    "GradCAM-ROI-SaliencyMapDecoder": {
        "required": [
            "README.md",
            "requirements.txt",
            "analyze_paper_roi.py",
            "Results/Paper_GradCAM_Input.csv",
            "Results/Paper_ROI_Intermediate_Data.csv",
            "Results/Paper_ROI_Summary.csv",
            "Results/Paper_ROI_Statistics.csv",
            "Results/Paper_ROI_Occlusion_Statistics.csv",
        ],
        "forbidden_roots": [".vscode"],
    },
}

FORBIDDEN_SUFFIXES = {
    ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx", ".pdf",
    ".fig", ".pth", ".pt", ".ckpt", ".sav", ".mat",
}
STALE_TOKEN_RE = re.compile(
    r"(^|[ _./-])(copy|backup|old|draft|temp|tmp|legacy)(?=$|[ _./-])",
    re.IGNORECASE,
)
LOCAL_PATH_RE = re.compile(r"(?:^|[\s'\"(])/(?:Users|home|Volumes)/|[A-Za-z]:\\")
SECRET_RE = re.compile(
    r"(?:BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY|"
    r"(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{16,})",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".html", ".js", ".css", ".sh",
}


def public_candidates() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item for item in result.stdout.decode().split("\0") if item]


def main() -> int:
    policy = POLICIES.get(REPO)
    if policy is None:
        print(f"Unsupported repository: {REPO}", file=sys.stderr)
        return 2

    problems: list[str] = []
    candidates = public_candidates()
    relative = [path.relative_to(ROOT).as_posix() for path in candidates]

    for required in policy["required"]:
        if not (ROOT / required).exists():
            problems.append(f"missing required release input: {required}")

    for path, rel in zip(candidates, relative):
        parts = Path(rel).parts
        if any(rel == root or rel.startswith(root + "/") for root in policy["forbidden_roots"]):
            problems.append(f"forbidden release path: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"non-public file type: {rel}")
        if STALE_TOKEN_RE.search(rel):
            problems.append(f"stale/backup-like filename: {rel}")
        if any(part.lower() in {"affect_human", "raw_human", "human_raw", "trial_level"} for part in parts):
            problems.append(f"raw human-data-like path: {rel}")
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            problems.append(f"index points to missing file: {rel}")
            continue
        if size > 95 * 1024 * 1024:
            problems.append(f"oversized release file (>95 MiB): {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES and path.name != Path(__file__).name:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                problems.append(f"cannot read text file {rel}: {exc}")
                continue
            if LOCAL_PATH_RE.search(text):
                problems.append(f"local absolute path found: {rel}")
            if SECRET_RE.search(text):
                problems.append(f"credential/private-key pattern found: {rel}")

    if problems:
        print("Public-release audit failed:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"- {problem}", file=sys.stderr)
        return 1

    print(f"Public-release audit passed for {REPO}: {len(candidates)} release candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


