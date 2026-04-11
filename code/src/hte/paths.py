from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
CODE_ROOT = SRC_ROOT.parent
REPO_ROOT = CODE_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
MODELS_ROOT = RESULTS_ROOT / "models"
LOGS_ROOT = RESULTS_ROOT / "logs"
AUDIT_ROOT = RESULTS_ROOT / "audit"
EVIDENCE_ROOT = RESULTS_ROOT / "evidence"
PAPER_ROOT = REPO_ROOT / "paper"


def ensure_runtime_directories() -> None:
    for path in (DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT, MODELS_ROOT, LOGS_ROOT, AUDIT_ROOT, EVIDENCE_ROOT, PAPER_ROOT):
        path.mkdir(parents=True, exist_ok=True)
