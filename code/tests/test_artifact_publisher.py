from __future__ import annotations

import json
from pathlib import Path

from hte.artifact_publisher import ArtifactPublisher, ArtifactPublishingConfig


def _build_config(repo_root: Path, enabled: bool = True) -> ArtifactPublishingConfig:
    data_root = repo_root / "data"
    results_root = repo_root / "results"
    return ArtifactPublishingConfig(
        enabled=enabled,
        route_prefix="/mcp/hormuz/artifacts",
        public_base_url="https://lightcap.ai",
        snapshot_root=results_root / "published_runs",
        repo_root=repo_root,
        allowed_roots=(results_root, data_root),
        include_default_inputs=True,
        default_input_files=(
            data_root / "aligned_hormuz_benchmark.csv",
            data_root / "source_manifest.json",
        ),
    )


def test_publish_creates_unique_run_snapshot_with_links(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    results_root = repo_root / "results"
    data_root.mkdir(parents=True)
    results_root.mkdir(parents=True)

    metrics_path = results_root / "model_metrics_lb12_hz1_u128-96-64.json"
    metrics_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    (data_root / "aligned_hormuz_benchmark.csv").write_text("timestamp,value\nx,1\n", encoding="utf-8")
    (data_root / "source_manifest.json").write_text(json.dumps({"sources": []}), encoding="utf-8")

    publisher = ArtifactPublisher(_build_config(repo_root))
    published = publisher.publish(
        tool_name="train_model",
        request_id="abc123",
        args=(),
        kwargs={"force_retrain": False},
        response_data={"metrics_path": str(metrics_path)},
    )

    assert published is not None
    assert published["enabled"] is True
    assert published["run_id"].endswith("_train_model_abc123")
    assert len(published["files"]) == 3

    record_path = Path(published["request_record"]["path"])
    index_path = Path(published["artifact_index"]["path"])
    assert record_path.exists()
    assert index_path.exists()

    urls = [item["url"] for item in published["files"]]
    assert all(url.startswith("https://lightcap.ai/mcp/hormuz/artifacts/") for url in urls)

    # One generated output + two default input files should be accessible.
    assert any(item["source_path"].endswith("model_metrics_lb12_hz1_u128-96-64.json") for item in published["files"])
    assert any(item["source_path"].endswith("aligned_hormuz_benchmark.csv") for item in published["files"])
    assert any(item["source_path"].endswith("source_manifest.json") for item in published["files"])

    # Route resolver must map run-relative paths back to stored files.
    first_public_path = published["files"][0]["public_path"]
    relative = first_public_path.removeprefix("/mcp/hormuz/artifacts/")
    resolved = publisher.resolve_public_relative_path(relative)
    assert resolved is not None
    assert resolved.exists()


def test_publish_respects_disable_flag(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    publisher = ArtifactPublisher(_build_config(repo_root, enabled=False))
    published = publisher.publish(
        tool_name="train_model",
        request_id="abc123",
        args=(),
        kwargs={},
        response_data={"path": "/does/not/matter"},
    )
    assert published == {"enabled": False}
    assert publisher.resolve_public_relative_path("anything") is None


def test_publish_skips_when_no_files_found(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data").mkdir(parents=True)
    (repo_root / "results").mkdir(parents=True)
    publisher = ArtifactPublisher(_build_config(repo_root))
    published = publisher.publish(
        tool_name="backend_status",
        request_id="abc123",
        args=(),
        kwargs={},
        response_data={"available": True, "note": "no files here"},
    )
    assert published is None


def test_publish_remaps_foreign_absolute_result_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    results_root = repo_root / "results"
    data_root.mkdir(parents=True)
    (results_root / "figures").mkdir(parents=True)
    (data_root / "aligned_hormuz_benchmark.csv").write_text("timestamp,value\nx,1\n", encoding="utf-8")
    (data_root / "source_manifest.json").write_text(json.dumps({"sources": []}), encoding="utf-8")
    local_figure = results_root / "figures" / "training_history_lb12_hz1_u128-96-64.png"
    local_figure.write_bytes(b"png")

    publisher = ArtifactPublisher(_build_config(repo_root))
    foreign_path = "/Users/example/project/results/figures/training_history_lb12_hz1_u128-96-64.png"
    published = publisher.publish(
        tool_name="train_model",
        request_id="abc123",
        args=(),
        kwargs={},
        response_data={"figures": {"training_history_figure": foreign_path}},
    )

    assert published is not None
    assert any(item["source_path"] == str(local_figure.resolve()) for item in published["files"])
