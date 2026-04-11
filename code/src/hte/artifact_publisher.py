from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_route_prefix(path: str) -> str:
    value = path.strip()
    if not value:
        raise ValueError("artifact route prefix cannot be empty")
    if not value.startswith("/"):
        raise ValueError("artifact route prefix must start with '/'")
    return value.rstrip("/") or "/"


def _clean_public_base_url(url: str | None) -> str | None:
    if url is None:
        return None
    value = url.strip()
    if not value:
        return None
    return value.rstrip("/")


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _collect_string_paths(value: Any, sink: set[str]) -> None:
    if isinstance(value, str):
        sink.add(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_string_paths(item, sink)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_string_paths(item, sink)


@dataclass(frozen=True)
class ArtifactPublishingConfig:
    enabled: bool
    route_prefix: str
    public_base_url: str | None
    snapshot_root: Path
    repo_root: Path
    allowed_roots: tuple[Path, ...]
    include_default_inputs: bool
    default_input_files: tuple[Path, ...]

    @classmethod
    def from_env(cls) -> "ArtifactPublishingConfig":
        streamable_path = os.environ.get("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp").strip() or "/mcp"
        default_route = f"{streamable_path.rstrip('/')}/artifacts"
        route_prefix = _clean_route_prefix(os.environ.get("HTE_ARTIFACT_PUBLIC_PATH", default_route))
        return cls(
            enabled=_env_flag("HTE_ARTIFACT_LINKS_ENABLED", True),
            route_prefix=route_prefix,
            public_base_url=_clean_public_base_url(
                os.environ.get("HTE_ARTIFACT_PUBLIC_BASE_URL")
                or os.environ.get("HTE_OAUTH_PUBLIC_BASE_URL")
            ),
            snapshot_root=RESULTS_ROOT / "published_runs",
            repo_root=REPO_ROOT,
            allowed_roots=(RESULTS_ROOT, DATA_ROOT),
            include_default_inputs=_env_flag("HTE_ARTIFACT_INCLUDE_DEFAULT_INPUTS", True),
            default_input_files=(
                DATA_ROOT / "aligned_hormuz_benchmark.csv",
                DATA_ROOT / "source_manifest.json",
            ),
        )


class ArtifactPublisher:
    def __init__(self, config: ArtifactPublishingConfig | None = None) -> None:
        self.config = ArtifactPublishingConfig.from_env() if config is None else config

    @property
    def route_prefix(self) -> str:
        return self.config.route_prefix

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _candidate_from_allowed_suffix(self, raw_text: str) -> Path | None:
        normalized = raw_text.replace("\\", "/")
        for root in self.config.allowed_roots:
            marker = f"/{root.name}/"
            if marker not in normalized:
                continue
            suffix = normalized.split(marker, 1)[1].lstrip("/")
            candidate = (root / suffix).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _source_candidate(self, value: str) -> Path | None:
        text = value.strip()
        if not text or "://" in text:
            return None
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = (self.config.repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            remapped = self._candidate_from_allowed_suffix(text)
            if remapped is None:
                return None
            candidate = remapped
        if not any(_path_within(candidate, root) for root in self.config.allowed_roots):
            return None
        return candidate

    def _relative_storage_path(self, source: Path) -> Path:
        if _path_within(source, self.config.repo_root):
            return source.relative_to(self.config.repo_root)
        return Path("external") / source.name

    def _public_url(self, relative_path: Path) -> str:
        rel = relative_path.as_posix()
        path = f"{self.config.route_prefix}/{rel}"
        if self.config.public_base_url:
            return f"{self.config.public_base_url}{path}"
        return path

    def _discover_files(self, payload: Any) -> list[Path]:
        raw_strings: set[str] = set()
        _collect_string_paths(payload, raw_strings)
        discovered: list[Path] = []
        seen: set[Path] = set()
        for item in sorted(raw_strings):
            candidate = self._source_candidate(item)
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            discovered.append(candidate)

        if discovered and self.config.include_default_inputs:
            for default_file in self.config.default_input_files:
                resolved = default_file.resolve()
                if resolved.exists() and resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    discovered.append(resolved)

        return discovered

    def publish(
        self,
        tool_name: str,
        request_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        response_data: Any,
    ) -> dict[str, Any] | None:
        if not self.config.enabled:
            return {"enabled": False}

        files = self._discover_files(response_data)
        if not files:
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{tool_name}_{request_id}"
        run_root = (self.config.snapshot_root / run_id).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        published_files: list[dict[str, str]] = []
        for source in files:
            relative_path = self._relative_storage_path(source)
            destination = run_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            public_relative = Path(run_id) / relative_path
            published_files.append(
                {
                    "source_path": str(source),
                    "stored_path": str(destination),
                    "public_path": f"{self.config.route_prefix}/{public_relative.as_posix()}",
                    "url": self._public_url(public_relative),
                }
            )

        request_record_path = run_root / "request_response.json"
        request_record = {
            "tool": tool_name,
            "request_id": request_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "args": _json_safe(list(args)),
            "kwargs": _json_safe(kwargs),
            "response_data": _json_safe(response_data),
            "published_file_count": len(published_files),
        }
        request_record_path.write_text(json.dumps(request_record, indent=2), encoding="utf-8")
        record_relative = Path(run_id) / "request_response.json"

        index_path = run_root / "artifact_index.json"
        index_payload = {
            "tool": tool_name,
            "request_id": request_id,
            "run_id": run_id,
            "files": published_files,
        }
        index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        index_relative = Path(run_id) / "artifact_index.json"

        return {
            "enabled": True,
            "run_id": run_id,
            "storage_root": str(run_root),
            "request_record": {
                "path": str(request_record_path),
                "public_path": f"{self.config.route_prefix}/{record_relative.as_posix()}",
                "url": self._public_url(record_relative),
            },
            "artifact_index": {
                "path": str(index_path),
                "public_path": f"{self.config.route_prefix}/{index_relative.as_posix()}",
                "url": self._public_url(index_relative),
            },
            "files": published_files,
        }

    def resolve_public_relative_path(self, relative_path: str) -> Path | None:
        if not self.config.enabled:
            return None
        candidate = (self.config.snapshot_root / relative_path).resolve()
        if not _path_within(candidate, self.config.snapshot_root.resolve()):
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate


_PUBLISHER = ArtifactPublisher()


def get_artifact_publisher() -> ArtifactPublisher:
    return _PUBLISHER
