from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    nsd_root: Path
    imagery_root: Path
    checkpoint_root: Path
    output_root: Path
    raw_nsd_root: Path | None = None
    openclip_version: Path | None = None
    generative_models_path: Path | None = None

    @classmethod
    def from_mapping(cls, data: dict, base_dir: str | Path) -> "ProjectPaths":
        base = Path(base_dir)

        def resolve(key: str) -> Path:
            value = Path(data[key])
            return value if value.is_absolute() else base / value

        def optional_resolve(key: str, *aliases: str) -> Path | None:
            value = data.get(key)
            for alias in aliases:
                if value not in {None, ""}:
                    break
                value = data.get(alias)
            if value in {None, ""}:
                return None
            path = Path(value)
            return path if path.is_absolute() else base / path

        return cls(
            data_root=resolve("data_root"),
            nsd_root=resolve("nsd_root"),
            imagery_root=resolve("imagery_root"),
            checkpoint_root=resolve("checkpoint_root"),
            output_root=resolve("output_root"),
            raw_nsd_root=optional_resolve("raw_nsd_root"),
            openclip_version=optional_resolve("openclip_version"),
            generative_models_path=optional_resolve("generative_models_path", "generative-models-path"),
        )

    def ensure_outputs(self) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
