"""Host-independent paths; configuration is read per job, never cached at import."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Runtime:
    source: Path
    python: Path
    models: Path
    cache: Path


def runtime_paths():
    values = {}
    for key in ("SOURCE", "PYTHON", "MODELS"):
        name = "UNIFORGE3D_" + key
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Set {name}; see ComfyUI-UniForge3D README installation instructions")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise RuntimeError(f"{name} must be an absolute path")
        values[key.lower()] = path
    source, python, models = (values[key] for key in ("source", "python", "models"))
    if not (source / "uniforge3d" / "__init__.py").is_file():
        raise RuntimeError(f"UNIFORGE3D_SOURCE is not an upstream UniForge3D checkout: {source}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"UNIFORGE3D_PYTHON is not an executable runtime Python: {python}")
    if not (models / "uniforge3d" / "pipeline.json").is_file():
        raise RuntimeError(f"Missing {models / 'uniforge3d' / 'pipeline.json'}; provision model weights first")
    cache = Path(os.environ.get("UNIFORGE3D_CACHE", str(models / "hf-cache"))).expanduser()
    if not cache.is_absolute():
        raise RuntimeError("UNIFORGE3D_CACHE must be an absolute path")
    return Runtime(source, python, models, cache)
