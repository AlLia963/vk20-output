"""Offline dependency bootstrap (Internet OFF on Kaggle).

Installs only the packages that are actually missing or stale from the
attached wheelhouse (never replaces healthy numpy/scipy), mirroring the
offline-safe behavior of the v8.2 runtime without its global machinery.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

from .resources import ResourceBundle

# name -> (import module, pip spec)
MODULE_SPECS: dict[str, tuple[str, str]] = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}

EXTRA_SPECS: dict[str, list[str]] = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}


def _module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def _import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, (module_name, _spec) in MODULE_SPECS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def _missing_names(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, (module, _spec) in MODULE_SPECS.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def _stale_names() -> list[str]:
    try:
        import zarr
        if int(str(getattr(zarr, "__version__", "0")).split(".", 1)[0]) < 3:
            return ["zarr"]
    except Exception:
        return ["zarr"]
    return []


def _specs_for(names: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = MODULE_SPECS[name][1].lower()
        if key not in seen:
            seen.add(key)
            specs.append(MODULE_SPECS[name][1])
        for spec in EXTRA_SPECS.get(name, []):
            key = spec.lower()
            if key not in seen:
                seen.add(key)
                specs.append(spec)
    return specs


def _purge(names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in names:
        roots.add(MODULE_SPECS[name][0].split(".")[0])
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def ensure_dependencies(bundle: ResourceBundle, python: str | None = None) -> None:
    """Install anything missing/stale from support_pack/wheels (Internet OFF)."""
    wheel_dir = bundle.support_pack / "wheels"
    if not wheel_dir.is_dir():
        raise FileNotFoundError(f"Missing wheelhouse: {wheel_dir}")

    for _ in range(5):
        stale = _stale_names()
        if stale:
            names = stale
        else:
            failures = _import_failures()
            names = [pkg for pkg, (module_name, _spec) in MODULE_SPECS.items() if _module_missing(module_name)]
            names.extend(_missing_names(failures))
            names = sorted(set(names))
        if not names:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        specs = _specs_for(names)
        force = bool({"polars", "zarr"} & set(names))
        cmd = [python or sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force:
            cmd.append("--force-reinstall")
        cmd.extend(["--find-links", str(wheel_dir)])
        cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", names)
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            print("Offline dependency install failed. Last pip output:")
            print((result.stdout or "")[-2000:])
            print((result.stderr or "")[-2000:])
            raise ImportError("Offline dependency install failed: " + ", ".join(names))
        _purge(names)

    raise ImportError("Dependency recovery did not converge after repeated offline installs.")
