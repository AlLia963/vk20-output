"""Single-root resource resolution for the Atria_v8 private dataset.

v9.0 mounts exactly one private dataset (Atria_v8). This module locates it,
verifies the bundled weights against HASHES.json, checks the offline wheel
set, and resolves the secondary-model weights for the two run modes:
  - verification (seed314159)  -> byte-parity gate vs v8.2
  - optimization (private)     -> r=0.50 merged weights
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

from .config import MODE_VERIFY, MODE_OPTIMIZE

INPUT_ROOT = Path("/kaggle/input")

REQUIRED_WHEEL_PREFIXES = [
    "tracksdata-", "zarr-", "pyscipopt-", "geff-", "geff_spec-",
    "ilpy-", "imagecodecs-", "rustworkx-", "numcodecs-", "donfig-", "bidict-",
]

SEED314159_SHA256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
PRIVATE_R050_SHA256 = "f689d4ae760c18500cc64bbf2f619abf7488724efab2ac6bb592b19caf3f5e25"

SECONDARY_REL = Path("weights/unet_transformer/split_0/edge_predictor_best.pth")


def _env_root(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


@dataclasses.dataclass(frozen=True)
class ResourceBundle:
    root: Path
    manifest: dict

    @property
    def support_pack(self) -> Path:
        return _env_root("BIOHUB_SUPPORT_PACK", self.root / "support_pack")

    @property
    def seed314159(self) -> Path:
        return _env_root("BIOHUB_SEED314159", self.root / "seed314159")

    @property
    def ranker(self) -> Path:
        return _env_root("BIOHUB_RANKER", self.root / "ranker")

    @property
    def deepcenter(self) -> Path:
        return self.root / "deepcenter"

    @property
    def private_weights(self) -> Path:
        return _env_root("BIOHUB_PRIVATE_WEIGHTS", self.root / "private_weights")


def locate_private_root(override: str = "") -> Path:
    """Resolve the single private dataset root.

    Prefers the BIOHUB_PRIVATE_ROOT environment variable; otherwise scans
    /kaggle/input and requires exactly one HASHES.json (i.e. only Atria_v8
    plus the competition data are mounted).
    """
    env_root = (override or os.environ.get("BIOHUB_PRIVATE_ROOT", "")).strip()
    if env_root:
        root = Path(env_root)
        if not (root / "HASHES.json").is_file():
            raise FileNotFoundError(f"BIOHUB_PRIVATE_ROOT has no HASHES.json: {root}")
        return root
    if not INPUT_ROOT.exists():
        raise RuntimeError("BIOHUB_PRIVATE_ROOT unset and /kaggle/input not found")
    hits = [p.parent for p in INPUT_ROOT.rglob("HASHES.json") if p.is_file()]
    if len(hits) != 1:
        raise RuntimeError(
            "Expected exactly one HASHES.json under /kaggle/input; "
            f"found {len(hits)}"
        )
    return hits[0]


def load_bundle(root: Path | str) -> ResourceBundle:
    root = Path(root)
    manifest_path = root / "HASHES.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return ResourceBundle(root=root, manifest=manifest)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(bundle: ResourceBundle) -> None:
    """Every file listed in HASHES.json must exist with the recorded SHA256."""
    for name, info in bundle.manifest.items():
        full = bundle.root / info["path"]
        if not full.is_file():
            raise FileNotFoundError(f"Missing private dataset file: {full}")
        actual = sha256_file(full)
        if actual != info["sha256"]:
            raise RuntimeError(
                f"Hash mismatch for {name}: expected {info['sha256']}, got {actual}"
            )
    print("V9 single-private hash verification: PASS")


def preflight_wheels(bundle: ResourceBundle) -> None:
    wheel_dir = bundle.support_pack / "wheels"
    names = [p.name for p in wheel_dir.glob("*.whl")] if wheel_dir.is_dir() else []
    missing = [
        prefix for prefix in REQUIRED_WHEEL_PREFIXES
        if not any(name.startswith(prefix) for name in names)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required wheels in support_pack/wheels: " + ", ".join(missing)
        )
    print("V9 wheel preflight: PASS")


def resolve_secondary(bundle: ResourceBundle, mode: str, private_rel: str = "") -> tuple[Path, str]:
    """Return (weights_path, expected_sha256) for the requested secondary mode."""
    if mode == MODE_VERIFY:
        weight = bundle.seed314159 / SECONDARY_REL
        expected = SEED314159_SHA256
    elif mode == MODE_OPTIMIZE:
        rel = Path(private_rel or os.environ.get(
            "BIOHUB_PRIVATE_WEIGHTS_REL",
            "weights/unet_transformer/split_0/edge_predictor_best.pth",
        ))
        weight = bundle.private_weights / rel
        expected = PRIVATE_R050_SHA256
    else:
        raise ValueError(f"unknown secondary mode: {mode}")
    if not weight.is_file():
        raise FileNotFoundError(f"Secondary weight missing: {weight}")
    return weight, expected
