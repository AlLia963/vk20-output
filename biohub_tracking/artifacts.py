"""Atria_v8 artifact materialization for the Kaggle runtime."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import MODE_VERIFY, MODE_OPTIMIZE
from .resources import ResourceBundle, resolve_secondary, sha256_file

PRIMARY_EXPECTED_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"


def verify_primary_manifest(bundle: ResourceBundle) -> dict:
    manifest_path = bundle.support_pack / "ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing primary manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    actual = str(manifest.get("model", {}).get("weight_sha256", ""))
    if actual != PRIMARY_EXPECTED_SHA256:
        raise RuntimeError(
            "Primary model checksum mismatch: "
            f"expected {PRIMARY_EXPECTED_SHA256}, got {actual or 'missing'}"
        )
    return manifest


def materialize_repo(bundle: ResourceBundle, working_dir: Path | str) -> Path:
    working_dir = Path(working_dir)
    source = bundle.support_pack / "repo"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing tracking repo in support_pack: {source}")
    repo_dir = working_dir / "tracking_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    shutil.copytree(source, repo_dir)

    # The prediction script resolves --weights relative to its working
    # directory, so the primary weights must live inside the repo tree.
    weights_src = bundle.support_pack / "weights"
    if not weights_src.is_dir():
        raise FileNotFoundError(f"Missing primary weights in support_pack: {weights_src}")
    weights_dst = repo_dir / "weights"
    if weights_dst.exists():
        shutil.rmtree(weights_dst)
    shutil.copytree(weights_src, weights_dst)

    required = [
        repo_dir / "scripts" / "predict_unet_transformer.py",
        repo_dir / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", repo_dir)
    print("Weights:", required[1])
    return repo_dir


def stage_secondary_weights(
    bundle: ResourceBundle,
    mode: str,
    working_dir: Path | str,
    private_rel: str = "",
) -> tuple[Path, str]:
    """Copy the active secondary weights next to a config.json and verify hash."""
    working_dir = Path(working_dir)
    source, expected = resolve_secondary(bundle, mode, private_rel=private_rel)
    target = working_dir / "secondary_seed_weights" / "unet_transformer" / "split_0"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / "edge_predictor_best.pth")
    config_source = source.parent / "config.json"
    if config_source.is_file():
        shutil.copy2(config_source, target / "config.json")
    actual = sha256_file(target / "edge_predictor_best.pth")
    if actual != expected:
        raise RuntimeError(
            f"Staged secondary checksum mismatch: expected {expected}, got {actual}"
        )
    return target / "edge_predictor_best.pth", actual


def mode_label(mode: str) -> str:
    return {
        MODE_VERIFY: "verification (seed314159, byte-parity vs v8.2)",
        MODE_OPTIMIZE: "optimization (private r=0.50)",
    }.get(mode, mode)
