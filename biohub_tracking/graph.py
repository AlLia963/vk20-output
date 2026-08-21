"""Track graph I/O, physical geometry and volume-frame access.

Kept free of configuration globals: callers pass a RunConfig where tuning
knobs are needed (e.g. synthetic-node refinement).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import blosc2
import numpy as np
import tracksdata as td

from .config import RunConfig

# Physical voxel size in microns, ordered z/y/x (competition convention).
VOXEL_UM = (1.625, 0.40625, 0.40625)
SCALE_ARRAY = np.asarray(VOXEL_UM, dtype=np.float64)


def load_track_graph(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def point_span_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_UM[0]
    dy = (a[1] - b[1]) * VOXEL_UM[1]
    dx = (a[2] - b[2]) * VOXEL_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def edge_span_um(source: dict[str, object], target: dict[str, object]) -> float:
    return point_span_um(node_xyz(source), node_xyz(target))


def node_xyz(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def node_position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]), float(node["y"]), float(node["x"])],
        dtype=np.float64,
    ) * SCALE_ARRAY


def edge_priority(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def fresh_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1


def read_frame_volume(
    dataset: str,
    t: int,
    test_dir: Path,
    frame_cache: dict[int, np.ndarray],
) -> np.ndarray:
    """Read one 3D frame of a test volume (zarr v3 chunk fast-path first)."""
    if t in frame_cache:
        return frame_cache[t]
    zarr_root = test_dir / f"{dataset}.zarr"
    meta = json.loads((zarr_root / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_root / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_root / "0", mode="r")[t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_node(
    cfg: RunConfig,
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    test_dir: Path,
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    """Shift a synthetic midpoint to the intensity centroid of its patch."""
    if not cfg.gap_refine_synthetic or dataset is None:
        return midpoint
    try:
        frame = read_frame_volume(dataset, t, test_dir, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - cfg.gap_refine_win_z)
        z1 = min(frame.shape[0], z + cfg.gap_refine_win_z + 1)
        y0 = max(0, y - cfg.gap_refine_win_yx)
        y1 = min(frame.shape[1], y + cfg.gap_refine_win_yx + 1)
        x0 = max(0, x - cfg.gap_refine_win_yx)
        x1 = min(frame.shape[2], x + cfg.gap_refine_win_yx + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_span_um(refined, midpoint) > cfg.gap_refine_max_shift_um:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint
