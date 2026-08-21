"""Local association ranker: artifact loader and 22-feature contract.

The scorer is an external artifact (Atria_v8/ranker). This module owns the
semantic feature contract, the matrix construction and the inference
context needed by motion relink and the v7.4 evidence pass. All functions
take their inputs explicitly (no module-level globals).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .graph import node_position_um

FALLBACK_FEATURES = [
    "edge_prob", "source_in_degree", "source_out_degree", "target_in_degree",
    "target_out_degree", "source_density_7um", "target_density_7um",
    "raw_distance_um", "motion_distance_um", "motion_gain_um", "candidate_rank",
    "candidate_count", "dz_um", "dy_um", "dx_um", "abs_dz_um", "abs_dy_um",
    "abs_dx_um", "velocity_um", "source_frame_size_norm",
    "target_frame_size_norm", "t_norm",
]


def normalize_feature_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def contract_aliases(
    *,
    edge_prob: float,
    has_learned_edge: float,
    source_in_degree: float,
    source_out_degree: float,
    target_in_degree: float,
    target_out_degree: float,
    source_frame_count: float,
    target_frame_count: float,
    source_density_7um: float,
    target_density_7um: float,
    candidate_rank_dist: float,
    candidate_count: float,
    edge_dz_um: float,
    edge_dy_um: float,
    edge_dx_um: float,
    edge_dist_um: float,
    edge_xy_um: float,
    edge_abs_z_um: float,
    motion_dist_um: float,
    motion_gain_um: float,
    source_has_prev: float,
    target_has_next: float,
    target_best_next_prob: float,
    t_norm: float,
    velocity_um: float = 0.0,
) -> dict[str, float]:
    """Expand the 22 semantic features into every historical alias name."""
    values = {
        "edge_prob": edge_prob,
        "has_learned_edge": has_learned_edge,
        "source_in_degree": source_in_degree,
        "source_out_degree": source_out_degree,
        "target_in_degree": target_in_degree,
        "target_out_degree": target_out_degree,
        "source_frame_count": source_frame_count,
        "target_frame_count": target_frame_count,
        "source_density_7um": source_density_7um,
        "target_density_7um": target_density_7um,
        "candidate_rank_dist": candidate_rank_dist,
        "candidate_rank": candidate_rank_dist,
        "candidate_count": candidate_count,
        "edge_dz_um": edge_dz_um,
        "edge_dy_um": edge_dy_um,
        "edge_dx_um": edge_dx_um,
        "edge_dist_um": edge_dist_um,
        "edge_xy_um": edge_xy_um,
        "edge_abs_z_um": edge_abs_z_um,
        "motion_dist_um": motion_dist_um,
        "motion_gain_um": motion_gain_um,
        "source_has_prev": source_has_prev,
        "target_has_next": target_has_next,
        "target_best_next_prob": target_best_next_prob,
        "t_norm": t_norm,
        "learned_edge_prob": edge_prob,
        "primary_prob": edge_prob,
        "prob": edge_prob,
        "src_in_degree": source_in_degree,
        "src_out_degree": source_out_degree,
        "dst_in_degree": target_in_degree,
        "dst_out_degree": target_out_degree,
        "source_frame_size": source_frame_count,
        "target_frame_size": target_frame_count,
        "src_density_7um": source_density_7um,
        "dst_density_7um": target_density_7um,
        "raw_distance_um": edge_dist_um,
        "distance_um": edge_dist_um,
        "dist_um": edge_dist_um,
        "motion_distance_um": motion_dist_um,
        "motion_dist": motion_dist_um,
        "motion_gain": motion_gain_um,
        "dz_um": edge_dz_um,
        "dy_um": edge_dy_um,
        "dx_um": edge_dx_um,
        "abs_dz_um": edge_abs_z_um,
        "abs_dy_um": abs(edge_dy_um),
        "abs_dx_um": abs(edge_dx_um),
        "velocity_um": velocity_um,
        "speed_um": velocity_um,
        "time_norm": t_norm,
        "bias": 1.0,
    }
    return {normalize_feature_name(key): float(value) for key, value in values.items()}


def build_feature_matrix(alias_rows: list[dict[str, float]], feature_names: list[str]) -> np.ndarray:
    missing: set[str] = set()
    rows: list[list[float]] = []
    for aliases in alias_rows:
        row: list[float] = []
        for feature_name in feature_names:
            key = normalize_feature_name(feature_name)
            if key not in aliases:
                missing.add(str(feature_name))
                row.append(0.0)
            else:
                row.append(float(aliases[key]))
        rows.append(row)
    if missing:
        raise RuntimeError(
            "The attached ranker requests unsupported feature names before inference: "
            + ", ".join(sorted(missing))
        )
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("Invalid public-ranker semantic preflight matrix.")
    return matrix


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _deep_values(obj, accepted_keys: set[str]):
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).lower().replace("-", "_").replace(" ", "_")
            if key_norm in accepted_keys:
                out.append(value)
            out.extend(_deep_values(value, accepted_keys))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_deep_values(value, accepted_keys))
    return out


def _to_1d(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def _extract_state(payload):
    if isinstance(payload, torch.nn.Module):
        return payload, None
    if isinstance(payload, dict):
        for key in ("model", "ranker", "network", "module"):
            value = payload.get(key)
            if isinstance(value, torch.nn.Module):
                return value, None
        for key in ("model_state_dict", "state_dict", "ranker_state_dict", "net_state_dict", "weights"):
            value = payload.get(key)
            if isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values()):
                return None, value
        if any(isinstance(v, torch.Tensor) for v in payload.values()):
            return None, payload
    raise RuntimeError("Unsupported local-ranker checkpoint payload. Expected a torch module or state_dict.")


class _InferredMLP(torch.nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], activation: str = "relu"):
        super().__init__()
        cleaned = {}
        for key, value in state.items():
            key2 = str(key)
            for prefix in ("module.", "model.", "ranker.", "network."):
                if key2.startswith(prefix):
                    key2 = key2[len(prefix):]
            cleaned[key2] = value.detach().cpu()
        weight_items = [(key, value) for key, value in cleaned.items() if key.endswith(".weight") and value.ndim == 2]
        weight_items.sort(key=lambda item: _natural_key(item[0]))
        if not weight_items:
            raise RuntimeError("No 2D linear weights were found in the ranker checkpoint.")
        self.layers = torch.nn.ModuleList()
        for weight_key, weight in weight_items:
            prefix = weight_key[: -len(".weight")]
            bias = cleaned.get(prefix + ".bias")
            layer = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
            with torch.no_grad():
                layer.weight.copy_(weight.to(dtype=torch.float32))
                if bias is not None:
                    layer.bias.copy_(bias.to(dtype=torch.float32))
            self.layers.append(layer)
        self.activation = activation.lower()

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index + 1 < len(self.layers):
                if self.activation == "gelu":
                    x = torch.nn.functional.gelu(x)
                elif self.activation in {"silu", "swish"}:
                    x = torch.nn.functional.silu(x)
                elif self.activation == "tanh":
                    x = torch.tanh(x)
                else:
                    x = torch.relu(x)
        return x


class AssociationRanker:
    """Loaded scorer with its feature contract and normalization metadata."""

    def __init__(self, root: Path):
        self.root = Path(root)
        checkpoints = []
        for pattern in ("**/*.pt", "**/*.pth", "**/*.jit", "**/*.torchscript"):
            checkpoints.extend(self.root.glob(pattern))

        def priority(path: Path):
            name = path.name.lower()
            return (
                0 if "local_association_ranker" in name else 1 if "ranker" in name else 2,
                0 if "best" in name else 1,
                len(path.parts),
                str(path),
            )

        checkpoints = sorted({path for path in checkpoints if path.is_file()}, key=priority)
        if not checkpoints:
            raise FileNotFoundError(f"No .pt/.pth ranker checkpoint found under {root}")
        self.checkpoint = checkpoints[0]
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(self.checkpoint, map_location="cpu")

        metadata_objects: list[object] = [payload]
        for path in sorted(self.root.glob("**/*.json")):
            try:
                if path.stat().st_size <= 4_000_000:
                    metadata_objects.append(json.loads(path.read_text()))
            except Exception:
                pass
        for path in sorted(self.root.glob("**/*.npz")):
            try:
                with np.load(path, allow_pickle=True) as data:
                    metadata_objects.append({key: data[key].tolist() for key in data.files})
            except Exception:
                pass

        module, state = _extract_state(payload)
        activation_values = []
        for obj in metadata_objects:
            activation_values.extend(_deep_values(obj, {"activation", "hidden_activation"}))
        activation = str(activation_values[0]) if activation_values else "relu"
        self.model = module if module is not None else _InferredMLP(state, activation=activation)
        self.model.eval().cpu()

        if module is not None:
            linear_layers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
            if not linear_layers:
                raise RuntimeError("Loaded ranker module contains no torch.nn.Linear input layer.")
            input_dim = int(linear_layers[0].in_features)
        else:
            input_dim = int(self.model.layers[0].in_features)
        self.input_dim = input_dim

        feature_candidates = []
        for obj in metadata_objects:
            feature_candidates.extend(_deep_values(obj, {
                "feature_names", "features", "input_features", "columns", "feature_columns",
            }))
        feature_names = None
        for candidate in feature_candidates:
            if isinstance(candidate, (list, tuple)) and candidate and all(isinstance(v, str) for v in candidate):
                if len(candidate) == input_dim:
                    feature_names = list(candidate)
                    break
        if feature_names is None:
            if input_dim != len(FALLBACK_FEATURES):
                raise RuntimeError(
                    f"Ranker input_dim={input_dim}, but no matching feature_names metadata was found. "
                    "The public fallback is defined only for 22 features."
                )
            feature_names = list(FALLBACK_FEATURES)
            self.feature_source = "public_22_feature_fallback"
        else:
            self.feature_source = "artifact_metadata"
        self.feature_names = feature_names

        mean_candidates = []
        std_candidates = []
        for obj in metadata_objects:
            mean_candidates.extend(_deep_values(obj, {
                "feature_mean", "feature_means", "x_mean", "mean", "scaler_mean", "means",
            }))
            std_candidates.extend(_deep_values(obj, {
                "feature_std", "feature_stds", "x_std", "std", "scale", "scaler_scale", "stds",
            }))
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.std = np.ones(input_dim, dtype=np.float32)
        for candidate in mean_candidates:
            arr = _to_1d(candidate)
            if arr is not None and len(arr) == input_dim and np.isfinite(arr).all():
                self.mean = arr.astype(np.float32)
                break
        for candidate in std_candidates:
            arr = _to_1d(candidate)
            if arr is not None and len(arr) == input_dim and np.isfinite(arr).all():
                self.std = np.maximum(arr.astype(np.float32), 1e-6)
                break

        positive_values = []
        for obj in metadata_objects:
            positive_values.extend(_deep_values(obj, {"positive_class", "positive_class_index", "pos_class"}))
        self.positive_class_index = int(positive_values[0]) if positive_values else 1

    @torch.inference_mode()
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dim:
            raise ValueError(f"Bad ranker feature matrix shape: {matrix.shape}; expected (*, {self.input_dim})")
        if not np.isfinite(matrix).all():
            raise ValueError("Ranker feature matrix contains non-finite values.")
        x = (matrix - self.mean[None, :]) / self.std[None, :]
        out = self.model(torch.from_numpy(x)).detach().cpu()
        if out.ndim == 1:
            out = out[:, None]
        if out.shape[1] == 1:
            values = out[:, 0]
            if bool(torch.all((values >= 0.0) & (values <= 1.0))):
                probs = values
            else:
                probs = torch.sigmoid(values)
        elif out.shape[1] == 2:
            probs = torch.softmax(out, dim=1)[:, self.positive_class_index]
        else:
            raise RuntimeError(f"Unexpected ranker output shape: {tuple(out.shape)}")
        probs_np = probs.numpy().astype(np.float64)
        if not np.isfinite(probs_np).all():
            raise RuntimeError("Ranker returned non-finite probabilities.")
        return np.clip(probs_np, 0.0, 1.0)


def load_ranker(root: Path) -> AssociationRanker:
    return AssociationRanker(root)


def build_ranker_context(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate per-frame / per-node statistics used by the feature contract."""
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}
    best_next_prob: dict[int, float] = {}
    for edge in raw_edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        out_degree[source_id] = out_degree.get(source_id, 0) + 1
        in_degree[target_id] = in_degree.get(target_id, 0) + 1
        value = edge.get("edge_prob")
        try:
            prob = float(value)
        except (TypeError, ValueError):
            prob = 0.0
        if np.isfinite(prob):
            if prob < 0.0 or prob > 1.0:
                prob = 1.0 / (1.0 + np.exp(-max(-20.0, min(20.0, prob))))
            best_next_prob[source_id] = max(best_next_prob.get(source_id, 0.0), float(np.clip(prob, 0.0, 1.0)))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    position_um = {node_id: node_position_um(node) for node_id, node in nodes_by_id.items()}
    density_7um: dict[int, float] = {}
    for t, ids in ids_by_t.items():
        if not ids:
            continue
        points = np.stack([position_um[node_id] for node_id in ids], axis=0)
        tree = cKDTree(points)
        counts = tree.query_ball_point(points, r=7.0, return_length=True)
        for node_id, count in zip(ids, counts):
            density_7um[node_id] = float(max(0, int(count) - 1))
    max_t = max((int(node["t"]) for node in nodes_by_id.values()), default=1)
    max_frame_size = max((len(ids) for ids in ids_by_t.values()), default=1)
    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "best_next_prob": best_next_prob,
        "ids_by_t": ids_by_t,
        "position_um": position_um,
        "density_7um": density_7um,
        "max_t": max(1, max_t),
        "max_frame_size": max(1, max_frame_size),
    }


def alias_row_for_candidate(
    nodes_by_id: dict[int, dict[str, object]],
    context: dict[str, object],
    source_id: int,
    target_id: int,
    raw_distance_um: float,
    motion_distance_um: float,
    primary_prob: float,
    has_learned_edge: bool,
    candidate_rank_dist: int,
    candidate_count: int,
    predicted_position_um: np.ndarray,
) -> dict[str, float]:
    source = nodes_by_id[source_id]
    target = nodes_by_id[target_id]
    source_pos = context["position_um"][source_id]
    target_pos = context["position_um"][target_id]
    delta_um = target_pos - source_pos
    velocity_um = predicted_position_um - source_pos
    source_frame_count = len(context["ids_by_t"].get(int(source["t"]), []))
    target_frame_count = len(context["ids_by_t"].get(int(target["t"]), []))
    edge_xy_um = float(np.linalg.norm(delta_um[1:]))
    t_norm = float(source["t"]) / float(context["max_t"])
    return contract_aliases(
        edge_prob=float(primary_prob),
        has_learned_edge=float(bool(has_learned_edge)),
        source_in_degree=float(context["in_degree"].get(source_id, 0)),
        source_out_degree=float(context["out_degree"].get(source_id, 0)),
        target_in_degree=float(context["in_degree"].get(target_id, 0)),
        target_out_degree=float(context["out_degree"].get(target_id, 0)),
        source_frame_count=float(source_frame_count),
        target_frame_count=float(target_frame_count),
        source_density_7um=float(context["density_7um"].get(source_id, 0.0)),
        target_density_7um=float(context["density_7um"].get(target_id, 0.0)),
        candidate_rank_dist=float(candidate_rank_dist),
        candidate_count=float(candidate_count),
        edge_dz_um=float(delta_um[0]),
        edge_dy_um=float(delta_um[1]),
        edge_dx_um=float(delta_um[2]),
        edge_dist_um=float(raw_distance_um),
        edge_xy_um=edge_xy_um,
        edge_abs_z_um=abs(float(delta_um[0])),
        motion_dist_um=float(motion_distance_um),
        motion_gain_um=float(raw_distance_um - motion_distance_um),
        source_has_prev=float(context["in_degree"].get(source_id, 0) > 0),
        target_has_next=float(context["out_degree"].get(target_id, 0) > 0),
        target_best_next_prob=float(context["best_next_prob"].get(target_id, 0.0)),
        t_norm=t_norm,
        velocity_um=float(np.linalg.norm(velocity_um)),
    )


def score_records(ranker: AssociationRanker, nodes_by_id: dict[int, dict[str, object]],
                  records: list[dict[str, object]]) -> np.ndarray:
    alias_rows = [
        alias_row_for_candidate(nodes_by_id, record["context"], record["source_id"], record["target_id"],
                                record["raw_distance_um"], record["motion_distance_um"], record["primary_prob"],
                                record["has_learned_edge"], record["candidate_rank_dist"], record["candidate_count"],
                                record["predicted_position_um"])
        for record in records
    ]
    matrix = build_feature_matrix(alias_rows, ranker.feature_names)
    return ranker.predict_proba(matrix)
