"""Graph repair stages (motion relink, gap recovery, safe divisions,
short-track filtering, linefit smoothing).

Every stage takes the RunConfig explicitly; there are no module-level
configuration globals. Numeric behavior matches the v8.2 champion pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .config import RunConfig
from .graph import (
    SCALE_ARRAY,
    VOXEL_UM,
    edge_span_um,
    fresh_node_id,
    node_position_um,
    node_xyz,
    point_span_um,
    refine_synthetic_node,
)
from .ranker import AssociationRanker, build_ranker_context, score_records


# --------------------------------------------------------------------------
# Forward-acceleration lookahead (association-context bonus)
# --------------------------------------------------------------------------

def acceleration_lookahead(
    source_id: int,
    target_id: int,
    node_time: dict[int, int],
    ids_by_t: dict[int, list[int]],
    position_um: dict[int, np.ndarray],
    next_step_gate_um: float,
) -> tuple[float | None, int]:
    source_pos = np.asarray(position_um[source_id], dtype=np.float64)
    target_pos = np.asarray(position_um[target_id], dtype=np.float64)
    target_t = int(node_time[target_id])
    future_ids = ids_by_t.get(target_t + 1, [])
    current_velocity = target_pos - source_pos
    residuals: list[float] = []
    for next_id in future_ids:
        next_pos = np.asarray(position_um[next_id], dtype=np.float64)
        next_velocity = next_pos - target_pos
        if float(np.linalg.norm(next_velocity)) > float(next_step_gate_um):
            continue
        residual = float(np.linalg.norm(next_velocity - current_velocity))
        if np.isfinite(residual):
            residuals.append(residual)
    if not residuals:
        return None, 0
    return min(residuals), len(residuals)


def acceleration_bonus(residual_um: float | None, max_accel_um: float, max_bonus: float) -> float:
    if residual_um is None or not np.isfinite(residual_um) or max_accel_um <= 0 or max_bonus <= 0:
        return 0.0
    support = max(0.0, 1.0 - float(residual_um) / float(max_accel_um))
    return float(max_bonus) * support


# --------------------------------------------------------------------------
# Motion relink: per-frame 1:1 assignment with ranker-fused cost
# --------------------------------------------------------------------------

def relink_by_motion(
    cfg: RunConfig,
    ranker: AssociationRanker,
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
    raw_edges_for_context: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not cfg.motion_relink or not nodes_by_id:
        return []
    if ranker is None:
        raise RuntimeError("Local association ranker was not loaded during preflight.")

    learned_edge_probs = learned_edge_probs or {}
    raw_edges_for_context = raw_edges_for_context or []
    context = build_ranker_context(nodes_by_id, raw_edges_for_context)

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > cfg.motion_relink_max_frame_nodes:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: node_position_um(node) for node_id, node in nodes_by_id.items()}
    node_time = {node_id: int(node["t"]) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def logit_score(s: float) -> float:
        s = float(s)
        s = min(max(s, 1e-6), 1.0 - 1e-6)
        return math.log(s / (1.0 - s))

    def assign_pass(source_ids: list[int], target_ids: list[int], gate_um: float):
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        baseline_cost = np.full_like(cost, big)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        primary_matrix = np.zeros_like(cost)
        ranker_matrix = np.zeros_like(cost)
        appearance_matrix = np.zeros_like(cost)
        predicted_by_source: dict[int, np.ndarray] = {}
        valid_by_row: dict[int, list[int]] = {}

        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            predicted = (
                source_pos
                if prev_pos is None
                else source_pos + cfg.motion_relink_velocity_weight * (source_pos - prev_pos)
            )
            predicted_by_source[source_id] = predicted
            valid_cols = []
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                primary_matrix[i, j] = prob
                src_score = logit_score(nodes_by_id[source_id].get("det_score", 0.5))
                tgt_score = logit_score(nodes_by_id[target_id].get("det_score", 0.5))
                appearance = cfg.appearance_weight * abs(src_score - tgt_score)
                appearance_matrix[i, j] = appearance
                baseline_cost[i, j] = (
                    motion + 0.05 * raw - cfg.motion_relink_learned_bonus * prob + appearance
                )
                valid_cols.append(j)
            valid_by_row[i] = valid_cols

        records = []
        record_locations = []
        for i, source_id in enumerate(source_ids):
            valid_cols = valid_by_row.get(i, [])
            if not valid_cols:
                continue
            ranked_cols_dist = sorted(valid_cols, key=lambda col: (raw_dist[i, col], target_ids[col]))
            rank_by_col_dist = {col: rank + 1 for rank, col in enumerate(ranked_cols_dist)}
            for col in valid_cols:
                target_id = target_ids[col]
                records.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "raw_distance_um": float(raw_dist[i, col]),
                    "motion_distance_um": float(motion_dist[i, col]),
                    "primary_prob": float(primary_matrix[i, col]),
                    "has_learned_edge": bool((source_id, target_id) in learned_edge_probs),
                    "candidate_rank_dist": int(rank_by_col_dist[col]),
                    "candidate_count": int(len(valid_cols)),
                    "predicted_position_um": predicted_by_source[source_id],
                    "context": context,
                })
                record_locations.append((i, col))
        if records:
            ranker_probs = score_records(ranker, nodes_by_id, records)
            for (i, col), prob in zip(record_locations, ranker_probs):
                ranker_matrix[i, col] = float(prob)
            stats["local_ranker_candidates"] += len(records)
            stats["local_ranker_scored"] += len(records)
            stats["local_ranker_probability_milli_sum"] += int(round(float(ranker_probs.sum()) * 1000.0))

        if cfg.ranker_mode == "full_motion_assignment":
            for i, valid_cols in valid_by_row.items():
                if not valid_cols:
                    continue
                stats["local_ranker_full_rows"] += 1
                for col in valid_cols:
                    evidence = (
                        cfg.ranker_full_weight * ranker_matrix[i, col]
                        + cfg.ranker_primary_retain_weight * primary_matrix[i, col]
                    )
                    cost[i, col] = (
                        motion_dist[i, col]
                        + 0.05 * raw_dist[i, col]
                        - cfg.motion_relink_learned_bonus * evidence
                        + appearance_matrix[i, col]
                    )
                    if cfg.use_lookahead:
                        target_id = target_ids[col]
                        residual_um, continuation_count = acceleration_lookahead(
                            source_ids[i],
                            target_id,
                            node_time,
                            ids_by_t,
                            position_um,
                            next_step_gate_um=cfg.motion_relink_relaxed_um,
                        )
                        if continuation_count > 0:
                            stats["forward_lookahead_candidates"] += int(continuation_count)
                            stats["forward_lookahead_supported_edges"] += 1
                        bonus = acceleration_bonus(
                            residual_um,
                            cfg.lookahead_max_accel_um,
                            cfg.lookahead_max_bonus,
                        )
                        if bonus > 0.0:
                            cost[i, col] -= bonus
                            stats["forward_lookahead_bonus_edges"] += 1
                            stats["forward_lookahead_bonus_milli_sum"] += int(round(bonus * 1000.0))
        elif cfg.ranker_mode == "low_margin_top2_rescue":
            cost[:, :] = baseline_cost
            for i, valid_cols in valid_by_row.items():
                if len(valid_cols) < 2:
                    continue
                ordered = sorted(valid_cols, key=lambda col: (baseline_cost[i, col], target_ids[col]))
                best_col, second_col = ordered[0], ordered[1]
                margin = float(baseline_cost[i, second_col] - baseline_cost[i, best_col])
                if margin > cfg.ranker_margin_um:
                    continue
                stats["local_ranker_ambiguous_rows"] += 1
                advantage = float(ranker_matrix[i, second_col] - ranker_matrix[i, best_col])
                if advantage < cfg.ranker_min_advantage:
                    continue
                bonus = min(cfg.ranker_max_bonus, advantage)
                cost[i, second_col] -= bonus
                stats["local_ranker_rescue_adjustments"] += 1
                stats["local_ranker_rescue_bonus_milli_sum"] += int(round(bonus * 1000.0))
        else:
            raise RuntimeError(f"Unknown BIOHUB_LOCAL_RANKER_MODE={cfg.ranker_mode!r}")

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(ranker_matrix[r, c]),
            ))
        return matches

    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches = []
        for pass_name, gate_um in (("tight", cfg.motion_relink_tight_um), ("relaxed", cfg.motion_relink_relaxed_um)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, ranker_prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, ranker_prob))
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, ranker_prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": ranker_prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
                "local_ranker_prob": ranker_prob,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges


# --------------------------------------------------------------------------
# Single-frame gap closing (shared synthetic-node budget)
# --------------------------------------------------------------------------

def _frame_local_spacing(cfg: RunConfig, nodes_by_id, all_ids_by_t, t, density_cache, stats) -> dict[int, float]:
    cached = density_cache.get(t)
    if cached is not None:
        return cached
    frame_ids = all_ids_by_t.get(t, [])
    if len(frame_ids) <= 1:
        result = {node_id: cfg.gap_density_reference_um for node_id in frame_ids}
        density_cache[t] = result
        return result
    positions = np.stack([node_position_um(nodes_by_id[node_id]) for node_id in frame_ids])
    tree = cKDTree(positions)
    query_k = min(len(frame_ids), max(2, cfg.gap_density_neighbors + 1))
    distances, _ = tree.query(positions, k=query_k)
    if distances.ndim == 1:
        distances = distances[:, None]
    result: dict[int, float] = {}
    for idx, node_id in enumerate(frame_ids):
        neighbour_distances = distances[idx, 1:]
        neighbour_distances = neighbour_distances[np.isfinite(neighbour_distances)]
        spacing = (
            float(np.median(neighbour_distances))
            if neighbour_distances.size
            else cfg.gap_density_reference_um
        )
        result[node_id] = spacing
    density_cache[t] = result
    stats["gap_density_nodes_scored"] += len(result)
    return result


def close_frame_gaps(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    test_dir: Path | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not cfg.gap_close or cfg.gap_close_max_gap < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    if "shared_synthetic_node_budget_remaining" not in stats:
        stats["shared_synthetic_node_budget_remaining"] = min(
            cfg.gap_max_added_abs,
            max(1, int(round(len(nodes_by_id) * cfg.gap_max_added_frac))) if cfg.gap_max_added_frac > 0 else 0,
        )
    next_id = fresh_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    test_dir = test_dir if test_dir is not None else Path("/kaggle/input")
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    new_edges: list[dict[str, object]] = []
    density_cache: dict[int, dict[int, float]] = {}

    effective_gap_max = min(cfg.gap_close_max_gap, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_xyz(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_xyz(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = cfg.gap_close_um * (gap + 1)
            d = np.zeros((len(end_ids), len(start_ids)), dtype=np.float64)
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = _frame_local_spacing(cfg, nodes_by_id, all_ids_by_t, t, density_cache, stats)
            target_spacing = _frame_local_spacing(
                cfg, nodes_by_id, all_ids_by_t, t + gap + 1, density_cache, stats)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_span_um(ep, sp)
                    if cfg.gap_density_adaptive:
                        local_spacing = 0.5 * (
                            source_spacing.get(end_ids[i], cfg.gap_density_reference_um)
                            + target_spacing.get(start_ids[j], cfg.gap_density_reference_um)
                        )
                        step_delta = float(np.clip(
                            cfg.gap_density_gain * (local_spacing - cfg.gap_density_reference_um),
                            -cfg.gap_density_max_step_delta_um,
                            cfg.gap_density_max_step_delta_um,
                        ))
                        adaptive_threshold[i, j] = threshold_um + step_delta * (gap + 1)
                        stats["gap_density_step_delta_milli_sum"] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold
            stats["gap_density_candidates_expanded"] += int((adaptive_allowed & ~base_allowed).sum())
            stats["gap_density_candidates_restricted"] += int((base_allowed & ~adaptive_allowed).sum())
            stats["gap_candidates"] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats["gap_density_selected_outside_base"] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                middle_reused = False
                if cfg.gap_reuse_existing:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_span_um(node_xyz(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= cfg.gap_reuse_um:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if stats["shared_synthetic_node_budget_remaining"] <= 0:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_node(
                        cfg, dataset, mid_t, mid_point, test_dir, frame_cache, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    stats["shared_synthetic_node_budget_remaining"] -= 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_span_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_span_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


# --------------------------------------------------------------------------
# Two-frame (strict) gap recovery
# --------------------------------------------------------------------------

def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_long_gaps(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    test_dir: Path | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not cfg.gap2_recovery or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    if "shared_synthetic_node_budget_remaining" not in stats:
        stats["shared_synthetic_node_budget_remaining"] = 2 * min(
            cfg.gap2_max_links_abs,
            max(1, int(round(len(edges) * cfg.gap2_max_links_frac))),
        )

    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * SCALE_ARRAY

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > cfg.gap2_max_total_um or dist / 3.0 > cfg.gap2_max_step_um:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if cfg.gap2_require_context:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if stats["shared_synthetic_node_budget_remaining"] < 2:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * cfg.gap2_frame_frac_cap)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1
        stats["shared_synthetic_node_budget_remaining"] -= 2

    if not selected:
        return nodes_by_id, edges

    next_node_id = fresh_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    test_dir = test_dir if test_dir is not None else Path("/kaggle/input")
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_node(
                cfg, dataset, mid_t, midpoint, test_dir, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": None,
                "distance_um": edge_span_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": None,
            "distance_um": edge_span_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


# --------------------------------------------------------------------------
# Safe divisions (fork candidates with top-k discipline)
# --------------------------------------------------------------------------

def append_safe_divisions(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    if not cfg.safe_divisions or not edges or not nodes_by_id:
        return edges

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * cfg.safe_div_global_frac_cap)))
    topk_cap = cfg.safe_div_topk
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [
            node_id for node_id in child_frame_ids
            if node_id not in incoming and node_id not in used_targets
        ]
        if not source_ids or not candidate_ids:
            continue

        frame_cap = max(1, int(round(len(source_ids) * cfg.safe_div_frame_frac_cap)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_span_um(source, existing_child)
            if child_dist > cfg.safe_div_existing_child_max_um:
                continue
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_span_um(source, candidate)
                if parent_dist > cfg.safe_div_max_um:
                    continue
                sister_dist = edge_span_um(existing_child, candidate)
                if sister_dist > cfg.safe_div_sister_max_um:
                    continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if topk_cap and len(added) >= topk_cap:
                stats["safe_division_skipped_topk"] = stats.get("safe_division_skipped_topk", 0) + 1
                break
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


# --------------------------------------------------------------------------
# Short-track filtering (with optional adaptive rescue) + linefit smoothing
# --------------------------------------------------------------------------

def drop_short_components(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not cfg.filter_short_tracks or cfg.min_track_len <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    t_min_global = t_max_global = None
    if cfg.boundary_track_rescue and nodes_by_id:
        _boundary_t_values = [int(node["t"]) for node in nodes_by_id.values()]
        t_min_global = min(_boundary_t_values)
        t_max_global = max(_boundary_t_values)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        is_long_enough = len(members) >= cfg.min_track_len
        is_division_component = cfg.keep_division_components and has_division
        is_boundary_truncated = False
        if (
            cfg.boundary_track_rescue
            and not is_long_enough
            and not is_division_component
            and len(members) >= cfg.boundary_track_min_len
            and t_min_global is not None
        ):
            member_ts = [int(nodes_by_id[node_id]["t"]) for node_id in members]
            span = max(member_ts) - min(member_ts) + 1
            touches_start = min(member_ts) == t_min_global
            touches_end = max(member_ts) == t_max_global
            if span < cfg.min_track_len and (touches_start or touches_end):
                is_boundary_truncated = True
        if is_long_enough or is_division_component or is_boundary_truncated:
            keep.update(members)
            if is_boundary_truncated:
                stats["boundary_track_rescued_components"] = (
                    stats.get("boundary_track_rescued_components", 0) + 1
                )
                stats["boundary_track_rescued_nodes"] = (
                    stats.get("boundary_track_rescued_nodes", 0) + len(members)
                )

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if cfg.adaptive_short_track_rescue:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= cfg.short_track_rescue_trigger_removed_frac:
            budget = min(
                cfg.short_track_rescue_max_nodes_abs,
                max(0, int(round(len(nodes_by_id) * cfg.short_track_rescue_max_nodes_frac))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < cfg.short_track_rescue_min_len or len(members) >= cfg.min_track_len:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []
                for edge in c_edges:
                    try:
                        prob = float(edge.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(edge.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < cfg.short_track_rescue_min_mean_edge_prob:
                    continue
                if mean_dist > cfg.short_track_rescue_max_mean_edge_dist_um:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(
        1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def smooth_by_linefit(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not cfg.linefit_smooth or cfg.linefit_weight <= 0 or cfg.linefit_window <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(cfg.linefit_weight, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for _ in range(1, cfg.linefit_window + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-_, current))

        current = node_id
        for _ in range(1, cfg.linefit_window + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((_, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array(
            [np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)],
            dtype=np.float64,
        )
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id
