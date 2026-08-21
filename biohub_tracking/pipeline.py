"""End-to-end graph repair orchestration for one movie."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import RunConfig
from .evidence import reconcile_atomic_ilp_backbone, reassign_by_evidence, reassign_by_ranker
from .graph import edge_priority, edge_span_um
from .ranker import AssociationRanker
from .repair import (
    append_safe_divisions,
    close_frame_gaps,
    drop_short_components,
    recover_long_gaps,
    relink_by_motion,
    smooth_by_linefit,
)


def fresh_stats() -> dict[str, object]:
    return {
        "raw_edges": 0,
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "shared_synthetic_node_budget_initial": 0,
        "safe_division_candidates": 0,
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_synthetic_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
        "local_ranker_candidates": 0,
        "local_ranker_scored": 0,
        "local_ranker_probability_milli_sum": 0,
        "local_ranker_full_rows": 0,
        "local_ranker_ambiguous_rows": 0,
        "local_ranker_rescue_adjustments": 0,
        "local_ranker_rescue_bonus_milli_sum": 0,
        "forward_lookahead_candidates": 0,
        "forward_lookahead_supported_edges": 0,
        "forward_lookahead_bonus_edges": 0,
        "forward_lookahead_bonus_milli_sum": 0,
    }


def repair_graph(
    cfg: RunConfig,
    ranker: AssociationRanker,
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
    test_dir: Path | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    stats = fresh_stats()
    stats["raw_edges"] = len(raw_edges)

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if cfg.enforce_next_frame and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_span_um(source, target)
        edge["distance_um"] = distance_um
        if cfg.output_edge_max_um > 0 and distance_um > cfg.output_edge_max_um:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if cfg.motion_relink:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = relink_by_motion(
            cfg, ranker, nodes_by_id, stats, learned_edge_probs, raw_edges_for_context=edges)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if cfg.single_parent_repair and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_priority(edge) > edge_priority(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if cfg.single_child_repair and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_priority(edge) > edge_priority(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    repair_frame_cache: dict[int, np.ndarray] = {}

    shared_budget_initial = min(
        cfg.shared_node_budget_abs,
        max(1, int(round(len(nodes_by_id) * cfg.shared_node_budget_frac)))
        if cfg.shared_node_budget_frac > 0
        else 0,
    )
    stats["shared_synthetic_node_budget_initial"] = shared_budget_initial
    stats["shared_synthetic_node_budget_remaining"] = shared_budget_initial

    nodes_by_id, edges = close_frame_gaps(
        cfg, nodes_by_id, edges, stats, dataset=dataset, test_dir=test_dir,
        frame_cache=repair_frame_cache)
    nodes_by_id, edges = recover_long_gaps(cfg, nodes_by_id, edges, stats, dataset=dataset, test_dir=test_dir)
    edges = append_safe_divisions(cfg, nodes_by_id, edges, stats)

    if cfg.prune_isolated:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [
                edge for edge in edges
                if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id
            ]

    nodes_by_id, edges = drop_short_components(cfg, nodes_by_id, edges, stats)
    nodes_by_id = smooth_by_linefit(cfg, nodes_by_id, edges, stats)

    if cfg.v72_reassign and edges:
        edges, stats = reassign_by_evidence(
            cfg, nodes_by_id, edges, raw_edges, stats,
        )
        stats["v72_reassign_enabled"] = 1
    else:
        stats["v72_reassign_enabled"] = 0

    if cfg.v74_reassign and edges:
        edges, stats = reassign_by_ranker(
            cfg, ranker, nodes_by_id, edges, raw_edges, stats,
        )
        stats["v74_reassign_enabled"] = 1
    else:
        stats["v74_reassign_enabled"] = 0

    # vk20: final, conservative reconciliation against the post-ILP backbone.
    # This only collapses a 2x2 pair of non-raw ordinary continuations into one
    # raw ILP continuation; divisions and dual-orphan motion rescues are kept.
    if cfg.atomic_ilp_reconcile and edges:
        edges, stats = reconcile_atomic_ilp_backbone(
            cfg, nodes_by_id, edges, raw_edges, stats,
        )
        stats["atomic_ilp_reconcile_enabled"] = 1
    else:
        stats["atomic_ilp_reconcile_enabled"] = 0

    return nodes_by_id, edges, stats
