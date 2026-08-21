"""Evidence-aware edge reassignment passes.

v7.2 learned-evidence reassign (S1 swap / S2 add) and the v7.4 ranker-
evidence arbitration are implemented here as pure graph passes that take
the RunConfig and the loaded ranker explicitly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import RunConfig
from .graph import SCALE_ARRAY
from .ranker import AssociationRanker, build_ranker_context, score_records


def _learned_map(raw_edges: list[dict[str, object]]) -> dict[tuple[int, int], float]:
    learned: dict[tuple[int, int], float] = {}
    for edge in raw_edges:
        p = edge.get("edge_prob")
        if p is None:
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if np.isfinite(p):
            key = (int(edge["source_id"]), int(edge["target_id"]))
            learned[key] = max(learned.get(key, float("-inf")), p)
    return learned


def _voxel_distance(nodes_by_id, s: int, t: int) -> float:
    return float(np.linalg.norm(
        np.asarray([float(nodes_by_id[s]["z"]), float(nodes_by_id[s]["y"]),
                    float(nodes_by_id[s]["x"])], float)
        - np.asarray([float(nodes_by_id[t]["z"]), float(nodes_by_id[t]["y"]),
                      float(nodes_by_id[t]["x"])], float)
    ))


def reassign_by_evidence(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    raw_edges: list[dict[str, object]],
    stats: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Swap/add edges by learned edge probability, single in-edge per target."""
    stats = stats if stats is not None else {}
    n0 = len(edges)
    learned_probs = _learned_map(raw_edges)

    def prob_of(edge) -> float:
        return float(learned_probs.get((int(edge["source_id"]), int(edge["target_id"])), 0.0))

    succ = defaultdict(list)
    for e in edges:
        succ[int(e["source_id"])].append(int(e["target_id"]))
    in_edges = {}
    for e in edges:
        in_edges[int(e["target_id"])] = e

    cand_in = defaultdict(list)
    for (s, t), p in learned_probs.items():
        s, t = int(s), int(t)
        if p >= cfg.v72_threshold and s in nodes_by_id and t in nodes_by_id:
            if cfg.v72_max_edge_um > 0:
                if _voxel_distance(nodes_by_id, s, t) > cfg.v72_max_edge_um:
                    continue
            cand_in[t].append((float(p), s, t))
    for t in cand_in:
        cand_in[t].sort(reverse=True)

    def out_degree(s) -> int:
        return len(succ.get(int(s), []))

    best_in_prob: dict[int, float] = {}
    for (_s, t), p in learned_probs.items():
        t = int(t)
        if p > best_in_prob.get(t, 0.0):
            best_in_prob[t] = float(p)

    n_swap = 0
    n_add = 0
    touched_targets: set[int] = set()

    for t in sorted(in_edges):
        cur = in_edges[t]
        cur_s = int(cur["source_id"])
        cur_p = prob_of(cur)
        if cur_p >= cfg.v72_threshold:
            continue
        best = None
        for p, s, _tt in cand_in.get(t, []):
            if s == cur_s:
                continue
            if p - cur_p < cfg.v72_margin:
                continue
            if out_degree(s) >= 2:
                continue
            if best_in_prob.get(s, 0.0) < cfg.v72_source_continuity_min:
                continue
            best = (p, s)
            break
        if best is None:
            continue
        p, s = best
        new_edge = {
            "source_id": s,
            "target_id": t,
            "edge_prob": p,
            "distance_um": _voxel_distance(nodes_by_id, s, t),
            "v72_reassign": 1,
        }
        for i, e in enumerate(edges):
            if e is cur:
                edges[i] = new_edge
                break
        succ[cur_s].remove(t)
        succ[s].append(t)
        in_edges[t] = new_edge
        touched_targets.add(t)
        n_swap += 1
        stats.setdefault("v72_swaps", []).append((cur_s, t, s, round(cur_p, 4), round(p, 4)))

    if cfg.v72_add_unassigned:
        add_count_by_source = defaultdict(int)
        for t in sorted(cand_in):
            if t in in_edges:
                continue
            for p, s, _tt in cand_in[t]:
                if out_degree(s) >= 2:
                    continue
                if add_count_by_source[s] >= 1:
                    continue
                if best_in_prob.get(s, 0.0) < cfg.v72_source_continuity_min:
                    continue
                edges.append({
                    "source_id": s,
                    "target_id": t,
                    "edge_prob": p,
                    "distance_um": _voxel_distance(nodes_by_id, s, t),
                    "v72_add": 1,
                })
                succ[s].append(t)
                in_edges[t] = edges[-1]
                add_count_by_source[s] += 1
                n_add += 1
                stats.setdefault("v72_adds", []).append((s, t, round(p, 4)))
                break

    by_target = defaultdict(list)
    by_source = defaultdict(list)
    for e in edges:
        by_target[int(e["target_id"])].append(e)
        by_source[int(e["source_id"])].append(e)
    drop = set()
    for t, es in by_target.items():
        if len(es) > 1:
            es.sort(key=lambda e: (prob_of(e), -float(e.get("distance_um", 0.0))), reverse=True)
            for e in es[1:]:
                drop.add(id(e))
    for s, es in by_source.items():
        if len(es) > 2:
            es.sort(key=lambda e: (prob_of(e), -float(e.get("distance_um", 0.0))), reverse=True)
            for e in es[2:]:
                drop.add(id(e))
    if drop:
        edges = [e for e in edges if id(e) not in drop]
        stats["v72_enforcement_dropped"] = len(drop)
    else:
        stats["v72_enforcement_dropped"] = 0

    stats["v72_edges_before"] = n0
    stats["v72_edges_after"] = len(edges)
    stats["v72_swaps_n"] = n_swap
    stats["v72_adds_n"] = n_add
    return edges, stats


def reassign_by_ranker(
    cfg: RunConfig,
    ranker: AssociationRanker,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    raw_edges: list[dict[str, object]],
    stats: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Per-target arbitration by local-ranker evidence (post v7.2 pass)."""
    stats = stats if stats is not None else {}
    if ranker is None:
        stats["v74_skipped"] = 1
        return edges, stats

    learned = _learned_map(raw_edges)

    succ = defaultdict(list)
    pred = defaultdict(list)
    for e in edges:
        succ[int(e["source_id"])].append(int(e["target_id"]))
        pred[int(e["target_id"])].append(int(e["source_id"]))
    position = {
        n: np.asarray([float(v["z"]), float(v["y"]), float(v["x"])], float)
        for n, v in nodes_by_id.items()
    }
    t_of = {n: int(v["t"]) for n, v in nodes_by_id.items()}
    ids_by_t = defaultdict(list)
    for n in nodes_by_id:
        ids_by_t[t_of[n]].append(n)
    for ids in ids_by_t.values():
        ids.sort()

    ctx = build_ranker_context(nodes_by_id, edges)
    relaxed = cfg.motion_relink_relaxed_um
    vel_w = cfg.motion_relink_velocity_weight

    records = []
    loc = []
    for t, srcs in sorted(ids_by_t.items()):
        tgts = ids_by_t.get(t + 1, [])
        if not srcs or not tgts:
            continue
        for s in srcs:
            s_pos = position[s]
            parents = pred.get(s)
            prev_pos = position[parents[0]] if parents else None
            predicted = s_pos if prev_pos is None else s_pos + vel_w * (s_pos - prev_pos)
            valid = []
            for tg in tgts:
                raw = float(np.linalg.norm((position[tg] - s_pos) * SCALE_ARRAY))
                if raw > relaxed:
                    continue
                valid.append((raw, tg))
            if not valid:
                continue
            valid.sort(key=lambda x: (x[0], x[1]))
            rank_by = {tg: i + 1 for i, (_r, tg) in enumerate(valid)}
            for raw, tg in valid:
                motion = float(np.linalg.norm((position[tg] - predicted) * SCALE_ARRAY))
                records.append({
                    "source_id": s,
                    "target_id": tg,
                    "raw_distance_um": raw,
                    "motion_distance_um": motion,
                    "primary_prob": float(learned.get((s, tg), 0.0)),
                    "has_learned_edge": bool((s, tg) in learned),
                    "candidate_rank_dist": int(rank_by[tg]),
                    "candidate_count": int(len(valid)),
                    "predicted_position_um": predicted,
                    "context": ctx,
                })
                loc.append((s, tg))
    if not records:
        stats["v74_scored"] = 0
        return edges, stats
    try:
        probs = score_records(ranker, nodes_by_id, records)
    except Exception as exc:
        stats["v74_error"] = f"{type(exc).__name__}: {exc}"
        stats["v74_scored"] = 0
        return edges, stats
    prob_map = {k: float(p) for k, p in zip(loc, probs)}
    stats["v74_scored"] = len(prob_map)

    best_in_prob: dict[int, float] = {}
    for (_s, t), p in learned.items():
        t = int(t)
        if p > best_in_prob.get(t, 0.0):
            best_in_prob[t] = float(p)

    in_edges = {}
    for e in edges:
        in_edges[int(e["target_id"])] = e

    def out_degree(s) -> int:
        return len(succ.get(int(s), []))

    cand_in = defaultdict(list)
    for (s, t), p in prob_map.items():
        cand_in[int(t)].append((float(p), int(s), int(t)))
    for t in cand_in:
        cand_in[t].sort(reverse=True)

    n_swap = n_add = 0
    stats["v74_swaps"] = []
    stats["v74_adds"] = []
    for t in sorted(in_edges):
        cur = in_edges[t]
        cur_s = int(cur["source_id"])
        cur_p = prob_map.get((cur_s, t), 0.0)
        best = None
        for p, s, _tt in cand_in.get(t, []):
            if s == cur_s:
                continue
            if p < cfg.v74_r_threshold or (p - cur_p) < cfg.v74_r_margin:
                continue
            if out_degree(s) >= 2:
                continue
            if best_in_prob.get(s, 0.0) < cfg.v74_source_continuity_min:
                continue
            best = (p, s)
            break
        if best is None:
            continue
        p, s = best
        new_edge = {"source_id": s, "target_id": t, "edge_prob": p, "v74_reassign": 1}
        for i, e in enumerate(edges):
            if e is cur:
                edges[i] = new_edge
                break
        succ[cur_s].remove(t)
        succ[s].append(t)
        in_edges[t] = new_edge
        stats["v74_swaps"].append((cur_s, t, s, round(cur_p, 4), round(p, 4)))
        n_swap += 1

    if cfg.v74_add:
        add_count = defaultdict(int)
        for t in sorted(cand_in):
            if t in in_edges:
                continue
            for p, s, _tt in cand_in[t]:
                if p < cfg.v74_r_threshold:
                    continue
                if out_degree(s) >= 2 or add_count[s] >= 1:
                    continue
                if best_in_prob.get(s, 0.0) < cfg.v74_source_continuity_min:
                    continue
                edges.append({"source_id": s, "target_id": t, "edge_prob": p, "v74_add": 1})
                succ[s].append(t)
                in_edges[t] = edges[-1]
                add_count[s] += 1
                stats["v74_adds"].append((s, t, round(p, 4)))
                n_add += 1
                break

    by_target = defaultdict(list)
    by_source = defaultdict(list)
    for e in edges:
        by_target[int(e["target_id"])].append(e)
        by_source[int(e["source_id"])].append(e)
    drop = set()
    for es in by_target.values():
        if len(es) > 1:
            es.sort(
                key=lambda e: prob_map.get((int(e["source_id"]), int(e["target_id"])), 0.0),
                reverse=True,
            )
            for e in es[1:]:
                drop.add(id(e))
    for es in by_source.values():
        if len(es) > 2:
            es.sort(
                key=lambda e: prob_map.get((int(e["source_id"]), int(e["target_id"])), 0.0),
                reverse=True,
            )
            for e in es[2:]:
                drop.add(id(e))
    if drop:
        edges = [e for e in edges if id(e) not in drop]
    stats["v74_swaps_n"] = n_swap
    stats["v74_adds_n"] = n_add
    stats["v74_dropped"] = len(drop)
    return edges, stats


# ---------------------------------------------------------------------------
# vk20 atomic ILP-backbone reconciliation
# ---------------------------------------------------------------------------
def reconcile_atomic_ilp_backbone(
    cfg: RunConfig,
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    raw_edges: list[dict[str, object]],
    stats: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Resolve a narrow 2x2 continuation conflict in favour of the ILP backbone.

    The saved raw GEFF is the *post-ILP solution graph*.  Motion relinking later
    replaces that graph with a dense 1:1 assignment and can therefore turn one
    ILP continuation ``s->t`` into two unsupported continuations ``s->u`` and
    ``v->t``.  That is exactly the pattern that creates a stolen target while
    also over-linking a source that ILP had chosen to terminate.

    This pass is deliberately narrower than the old generic raw-edge restore:

    * raw ``s->t`` must be an ordinary one-to-one ILP continuation (raw
      out-degree(s)==1 and raw in-degree(t)==1);
    * the raw learned probability must be >= ``atomic_ilp_min_prob``;
    * the current graph must contain exactly one different outgoing edge from
      ``s`` and exactly one different incoming edge to ``t``;
    * both current edges must be non-raw, and neither current source may be a
      division source (out-degree must be exactly one).

    We then remove the two unsupported edges atomically and add the one raw ILP
    edge.  Net edge count decreases by one, providing an explicit birth/death
    outcome instead of forcing a complete 1:1 assignment.  Motion-only rescue
    remains untouched when both ILP endpoints were unassigned, and all current
    division sources are protected.
    """
    stats = stats if stats is not None else {}
    if not cfg.atomic_ilp_reconcile or not edges or not raw_edges:
        stats["atomic_ilp_candidates"] = 0
        stats["atomic_ilp_collapses"] = 0
        stats["atomic_ilp_removed_nonraw"] = 0
        stats["atomic_ilp_added_raw"] = 0
        stats["atomic_ilp_prob_milli_sum"] = 0
        return edges, stats

    learned = _learned_map(raw_edges)

    # Raw graph degrees are computed only on valid surviving consecutive edges.
    # We intentionally do not use final smoothed distance as a gate: the edge was
    # already accepted by the ILP graph before smoothing, and linefit can move the
    # displayed centroid after topology has been fixed.
    raw_pair_edge: dict[tuple[int, int], dict[str, object]] = {}
    raw_in: dict[int, set[tuple[int, int]]] = defaultdict(set)
    raw_out: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for edge in raw_edges:
        s = int(edge["source_id"])
        t = int(edge["target_id"])
        if s not in nodes_by_id or t not in nodes_by_id:
            continue
        if int(nodes_by_id[t]["t"]) != int(nodes_by_id[s]["t"]) + 1:
            continue
        pair = (s, t)
        # Saved ILP GEFF should not contain duplicates.  If it does, retain the
        # highest-probability copy deterministically.
        prev = raw_pair_edge.get(pair)
        if prev is not None:
            prev_p = float(learned.get(pair, 0.0))
            try:
                this_p = float(edge.get("edge_prob", 0.0))
            except (TypeError, ValueError):
                this_p = 0.0
            if this_p <= prev_p:
                continue
        raw_pair_edge[pair] = edge
        raw_out[s].add(pair)
        raw_in[t].add(pair)

    # Current graph is expected to be duplicate-free; use pair sets so updates
    # are O(1) even on the 70k-node dense movie.
    current: set[tuple[int, int]] = set()
    current_edge: dict[tuple[int, int], dict[str, object]] = {}
    cur_in: dict[int, set[tuple[int, int]]] = defaultdict(set)
    cur_out: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for edge in edges:
        pair = (int(edge["source_id"]), int(edge["target_id"]))
        if pair in current:
            continue
        current.add(pair)
        current_edge[pair] = edge
        cur_out[pair[0]].add(pair)
        cur_in[pair[1]].add(pair)

    raw_pairs = set(raw_pair_edge)

    def remove_pair(pair: tuple[int, int]) -> None:
        if pair not in current:
            return
        current.remove(pair)
        cur_out[pair[0]].discard(pair)
        cur_in[pair[1]].discard(pair)

    def add_pair(pair: tuple[int, int], edge: dict[str, object]) -> None:
        if pair in current:
            return
        current.add(pair)
        current_edge[pair] = edge
        cur_out[pair[0]].add(pair)
        cur_in[pair[1]].add(pair)

    candidates = []
    for pair, raw_edge in raw_pair_edge.items():
        if pair in current:
            continue
        s, t = pair
        p = float(learned.get(pair, 0.0))
        if p < float(cfg.atomic_ilp_min_prob):
            continue
        if len(raw_out.get(s, ())) != 1 or len(raw_in.get(t, ())) != 1:
            continue
        candidates.append((p, s, t, raw_edge))
    # High-confidence conflicts first; IDs make ties deterministic.
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    stats["atomic_ilp_candidates"] = len(candidates)

    collapses = 0
    removed_nonraw = 0
    prob_milli_sum = 0
    for p, s, t, raw_edge in candidates:
        pair = (s, t)
        incoming = list(cur_in.get(t, ()))
        outgoing = list(cur_out.get(s, ()))
        if len(incoming) != 1 or len(outgoing) != 1:
            continue
        old_in = incoming[0]       # v -> t
        old_out = outgoing[0]     # s -> u
        if old_in in raw_pairs:
            continue
        # raw_out(s)==1 and pair is the only raw edge, so old_out is necessarily
        # non-raw.  Protect the competing parent if it is a division source.
        competing_parent = int(old_in[0])
        if len(cur_out.get(competing_parent, ())) != 1:
            continue

        # Atomic 2->1 collapse.  Roll back if an unexpected collision appears.
        removed = [old_in] if old_in == old_out else [old_in, old_out]
        old_objects = [(old, current_edge[old]) for old in removed]
        for old in removed:
            remove_pair(old)
        if cur_in.get(t) or cur_out.get(s):
            for old, obj in old_objects:
                add_pair(old, obj)
            continue

        new_edge = dict(raw_edge)
        new_edge["source_id"] = s
        new_edge["target_id"] = t
        new_edge["edge_prob"] = p
        new_edge["atomic_ilp_reconcile"] = 1
        add_pair(pair, new_edge)
        collapses += 1
        removed_nonraw += len(removed)
        prob_milli_sum += int(round(p * 1000.0))

    # Preserve original ordering for untouched edges, then append restored raw
    # edges in deterministic candidate order.  Graph semantics are unchanged by
    # edge row order because the result is already degree-valid.
    result = []
    emitted: set[tuple[int, int]] = set()
    for edge in edges:
        pair = (int(edge["source_id"]), int(edge["target_id"]))
        if pair in current and pair not in emitted:
            result.append(current_edge[pair])
            emitted.add(pair)
    for _p, s, t, _raw_edge in candidates:
        pair = (s, t)
        if pair in current and pair not in emitted:
            result.append(current_edge[pair])
            emitted.add(pair)
    # Defensive fallback for any surviving pair not reached above.
    for pair in sorted(current):
        if pair not in emitted:
            result.append(current_edge[pair])
            emitted.add(pair)

    stats["atomic_ilp_collapses"] = collapses
    stats["atomic_ilp_removed_nonraw"] = removed_nonraw
    stats["atomic_ilp_added_raw"] = collapses
    stats["atomic_ilp_prob_milli_sum"] = prob_milli_sum
    stats["atomic_ilp_edges_before"] = len(edges)
    stats["atomic_ilp_edges_after"] = len(result)
    return result, stats
