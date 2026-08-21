"""Submission writer, per-movie statistics and clean-graph audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODE_VERIFY, RunConfig
from .graph import load_track_graph
from .pipeline import repair_graph
from .ranker import AssociationRanker

CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

V82_REFERENCE_HASH = "A47F8FA2D48A775CD1B693B5E7F2EF82D1758FEBD84F39902D8E8CB180CC38E2"
V82_SIGNATURE = {"rows": 243267, "nodes": 123567, "edges": 119700, "divisions": 611}

def graph_contracts(graph) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    nodes_by_id: dict[int, dict[str, object]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        det_score = row.get("det_score") if hasattr(row, "get") else None
        nodes_by_id[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
            "det_score": 0.5 if det_score is None else float(det_score),
        }
    raw_edges: list[dict[str, object]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
        raw_edges.append({
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": None if edge_prob is None else float(edge_prob),
        })
    return nodes_by_id, raw_edges


def write_submission(
    cfg: RunConfig,
    ranker: AssociationRanker,
    geffs: list[Path],
    test_dir: Path,
    test_stems: list[str],
    experiment_tag: str,
    predict_seconds: float,
    submission_path: Path,
    run_stats_path: Path,
) -> None:
    stats_rows: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    row_id = 0
    total_nodes = 0
    total_edges = 0

    with submission_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for geff_path in geffs:
            dataset = geff_path.stem
            seen_datasets.add(dataset)
            graph = load_track_graph(geff_path)
            nodes_by_id, raw_edges = graph_contracts(graph)
            raw_node_count = len(nodes_by_id)
            nodes_by_id, edges, filter_stats = repair_graph(
                cfg, ranker, nodes_by_id, raw_edges, dataset=dataset, test_dir=test_dir)
            if not nodes_by_id:
                raise AssertionError(f"{dataset}: post-processing removed every node")

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": max(0, int(round(float(node["z"])))),
                    "y": max(0, int(round(float(node["y"])))),
                    "x": max(0, int(round(float(node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1

            division_sources: dict[int, int] = {}
            for edge in edges:
                source_id = int(edge["source_id"])
                target_id = int(edge["target_id"])
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise AssertionError(f"{dataset}: dangling edge after filtering")
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source_id,
                    "target_id": target_id,
                })
                row_id += 1
                division_sources[source_id] = division_sources.get(source_id, 0) + 1

            node_count = len(nodes_by_id)
            edge_count = len(edges)
            total_nodes += node_count
            total_edges += edge_count
            stats_rows.append({
                "dataset": dataset,
                "raw_nodes": raw_node_count,
                "nodes": node_count,
                "raw_edges": filter_stats["raw_edges"],
                "edges": edge_count,
                "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
                "edge_to_node_ratio": edge_count / max(node_count, 1),
                "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
                **filter_stats,
            })

    expected_datasets = set(test_stems)
    missing_datasets = sorted(expected_datasets - seen_datasets)
    extra_datasets = sorted(seen_datasets - expected_datasets)
    if missing_datasets or extra_datasets:
        raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
    assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
    assert total_nodes > 0, "No node rows produced"

    header = submission_path.open().readline().strip().split(",")
    assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

    stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
    stats["predict_minutes_total"] = predict_seconds / 60.0
    stats["experiment_tag"] = experiment_tag
    stats.to_csv(run_stats_path, index=False)
    print(f"Wrote {submission_path} with {row_id:,} rows")
    print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
    print(f"Wrote {run_stats_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_submission(
    cfg: RunConfig,
    submission_path: Path,
    run_stats_path: Path,
    audit_json_path: Path,
) -> dict:
    if not submission_path.is_file():
        raise FileNotFoundError(f"Submission file was not generated: {submission_path}")
    submission = pd.read_csv(submission_path)
    missing_columns = sorted(set(CSV_COLUMNS) - set(submission.columns))
    if missing_columns:
        raise AssertionError(f"Missing submission columns: {missing_columns}")

    node_rows = submission.loc[submission["row_type"].eq("node")].copy()
    edge_rows = submission.loc[submission["row_type"].eq("edge")].copy()
    for _col in ["node_id", "t", "z", "y", "x"]:
        node_rows[_col] = pd.to_numeric(node_rows[_col], errors="coerce")
    for _col in ["source_id", "target_id"]:
        edge_rows[_col] = pd.to_numeric(edge_rows[_col], errors="coerce")

    nonfinite_coordinate_nodes = int(
        (~np.isfinite(node_rows[["t", "z", "y", "x"]].to_numpy(dtype=float))).any(axis=1).sum())
    negative_time_nodes = int((node_rows["t"] < 0).sum())
    duplicate_nodes = int(node_rows.duplicated(["dataset", "node_id"], keep=False).sum())
    duplicate_edges = int(edge_rows.duplicated(["dataset", "source_id", "target_id"], keep=False).sum())

    node_lookup: dict[tuple[str, int], dict[str, int]] = {}
    node_id_datasets: dict[int, set[str]] = defaultdict(set)
    for _row in node_rows.itertuples(index=False):
        if not np.isfinite(float(_row.node_id)):
            continue
        _dataset = str(_row.dataset)
        _node_id = int(_row.node_id)
        node_lookup[(_dataset, _node_id)] = {
            "t": int(_row.t), "z": int(_row.z), "y": int(_row.y), "x": int(_row.x),
        }
        node_id_datasets[_node_id].add(_dataset)

    missing_edge_endpoints = same_frame_edges = backward_edges = non_adjacent_edges = 0
    cross_dataset_edge_suspects = 0
    indegree = Counter()
    outdegree = Counter()
    for _row in edge_rows.itertuples(index=False):
        _dataset = str(_row.dataset)
        if not np.isfinite(float(_row.source_id)) or not np.isfinite(float(_row.target_id)):
            missing_edge_endpoints += 1
            continue
        _source_id = int(_row.source_id)
        _target_id = int(_row.target_id)
        _source = node_lookup.get((_dataset, _source_id))
        _target = node_lookup.get((_dataset, _target_id))
        if _source is None or _target is None:
            missing_edge_endpoints += 1
            if (
                (_source is None and node_id_datasets.get(_source_id, set()) - {_dataset})
                or (_target is None and node_id_datasets.get(_target_id, set()) - {_dataset})
            ):
                cross_dataset_edge_suspects += 1
            continue
        _dt = int(_target["t"]) - int(_source["t"])
        if _dt == 0:
            same_frame_edges += 1
        if _dt < 0:
            backward_edges += 1
        if _dt != 1:
            non_adjacent_edges += 1
        outdegree[(_dataset, _source_id)] += 1
        indegree[(_dataset, _target_id)] += 1

    maximum_indegree = max(indegree.values(), default=0)
    maximum_outdegree = max(outdegree.values(), default=0)
    divisions = sum(1 for value in outdegree.values() if value == 2)
    outdegree_over_2 = sum(1 for value in outdegree.values() if value > 2)
    indegree_over_1 = sum(1 for value in indegree.values() if value > 1)

    submission_sha256 = sha256_file(submission_path)
    audit = {
        "notebook": "biohub-cell-tracking_v9.0.ipynb",
        "experiment": cfg.experiment_tag,
        "mode": cfg.mode,
        "rows": int(len(submission)),
        "nodes": int(len(node_rows)),
        "edges": int(len(edge_rows)),
        "divisions": int(divisions),
        "datasets": int(submission["dataset"].nunique()),
        "negative_time_nodes": negative_time_nodes,
        "nonfinite_coordinate_nodes": nonfinite_coordinate_nodes,
        "duplicate_nodes": duplicate_nodes,
        "duplicate_edges": duplicate_edges,
        "missing_edge_endpoints": missing_edge_endpoints,
        "same_frame_edges": same_frame_edges,
        "backward_edges": backward_edges,
        "non_adjacent_edges": non_adjacent_edges,
        "cross_dataset_edge_suspects": cross_dataset_edge_suspects,
        "maximum_indegree": int(maximum_indegree),
        "maximum_outdegree": int(maximum_outdegree),
        "indegree_over_1": int(indegree_over_1),
        "outdegree_over_2": int(outdegree_over_2),
        "submission_sha256": submission_sha256,
        "v82_reference_hash": V82_REFERENCE_HASH,
        "known_output_signature": V82_SIGNATURE,
    }
    audit["baseline_output_match"] = all(
        audit[key] == expected for key, expected in V82_SIGNATURE.items())
    audit["baseline_output_delta"] = {
        key: int(audit[key] - expected) for key, expected in V82_SIGNATURE.items()}
    audit["candidate_output_changed"] = submission_sha256.lower() != V82_REFERENCE_HASH.lower()

    _invariant_keys = [
        "negative_time_nodes", "nonfinite_coordinate_nodes", "duplicate_nodes",
        "duplicate_edges", "missing_edge_endpoints", "same_frame_edges",
        "backward_edges", "non_adjacent_edges", "cross_dataset_edge_suspects",
        "indegree_over_1", "outdegree_over_2",
    ]
    audit["clean_graph_audit"] = bool(all(audit[key] == 0 for key in _invariant_keys))

    if cfg.mode == MODE_VERIFY:
        audit["verification_gate"] = submission_sha256.lower() == V82_REFERENCE_HASH.lower()
        audit["recommendation"] = (
            "PASS: v9.0 restructure is byte-identical to v8.2. "
            "Next: run optimization mode (BIOHUB_V9_OPTIMIZE=1)."
            if audit["verification_gate"]
            else "FAIL: submission hash differs from v8.2. Review the port before optimizing."
        )
    else:
        audit["recommendation"] = (
            "Optimization run: compare the official score against v8.2 (0.921); "
            "keep only if it improves."
        )
    audit_json_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"\nAudit JSON: {audit_json_path}")
    return audit
