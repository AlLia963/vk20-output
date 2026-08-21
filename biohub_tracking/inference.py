"""Volumetric inference orchestration (Kaggle GPU runtime).

The learned model itself lives in Atria_v8 (support_pack/repo +
edge_predictor_best.pth). This module owns the driver: it applies the
behavior-required runtime patches to the repo prediction script (8-view
detection TTA, calibrated dual-seed fusion, four-view harmonic edge TTA,
detection-score persistence), then shards the video list across GPUs,
waits for the shards and merges the prediction graphs.

The patch manifest is data, not code: every (old, new) pair must match
exactly once, and the patched script is compile-checked before use.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch as _torch

from .config import RunConfig

WEIGHTS_RELATIVE = "weights/unet_transformer/split_0/edge_predictor_best.pth"


# --- Behavior-required runtime patches (must match the repo script exactly) ---

TTA_PATCH = (
    """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4""",
    """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv""",
)


ENSEMBLE_PATCHES = [
    (
        "    downsample: tuple[int, ...] = (1, 4, 4),\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:",
        "    downsample: tuple[int, ...] = (1, 4, 4),\n    secondary_model: UNetNodeTransformer | None = None,\n    secondary_edge_weight: float = 0.0,\n    secondary_detection_weight: float = 0.0,\n    secondary_link_mode: str = \"raw\",\n    secondary_mix_temperature: float = 1.0,\n    secondary_low_margin_max: float = 0.2,\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:",
    ),
    (
        "            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        del imgs",
        "            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        secondary_unet_out = None\n        if secondary_model is not None:\n            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):\n                    primary_det = det_logits[f]\n                    secondary_det = secondary_det_logits[f]\n                    primary_mean = primary_det.mean()\n                    secondary_mean = secondary_det.mean()\n                    primary_scale = primary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    secondary_scale = secondary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_det_aligned = (\n                        (secondary_det - secondary_mean) * scale_ratio + primary_mean\n                    )\n                    det_logits[f] = (\n                        (1.0 - secondary_detection_weight) * primary_det\n                        + secondary_detection_weight * secondary_det_aligned\n                    )\n\n            del secondary_det_logits\n\n        _edge_tta_imgs = imgs",
    ),
    (
        "            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            raw = edge_logits_pair[0]",
        "            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n                if secondary_unet_out is None:\n                    raise RuntimeError(\"Secondary model is loaded but its feature map is missing\")\n                secondary_feat_src = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx], p_coords_src, p_mask_src,\n                )\n                secondary_feat_tgt = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,\n                )\n                secondary_logits_pair = secondary_model.predict_edges(\n                    secondary_feat_src, secondary_feat_tgt,\n                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                    p_pos_src, p_pos_tgt,\n                    p_mask_src, p_mask_tgt,\n                )\n\n                if secondary_link_mode == \"raw\":\n                    secondary_for_mix = secondary_logits_pair\n                    blend_weight = secondary_edge_weight\n                elif secondary_link_mode in {\n                    \"calibrated\", \"adaptive\", \"low_margin_consensus\"\n                }:\n                    primary_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    primary_scale = edge_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)\n                    secondary_scale = secondary_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_for_mix = (\n                        (secondary_logits_pair - secondary_center) * secondary_scale_ratio\n                        + primary_center\n                    )\n                    if secondary_link_mode == \"calibrated\":\n                        blend_weight = secondary_edge_weight\n                    elif secondary_link_mode == \"adaptive\":\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            secondary_margin = secondary_top2.values[0] - secondary_top2.values[1]\n                            local_weight = (\n                                secondary_edge_weight + secondary_margin - primary_margin\n                            ).clamp(0.15, 0.75)\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            local_weight = torch.where(\n                                same_parent,\n                                torch.maximum(\n                                    local_weight,\n                                    torch.full_like(local_weight, secondary_edge_weight),\n                                ),\n                                local_weight,\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = secondary_edge_weight\n                    else:\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            uncertainty = (\n                                (secondary_low_margin_max - primary_margin)\n                                / secondary_low_margin_max\n                            ).clamp(0.0, 1.0)\n                            local_weight = secondary_edge_weight * uncertainty\n                            local_weight = torch.where(\n                                same_parent,\n                                local_weight,\n                                torch.zeros_like(local_weight),\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = 0.0\n                else:\n                    raise ValueError(f\"Unsupported secondary link mode: {secondary_link_mode}\")\n\n                edge_logits_pair = (\n                    (1.0 - blend_weight) * edge_logits_pair\n                    + blend_weight * secondary_for_mix\n                )\n                if secondary_mix_temperature != 1.0:\n                    mixed_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    edge_logits_pair = mixed_center + (\n                        edge_logits_pair - mixed_center\n                    ) / secondary_mix_temperature\n\n            raw = edge_logits_pair[0]",
    ),
    (
        "        del unet_out\n",
        "        del unet_out\n        del _edge_tta_imgs\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n",
    ),
    (
        "    model, window_size, downsample = load_model(weights_path, device)\n    print(",
        "    model, window_size, downsample = load_model(weights_path, device)\n\n    secondary_model = None\n    secondary_weights_text = os.environ.get(\"BIOHUB_SECONDARY_WEIGHTS\", \"\").strip()\n    secondary_edge_weight = float(os.environ.get(\"BIOHUB_SECONDARY_EDGE_WEIGHT\", \"0\"))\n    secondary_detection_weight = float(\n        os.environ.get(\"BIOHUB_SECONDARY_DETECTION_WEIGHT\", \"0\")\n    )\n    secondary_link_mode = os.environ.get(\"BIOHUB_SECONDARY_LINK_MODE\", \"raw\").strip()\n    secondary_mix_temperature = float(\n        os.environ.get(\"BIOHUB_SECONDARY_MIX_TEMPERATURE\", \"1\")\n    )\n    secondary_low_margin_max = float(\n        os.environ.get(\"BIOHUB_SECONDARY_LOW_MARGIN_MAX\", \"0.2\")\n    )\n    edge_candidate_threshold = float(\n        os.environ.get(\"BIOHUB_DUAL_SEED_EDGE_THRESHOLD\", str(cfg.threshold))\n    )\n    if secondary_weights_text:\n        if not 0.0 < secondary_edge_weight < 1.0:\n            raise ValueError(\"BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1\")\n        if not 0.0 <= secondary_detection_weight < 1.0:\n            raise ValueError(\n                \"BIOHUB_SECONDARY_DETECTION_WEIGHT must be in the half-open interval [0, 1)\"\n            )\n        if secondary_link_mode not in {\n            \"raw\", \"calibrated\", \"adaptive\", \"low_margin_consensus\"\n        }:\n            raise ValueError(\n                \"BIOHUB_SECONDARY_LINK_MODE must be raw, calibrated, adaptive, \"\n                \"or low_margin_consensus\"\n            )\n        if not 0.5 <= secondary_mix_temperature <= 2.0:\n            raise ValueError(\"BIOHUB_SECONDARY_MIX_TEMPERATURE must be in [0.5, 2.0]\")\n        if not 0.0 < edge_candidate_threshold < 1.0:\n            raise ValueError(\"BIOHUB_DUAL_SEED_EDGE_THRESHOLD must be strictly between 0 and 1\")\n        if not 0.0 < secondary_low_margin_max <= 1.0:\n            raise ValueError(\"BIOHUB_SECONDARY_LOW_MARGIN_MAX must be in (0, 1]\")\n        secondary_model, secondary_window_size, secondary_downsample = load_model(\n            Path(secondary_weights_text), device,\n        )\n        if secondary_window_size != window_size or secondary_downsample != downsample:\n            raise ValueError(\n                \"Primary and secondary models have incompatible inference grids: \"\n                f\"primary=(window={window_size}, downsample={downsample}), \"\n                f\"secondary=(window={secondary_window_size}, downsample={secondary_downsample})\"\n            )\n        cfg.threshold = edge_candidate_threshold\n        print(\n            f\"Secondary model: {secondary_weights_text} | \"\n            f\"edge weight={secondary_edge_weight:.3f} | \"\n            f\"detection weight={secondary_detection_weight:.3f} | \"\n            f\"link mode={secondary_link_mode} | \"\n            f\"temperature={secondary_mix_temperature:.3f} | \"\n            f\"low-margin max={secondary_low_margin_max:.3f} | \"\n            f\"edge threshold={cfg.threshold:.3f}\",\n            flush=True,\n        )\n\n    print(",
    ),
    (
        "                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n            )",
        "                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n                secondary_model=secondary_model,\n                secondary_edge_weight=secondary_edge_weight,\n                secondary_detection_weight=secondary_detection_weight,\n                secondary_link_mode=secondary_link_mode,\n                secondary_mix_temperature=secondary_mix_temperature,\n                secondary_low_margin_max=secondary_low_margin_max,\n            )",
    ),
]


BIDIRECTIONAL_OLD = """            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)

            if secondary_model is not None:
"""

BIDIRECTIONAL_NEW = """            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)

            _bidirectional_weight = float(
                os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0")
            )
            _edge_tta_mode = os.environ.get("BIOHUB_EDGE_TTA_MODE", "off")
            _edge_tta_views = int(os.environ.get("BIOHUB_EDGE_TTA_VIEWS", "1"))
            if _edge_tta_mode != "js_reliability_log_pool" or _edge_tta_views != 4:
                raise RuntimeError(
                    "Biohub 154 requires BIOHUB_EDGE_TTA_MODE=js_reliability_log_pool "
                    "and BIOHUB_EDGE_TTA_VIEWS=4"
                )

            def _harmonic_probability_from_unet(_view_unet_out):
                _feat_src = model._index_features(
                    _view_unet_out[:, f_idx], p_coords_src, p_mask_src,
                )
                _feat_tgt = model._index_features(
                    _view_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
                )
                _forward = model.predict_edges(
                    _feat_src, _feat_tgt,
                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    p_pos_src, p_pos_tgt,
                    p_mask_src, p_mask_tgt,
                )
                _reverse_native = model.predict_edges(
                    _feat_tgt, _feat_src,
                    p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,
                    p_pos_tgt, p_pos_src,
                    p_mask_tgt, p_mask_src,
                )
                _reverse = _reverse_native.transpose(1, 2)
                _forward_center = _forward.mean(dim=1, keepdim=True)
                _forward_scale = _forward.float().std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                _reverse_center = _reverse.mean(dim=1, keepdim=True)
                _reverse_scale = _reverse.float().std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                _reverse_ratio = (_forward_scale / _reverse_scale).clamp(0.5, 2.0)
                _reverse_aligned = (
                    (_reverse - _reverse_center) * _reverse_ratio.to(_reverse.dtype)
                    + _forward_center
                )
                _forward_prob = torch.softmax(_forward.float(), dim=1).clamp_min(1e-8)
                _reverse_prob = torch.softmax(
                    _reverse_aligned.float(), dim=1
                ).clamp_min(1e-8)
                _harmonic = 1.0 / (
                    (1.0 - _bidirectional_weight) / _forward_prob
                    + _bidirectional_weight / _reverse_prob
                )
                _harmonic = _harmonic / _harmonic.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                return _harmonic, _forward_center, _forward_scale

            _view_probs = []
            _identity_prob, _identity_center, _identity_scale = (
                _harmonic_probability_from_unet(unet_out)
            )
            _view_probs.append(_identity_prob)

            for _tta_kind in ("flip_x", "flip_y", "transpose"):
                if _tta_kind == "flip_x":
                    _tta_input = _edge_tta_imgs.flip(dims=(-1,))
                elif _tta_kind == "flip_y":
                    _tta_input = _edge_tta_imgs.flip(dims=(-2,))
                else:
                    _tta_input = _edge_tta_imgs.transpose(-1, -2)
                _tta_unet, _tta_det_unused = model.encode(_tta_input)
                if _tta_kind == "flip_x":
                    _tta_unet = _tta_unet.flip(dims=(-1,))
                elif _tta_kind == "flip_y":
                    _tta_unet = _tta_unet.flip(dims=(-2,))
                else:
                    _tta_unet = _tta_unet.transpose(-1, -2)
                _tta_prob, _, _ = _harmonic_probability_from_unet(_tta_unet)
                _view_probs.append(_tta_prob)
                del _tta_input, _tta_unet, _tta_det_unused, _tta_prob

            _prob_stack = torch.stack(_view_probs, dim=0).clamp_min(1e-8)

            _consensus_prob = _prob_stack.mean(dim=0).clamp_min(1e-8)
            _js_mix = 0.5 * (
                _prob_stack + _consensus_prob.unsqueeze(0)
            )
            _js_left = (
                _prob_stack
                * (torch.log(_prob_stack) - torch.log(_js_mix))
            ).sum(dim=2)
            _js_right = (
                _consensus_prob.unsqueeze(0)
                * (torch.log(_consensus_prob.unsqueeze(0)) - torch.log(_js_mix))
            ).sum(dim=2)
            _js_distance = 0.5 * (_js_left + _js_right)
            _js_scale = torch.median(_js_distance, dim=0).values.clamp_min(1e-6)
            _view_weight = 1.0 / (
                1.0 + _js_distance / _js_scale.unsqueeze(0)
            )
            _view_weight = _view_weight / _view_weight.sum(
                dim=0, keepdim=True
            ).clamp_min(1e-8)

            _pooled_log = (
                _view_weight.unsqueeze(2) * torch.log(_prob_stack)
            ).sum(dim=0)
            _pooled_prob = torch.softmax(_pooled_log, dim=1).clamp_min(1e-8)
            _pooled_prob = _pooled_prob / _pooled_prob.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

            _pooled_logits = torch.log(_pooled_prob)
            _pooled_center = _pooled_logits.mean(dim=1, keepdim=True)
            _pooled_scale = _pooled_logits.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-4)
            _pooled_ratio = (_identity_scale / _pooled_scale).clamp(0.5, 2.0)
            edge_logits_pair = (
                (_pooled_logits - _pooled_center) * _pooled_ratio
                + _identity_center
            ).to(unet_feat_src.dtype)
            del (
                _view_probs,
                _prob_stack,
                _consensus_prob,
                _js_mix,
                _js_left,
                _js_right,
                _js_distance,
                _js_scale,
                _view_weight,
                _pooled_log,
                _pooled_prob,
                _pooled_logits,
                _identity_prob,
            )

            if secondary_model is not None:
"""


DET_SCORE_PATCHES = [
    (
        "def build_graph(\n    coords: np.ndarray,\n    edges: list[tuple[int, int, float, float]],\n) -> td.graph.InMemoryGraph:",
        "def build_graph(\n    coords: np.ndarray,\n    edges: list[tuple[int, int, float, float]],\n    det_scores: np.ndarray | None = None,\n) -> td.graph.InMemoryGraph:",
    ),
    (
        'for key in ["z", "y", "x"]:',
        'for key in ["z", "y", "x", "det_score"]:',
    ),
    (
        '    node_ids = graph.bulk_add_nodes([\n        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}\n        for t, z, y, x in coords\n    ])',
        '    det_score_default = 0.5\n    node_rows = []\n    for _idx, (_t, _z, _y, _x) in enumerate(coords):\n        _ds = float(det_scores[_idx]) if det_scores is not None else det_score_default\n        node_rows.append({"t": int(_t), "z": float(_z), "y": float(_y), "x": float(_x), "det_score": _ds})\n    node_ids = graph.bulk_add_nodes(node_rows)',
    ),
    (
        "    return np.concatenate([t_col, coords], axis=1).astype(np.int16)\n\n\n@torch.no_grad()\ndef predict_video(",
        "    return np.concatenate([t_col, coords], axis=1).astype(np.int16)\n\n\ndef _detection_scores_for_arr(det_logits, arr):\n    if arr.shape[0] == 0:\n        return np.empty((0,), dtype=np.float32)\n    z = arr[:, 1].astype(np.int64)\n    y = arr[:, 2].astype(np.int64)\n    x = arr[:, 3].astype(np.int64)\n    vals = det_logits[0, z, y, x]\n    return torch.sigmoid(vals).cpu().numpy().astype(np.float32)\n\n\n@torch.no_grad()\ndef predict_video(",
    ),
    (
        "    coord_lists: list[np.ndarray] = []",
        "    coord_lists: list[np.ndarray] = []\n    score_lists: list[np.ndarray] = []",
    ),
    (
        "                coord_lists.append(arr)\n                seen_frames.add(t)",
        "                coord_lists.append(arr)\n                _scores = _detection_scores_for_arr(det_logits[f_idx][0], arr)\n                score_lists.append(_scores)\n                seen_frames.add(t)",
    ),
    (
        '    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)\n    # Scale spatial coords back to original resolution.\n    coords = coords.astype(np.float32)\n    coords[:, 1:] *= ds_arr\n    coords = coords.astype(np.int16)\n    return coords, all_edges',
        '    det_scores = np.concatenate(score_lists) if score_lists else np.empty((0,), dtype=np.float32)\n    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)\n    # Scale spatial coords back to original resolution.\n    coords = coords.astype(np.float32)\n    coords[:, 1:] *= ds_arr\n    coords = coords.astype(np.int16)\n    return coords, all_edges, det_scores',
    ),
    (
        "        coords, edges = predict_video(",
        "        coords, edges, det_scores = predict_video(",
    ),
    (
        "        graph = build_graph(coords, edges)",
        "        graph = build_graph(coords, edges, det_scores=det_scores)",
    ),
]


def apply_repo_patches(repo_dir: Path) -> None:
    script = repo_dir / "scripts" / "predict_unet_transformer.py"
    source = script.read_text(encoding="utf-8")

    old, new = TTA_PATCH
    if old in source:
        source = source.replace(old, new, 1)
        print("TTA patch applied (400ep spatial D4-style)")
    else:
        print("TTA WARNING: block not found - using default 4-way")

    for index, (old, new) in enumerate(ENSEMBLE_PATCHES, start=1):
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"Calibrated dual-seed patch {index} expected one match, found {count}"
            )
        source = source.replace(old, new, 1)
    print("Calibrated dual-seed runtime patch applied")

    bidirectional_weight = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0"))
    if not 0.0 < bidirectional_weight <= 0.35:
        raise ValueError("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT must be in (0, 0.35]")
    count = source.count(BIDIRECTIONAL_OLD)
    if count != 1:
        raise RuntimeError(
            f"Bidirectional edge patch expected one transformed block, found {count}"
        )
    source = source.replace(BIDIRECTIONAL_OLD, BIDIRECTIONAL_NEW, 1)
    print("Four-view trimmed harmonic edge TTA applied | bidirectional weight=", bidirectional_weight)

    for index, (old, new) in enumerate(DET_SCORE_PATCHES, start=1):
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"v8.2 detection-score patch {index} expected one match, found {count}"
            )
        source = source.replace(old, new, 1)
    compile(source, str(script), "exec")
    script.write_text(source, encoding="utf-8")
    print("v8.2 detection-score persistence patch applied")


def list_test_stems(test_dir: Path) -> list[str]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    stems = sorted(path.name[:-5] for path in test_dir.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {test_dir}")
    return stems


def write_test_splits(repo_dir: Path, test_stems: list[str]) -> Path:
    splits_path = repo_dir / "kaggle_test_splits_50ep.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))
    return splits_path


def _visible_cuda_tokens(count: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw != "-1":
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) < count:
            raise RuntimeError(
                f"torch reports {count} CUDA devices but CUDA_VISIBLE_DEVICES={raw!r}"
            )
        return tokens[:count]
    return [str(index) for index in range(count)]


def _prediction_dir_for_method(repo_dir: Path, method: str) -> Path:
    matches = sorted((repo_dir / "predictions").glob(f"*/{method}/split_0"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction directory for {method!r}, found {matches}"
        )
    return matches[0]


def _wait_for_prediction_shards(processes: dict[int, subprocess.Popen], commands: dict[int, list[str]]) -> None:
    while processes:
        failed: tuple[int, int] | None = None
        for shard_index, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del processes[shard_index]
            if return_code != 0:
                failed = (shard_index, return_code)
                break
        if failed is None:
            if processes:
                time.sleep(1.0)
            continue
        failed_index, failed_code = failed
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise subprocess.CalledProcessError(failed_code, commands[failed_index])


def _merge_prediction_shards(repo_dir: Path, method: str, worker_count: int, test_stems: list[str]) -> Path:
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(test_stems)

    for shard_index in range(worker_count):
        shard_method = f"{method}_gpu{shard_index}"
        shard_dir = _prediction_dir_for_method(repo_dir, shard_method)
        expected = set(test_stems[shard_index::worker_count])
        shard_paths = sorted(shard_dir.glob("*.geff"))
        found = {path.stem for path in shard_paths}
        if found != expected:
            raise RuntimeError(
                f"GPU shard {shard_index} output mismatch: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        overlap = seen & found
        if overlap:
            raise RuntimeError(f"Duplicate datasets across GPU shards: {sorted(overlap)}")
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(
            f"Merged GPU shards do not cover the test set: "
            f"missing={sorted(expected_all - seen)}, extra={sorted(seen - expected_all)}"
        )

    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}
    if len(username_roots) != 1:
        raise RuntimeError(f"GPU shards used inconsistent prediction roots: {username_roots}")

    final_root = next(iter(username_roots)) / method
    final_dir = final_root / "split_0"
    staging_dir = final_root / "split_0_dual_gpu_staging"
    if staging_dir.exists():
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir)
        else:
            staging_dir.unlink()
    staging_dir.mkdir(parents=True, exist_ok=False)

    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob("*.geff")):
            destination = staging_dir / source.name
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite duplicate merged output: {destination}")
            shutil.move(str(source), str(destination))

    merged = {path.stem for path in staging_dir.glob("*.geff")}
    if merged != expected_all:
        raise RuntimeError(
            f"Staged prediction directory failed verification: "
            f"missing={sorted(expected_all - merged)}, extra={sorted(merged - expected_all)}"
        )

    if final_dir.exists():
        if final_dir.is_dir():
            shutil.rmtree(final_dir)
        else:
            final_dir.unlink()
    staging_dir.rename(final_dir)
    for shard_dir in shard_dirs:
        shutil.rmtree(shard_dir.parent)
    print(f"Merged {len(merged)} prediction graphs into {final_dir}")
    return final_dir


def run_prediction(
    cfg: RunConfig,
    repo_dir: Path,
    test_dir: Path,
    test_stems: list[str],
    method: str = "unet_transformer",
) -> float:
    """Run inference across available GPUs; returns elapsed seconds."""
    if not _torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."
        )
    print("CUDA device:", _torch.cuda.get_device_name(0))

    apply_repo_patches(repo_dir)

    splits_path = write_test_splits(repo_dir, test_stems)
    predict_cmd = [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--data-dir", str(test_dir),
        "--splits", str(splits_path.name),
        "--split", "0",
        "--weights", WEIGHTS_RELATIVE,
        "--unet-batch-size", str(cfg.unet_batch_size),
        "--det-threshold", str(cfg.det_threshold),
        "--ilp-edge-weight", str(cfg.ilp_edge_weight),
        "--ilp-appearance-weight", str(cfg.ilp_appearance_weight),
        "--ilp-disappearance-weight", str(cfg.ilp_disappearance_weight),
        "--ilp-division-weight", str(cfg.ilp_division_weight),
    ]
    if cfg.use_ilp:
        predict_cmd.append("--use-ilp")
    if cfg.slice:
        predict_cmd.extend(["--slice", cfg.slice])

    start_time = time.time()
    available_gpu_count = _torch.cuda.device_count()
    worker_count = min(2, available_gpu_count, len(test_stems))

    if worker_count >= 2 and not cfg.slice:
        cuda_tokens = _visible_cuda_tokens(worker_count)
        processes: dict[int, subprocess.Popen] = {}
        commands: dict[int, list[str]] = {}
        print(f"Launching {worker_count} independent video shards on CUDA devices {cuda_tokens}")
        for shard_index in range(worker_count):
            shard_method = f"{method}_gpu{shard_index}"
            shard_cmd = [
                *predict_cmd,
                "--method", shard_method,
                "--slice", f"{shard_index}::{worker_count}",
            ]
            shard_env = {**os.environ, "PYTHONPATH": "src"}
            shard_env["CUDA_VISIBLE_DEVICES"] = cuda_tokens[shard_index]
            shard_env["BIOHUB_GPU_SHARD"] = f"{shard_index}/{worker_count}"
            print(
                f"GPU shard {shard_index}: CUDA_VISIBLE_DEVICES={cuda_tokens[shard_index]} | "
                + " ".join(shard_cmd),
                flush=True,
            )
            commands[shard_index] = shard_cmd
            processes[shard_index] = subprocess.Popen(shard_cmd, cwd=repo_dir, env=shard_env)
        _wait_for_prediction_shards(processes, commands)
        _merge_prediction_shards(repo_dir, method, worker_count, test_stems)
    else:
        reason = "SLICE is active" if cfg.slice else f"only {available_gpu_count} CUDA device(s) available"
        print(f"Using single-process prediction because {reason}.")
        print(" ".join(predict_cmd))
        subprocess.run(predict_cmd, cwd=repo_dir, env={**os.environ, "PYTHONPATH": "src"}, check=True)

    predict_seconds = time.time() - start_time
    print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")
    return predict_seconds
