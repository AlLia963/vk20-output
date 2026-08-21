"""Typed run configuration for v9.0.

v8.2 expressed its champion configuration as a chain of `os.environ[...]`
assignments in the notebook. This module replaces that chain with one typed
object while keeping every *effective* value identical. Each field can still
be overridden by the same ``BIOHUB_*`` environment variable, so the runtime
behavior on Kaggle does not change.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

EXPERIMENT_TAG = "biohub_vk20_atomic_ilp_reconcile_v1"
PRESET = "gap2_joint_node_budget"

# Secondary-model modes.
MODE_VERIFY = "seed314159"  # byte-parity check against v8.2
MODE_OPTIMIZE = "private"   # r=0.50 merged weights (the optimization switch)
PRISTINE_ENV = "BIOHUB_V9_PRISTINE"


def _raw(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _boolean(name: str, default: bool) -> bool:
    return _raw(name, "1" if default else "0") not in ("0", "", "false", "False")


def _number(name: str, default: float, cast=float) -> float:
    try:
        return cast(_raw(name, str(default)))
    except ValueError:
        return default


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One immutable snapshot of every knob the pipeline reads."""

    # Identity / execution mode
    experiment_tag: str
    mode: str                       # MODE_VERIFY or MODE_OPTIMIZE
    pristine: bool                  # True -> force v8.2 champion behavior
    slice: str = ""                 # e.g. ":1" for a Kaggle smoke run

    # Inputs / outputs
    competition: str = "biohub-cell-tracking-during-development"
    test_dir_override: str = ""
    working_dir: str = "/kaggle/working"
    submission_path: str = "submission.csv"
    run_stats_path: str = "run_stats.csv"

    # Detection / ILP
    det_threshold: float = 0.96875
    unet_batch_size: int = 4
    use_ilp: bool = True
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.0
    ilp_disappearance_weight: float = 1.5
    ilp_division_weight: float = 1.0

    # Model fusion / TTA
    bidirectional_edge_weight: float = 0.20
    bidirectional_fusion_mode: str = "harmonic_probability"
    edge_tta_mode: str = "js_reliability_log_pool"
    edge_tta_views: int = 4

    # Local association ranker
    use_ranker: bool = True
    ranker_mode: str = "full_motion_assignment"
    ranker_full_weight: float = 0.85
    ranker_primary_retain_weight: float = 0.15
    ranker_margin_um: float = 0.35
    ranker_min_advantage: float = 0.15
    ranker_max_bonus: float = 0.20

    # Forward-acceleration lookahead
    use_lookahead: bool = True
    lookahead_max_accel_um: float = 4.0
    lookahead_max_bonus: float = 0.20

    # Motion relink (edge repair)
    motion_relink: bool = True
    motion_relink_tight_um: float = 6.0
    motion_relink_relaxed_um: float = 10.0
    motion_relink_velocity_weight: float = 0.5
    motion_relink_learned_bonus: float = 1.0
    motion_relink_max_frame_nodes: int = 2600
    appearance_weight: float = 1.0      # v8.2 detection-score appearance cost

    # Gap close (single-frame)
    gap_close: bool = True
    gap_close_max_gap: int = 1          # effective value (preset 2, capped to 1)
    gap_close_um: float = 5.8
    gap_density_adaptive: bool = True
    gap_density_reference_um: float = 6.5
    gap_density_gain: float = 0.040
    gap_density_max_step_delta_um: float = 0.125
    gap_density_neighbors: int = 3
    gap_reuse_existing: bool = True
    gap_reuse_um: float = 3.2
    gap_max_added_frac: float = 0.05
    gap_max_added_abs: int = 2000
    gap_refine_synthetic: bool = True
    gap_refine_win_z: int = 1
    gap_refine_win_yx: int = 3
    gap_refine_max_shift_um: float = 3.2

    # Gap2 recovery (two-frame) + shared synthetic-node budget
    gap2_recovery: bool = True
    gap2_max_total_um: float = 10.2
    gap2_max_step_um: float = 4.4
    gap2_max_links_frac: float = 0.0045
    gap2_max_links_abs: int = 180
    gap2_require_context: bool = True
    gap2_frame_frac_cap: float = 0.006
    shared_node_budget_frac: float = 0.05
    shared_node_budget_abs: int = 2000

    # Safe divisions (fork cap)
    safe_divisions: bool = True
    safe_div_max_um: float = 4.66
    safe_div_sister_max_um: float = 8.5
    safe_div_existing_child_max_um: float = 7.65
    safe_div_frame_frac_cap: float = 0.0076
    safe_div_global_frac_cap: float = 0.00375
    safe_div_topk: int = 20

    # Short tracks / linefit
    filter_short_tracks: bool = True
    min_track_len: int = 6
    keep_division_components: bool = True
    adaptive_short_track_rescue: bool = False
    boundary_track_rescue: bool = True
    boundary_track_min_len: int = 2
    linefit_smooth: bool = True
    linefit_weight: float = 0.8
    linefit_window: int = 2

    # Output graph guards
    output_edge_max_um: float = 14.0
    enforce_next_frame: bool = True
    single_parent_repair: bool = True
    single_child_repair: bool = False
    prune_isolated: bool = True

    # DeepCenter auxiliary gate (kept for compatibility, veto disabled)
    use_deepcenter_veto: bool = False
    require_deepcenter_veto: bool = False
    deepcenter_expected_epoch: int = 0
    deepcenter_gap_confirm_min_span_um: float = 8.0

    # v7.2 learned-evidence reassign
    v72_reassign: bool = True
    v72_threshold: float = 0.80
    v72_margin: float = 0.10
    v72_add_unassigned: bool = True
    v72_source_continuity_min: float = 0.70
    v72_max_edge_um: float = 0.0

    # v7.4 ranker-evidence reassign
    v74_reassign: bool = True
    v74_r_threshold: float = 0.80
    v74_r_margin: float = 0.10
    v74_add: bool = True
    v74_source_continuity_min: float = 0.70

    # vk20: atomic ILP-backbone reconciliation (post v7.4)
    atomic_ilp_reconcile: bool = True
    atomic_ilp_min_prob: float = 0.50

    # Secondary model (verification vs optimization)
    secondary_edge_weight: float = 0.15
    secondary_detection_weight: float = 0.475
    secondary_link_mode: str = "low_margin_consensus"
    secondary_mix_temperature: float = 1.0
    secondary_low_margin_max: float = 0.35
    dual_seed_edge_threshold: float = 0.48

    # Operational flags
    run_output_diagnostics: bool = False
    allow_pip_install: bool = False
    allow_artifact_fallback: bool = False

    def to_display(self) -> dict[str, Any]:
        return {
            "experiment_tag": self.experiment_tag,
            "mode": self.mode,
            "pristine": self.pristine,
            "det_threshold": self.det_threshold,
            "unet_batch_size": self.unet_batch_size,
            "ilp_edge_weight": self.ilp_edge_weight,
            "ilp_appearance_weight": self.ilp_appearance_weight,
            "ilp_disappearance_weight": self.ilp_disappearance_weight,
            "ilp_division_weight": self.ilp_division_weight,
            "safe_div_topk": self.safe_div_topk,
            "appearance_weight": self.appearance_weight,
            "linefit_window": self.linefit_window,
            "secondary_mode": self.mode,
            "secondary_edge_weight": self.secondary_edge_weight,
            "secondary_detection_weight": self.secondary_detection_weight,
            "secondary_link_mode": self.secondary_link_mode,
            "secondary_low_margin_max": self.secondary_low_margin_max,
            "dual_seed_edge_threshold": self.dual_seed_edge_threshold,
            "atomic_ilp_reconcile": self.atomic_ilp_reconcile,
            "atomic_ilp_min_prob": self.atomic_ilp_min_prob,
        }


def export_runtime_env(cfg: RunConfig) -> None:
    """Mirror the runtime knobs into os.environ for the prediction subprocess."""
    os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = str(cfg.bidirectional_edge_weight)
    os.environ["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = cfg.bidirectional_fusion_mode
    os.environ["BIOHUB_EDGE_TTA_MODE"] = cfg.edge_tta_mode
    os.environ["BIOHUB_EDGE_TTA_VIEWS"] = str(cfg.edge_tta_views)
    os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = str(cfg.secondary_edge_weight)
    os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = str(cfg.secondary_detection_weight)
    os.environ["BIOHUB_SECONDARY_LINK_MODE"] = cfg.secondary_link_mode
    os.environ["BIOHUB_SECONDARY_MIX_TEMPERATURE"] = str(cfg.secondary_mix_temperature)
    os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = str(cfg.secondary_low_margin_max)
    os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = str(cfg.dual_seed_edge_threshold)


def make_config() -> RunConfig:
    """Build the effective run configuration from the environment.

    Defaults are the v8.2 champion values (after the v8.2 preset cell).
    ``BIOHUB_V9_PRISTINE=1`` forces the v8.2 verification behavior and the
    seed314159 secondary, ignoring any optimization switch.
    """

    pristine = _boolean(PRISTINE_ENV, False)
    mode = MODE_OPTIMIZE if _raw("BIOHUB_V9_OPTIMIZE", "0") == "1" else MODE_VERIFY
    if pristine:
        mode = MODE_VERIFY

    def f(name: str, default: float) -> float:
        return _number(name, default, float)

    def i(name: str, default: int) -> int:
        return int(_number(name, float(default), float))

    return RunConfig(
        experiment_tag=_raw("BIOHUB_EXPERIMENT_TAG", EXPERIMENT_TAG),
        mode=mode,
        pristine=pristine,
        slice=_raw("BIOHUB_SLICE", ""),
        test_dir_override=_raw("BIOHUB_TEST_DIR", ""),
        working_dir=_raw("BIOHUB_WORKING_DIR", "/kaggle/working"),
        submission_path=_raw("BIOHUB_SUBMISSION_PATH", "submission.csv"),
        run_stats_path=_raw("BIOHUB_RUN_STATS_PATH", "run_stats.csv"),
        det_threshold=f("BIOHUB_DET_THRESHOLD", 0.96875),
        unet_batch_size=i("BIOHUB_UNET_BATCH_SIZE", 4),
        use_ilp=_boolean("BIOHUB_USE_ILP", True),
        ilp_edge_weight=f("BIOHUB_ILP_EDGE_WEIGHT", -1.0),
        ilp_appearance_weight=f("BIOHUB_ILP_APPEARANCE_WEIGHT", 0.0),
        ilp_disappearance_weight=f("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", 1.5),
        ilp_division_weight=f("BIOHUB_ILP_DIVISION_WEIGHT", 1.0),
        bidirectional_edge_weight=f("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", 0.20),
        bidirectional_fusion_mode=_raw("BIOHUB_BIDIRECTIONAL_FUSION_MODE", "harmonic_probability"),
        edge_tta_mode=_raw("BIOHUB_EDGE_TTA_MODE", "js_reliability_log_pool"),
        edge_tta_views=i("BIOHUB_EDGE_TTA_VIEWS", 4),
        use_ranker=_boolean("BIOHUB_USE_LOCAL_ASSOCIATION_RANKER", True),
        ranker_mode=_raw("BIOHUB_LOCAL_RANKER_MODE", "full_motion_assignment"),
        ranker_full_weight=f("BIOHUB_LOCAL_RANKER_FULL_WEIGHT", 0.85),
        ranker_primary_retain_weight=f("BIOHUB_LOCAL_RANKER_PRIMARY_RETAIN_WEIGHT", 0.15),
        ranker_margin_um=f("BIOHUB_LOCAL_RANKER_MARGIN_UM", 0.35),
        ranker_min_advantage=f("BIOHUB_LOCAL_RANKER_MIN_ADVANTAGE", 0.15),
        ranker_max_bonus=f("BIOHUB_LOCAL_RANKER_MAX_BONUS", 0.20),
        use_lookahead=_boolean("BIOHUB_USE_FORWARD_ACCELERATION_LOOKAHEAD", True),
        lookahead_max_accel_um=f("BIOHUB_FORWARD_LOOKAHEAD_MAX_ACCEL_UM", 4.0),
        lookahead_max_bonus=f("BIOHUB_FORWARD_LOOKAHEAD_MAX_BONUS", 0.20),
        motion_relink=_boolean("BIOHUB_OUTPUT_MOTION_RELINK", True),
        motion_relink_tight_um=f("BIOHUB_MOTION_RELINK_TIGHT_UM", 6.0),
        motion_relink_relaxed_um=f("BIOHUB_MOTION_RELINK_RELAXED_UM", 10.0),
        motion_relink_velocity_weight=f("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", 0.5),
        motion_relink_learned_bonus=f("BIOHUB_MOTION_RELINK_LEARNED_BONUS", 1.0),
        motion_relink_max_frame_nodes=i("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", 2600),
        appearance_weight=f("BIOHUB_V82_APPEARANCE_W", 1.0 if not pristine else 1.0),
        gap_close=_boolean("BIOHUB_OUTPUT_GAP_CLOSE", True),
        gap_close_max_gap=i("BIOHUB_GAP_CLOSE_MAX_GAP", 1),
        gap_close_um=f("BIOHUB_GAP_CLOSE_UM", 5.8),
        gap_density_adaptive=_boolean("BIOHUB_GAP_DENSITY_ADAPTIVE", True),
        gap_density_reference_um=f("BIOHUB_GAP_DENSITY_REFERENCE_UM", 6.5),
        gap_density_gain=f("BIOHUB_GAP_DENSITY_GAIN", 0.040),
        gap_density_max_step_delta_um=f("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", 0.125),
        gap_density_neighbors=i("BIOHUB_GAP_DENSITY_NEIGHBORS", 3),
        gap_reuse_existing=_boolean("BIOHUB_GAP_CLOSE_REUSE_EXISTING", True),
        gap_reuse_um=f("BIOHUB_GAP_CLOSE_REUSE_UM", 3.2),
        gap_max_added_frac=f("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", 0.05),
        gap_max_added_abs=i("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", 2000),
        gap_refine_synthetic=_boolean("BIOHUB_GAP_REFINE_SYNTHETIC", True),
        gap_refine_win_z=i("BIOHUB_GAP_REFINE_WIN_Z", 1),
        gap_refine_win_yx=i("BIOHUB_GAP_REFINE_WIN_YX", 3),
        gap_refine_max_shift_um=f("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", 3.2),
        gap2_recovery=_boolean("BIOHUB_OUTPUT_GAP2_RECOVERY", True),
        gap2_max_total_um=f("BIOHUB_GAP2_MAX_TOTAL_UM", 10.2),
        gap2_max_step_um=f("BIOHUB_GAP2_MAX_STEP_UM", 4.4),
        gap2_max_links_frac=f("BIOHUB_GAP2_MAX_LINKS_FRAC", 0.0045),
        gap2_max_links_abs=i("BIOHUB_GAP2_MAX_LINKS_ABS", 180),
        gap2_require_context=_boolean("BIOHUB_GAP2_REQUIRE_CONTEXT", True),
        gap2_frame_frac_cap=f("BIOHUB_GAP2_FRAME_FRAC_CAP", 0.006),
        shared_node_budget_frac=f("BIOHUB_SHARED_SYNTHETIC_NODE_BUDGET_FRAC", 0.05),
        shared_node_budget_abs=i("BIOHUB_SHARED_SYNTHETIC_NODE_BUDGET_ABS", 2000),
        safe_divisions=_boolean("BIOHUB_OUTPUT_SAFE_DIVISIONS", True),
        safe_div_max_um=f("BIOHUB_SAFE_DIV_MAX_UM", 4.66),
        safe_div_sister_max_um=f("BIOHUB_SAFE_DIV_SISTER_MAX_UM", 8.5),
        safe_div_existing_child_max_um=f("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", 7.65),
        safe_div_frame_frac_cap=f("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", 0.0076),
        safe_div_global_frac_cap=f("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", 0.00375),
        safe_div_topk=i("BIOHUB_SAFE_DIV_TOP_K", 20),
        filter_short_tracks=_boolean("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", True),
        min_track_len=i("BIOHUB_OUTPUT_MIN_TRACK_LEN", 6),
        keep_division_components=_boolean("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", True),
        adaptive_short_track_rescue=_boolean("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", False),
        boundary_track_rescue=_boolean("BIOHUB_OUTPUT_BOUNDARY_TRACK_RESCUE", not pristine),
        boundary_track_min_len=i("BIOHUB_OUTPUT_BOUNDARY_TRACK_MIN_LEN", 2),
        linefit_smooth=_boolean("BIOHUB_OUTPUT_LINEFIT_SMOOTH", True),
        linefit_weight=f("BIOHUB_OUTPUT_LINEFIT_WEIGHT", 0.8),
        linefit_window=i("BIOHUB_OUTPUT_LINEFIT_WINDOW", 2),
        output_edge_max_um=f("BIOHUB_OUTPUT_EDGE_MAX_UM", 14.0),
        enforce_next_frame=_boolean("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", True),
        single_parent_repair=_boolean("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", True),
        single_child_repair=_boolean("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", False),
        prune_isolated=_boolean("BIOHUB_OUTPUT_PRUNE_ISOLATED", True),
        use_deepcenter_veto=_boolean("BIOHUB_USE_DEEPCENTER_VETO", False),
        require_deepcenter_veto=_boolean("BIOHUB_REQUIRE_DEEPCENTER_VETO", False),
        deepcenter_expected_epoch=i("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", 0),
        deepcenter_gap_confirm_min_span_um=f("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", 8.0),
        v72_reassign=_boolean("BIOHUB_V72_REASSIGN", True),
        v72_threshold=f("BIOHUB_V72_THRESHOLD", 0.80),
        v72_margin=f("BIOHUB_V72_MARGIN", 0.10),
        v72_add_unassigned=_boolean("BIOHUB_V72_ADD_UNASSIGNED", True),
        v72_source_continuity_min=f("BIOHUB_V72_SOURCE_CONTINUITY_MIN", 0.70),
        v72_max_edge_um=f("BIOHUB_V72_MAX_EDGE_UM", 0.0),
        v74_reassign=_boolean("BIOHUB_V74_REASSIGN", True),
        v74_r_threshold=f("BIOHUB_V74_R_THR", 0.80),
        v74_r_margin=f("BIOHUB_V74_R_MARGIN", 0.10),
        v74_add=_boolean("BIOHUB_V74_ADD", True),
        v74_source_continuity_min=f("BIOHUB_V74_SOURCE_CONTINUITY_MIN", 0.70),
        atomic_ilp_reconcile=(False if pristine else _boolean("BIOHUB_ATOMIC_ILP_RECONCILE", True)),
        atomic_ilp_min_prob=f("BIOHUB_ATOMIC_ILP_MIN_PROB", 0.50),
        secondary_edge_weight=f("BIOHUB_SECONDARY_EDGE_WEIGHT", 0.15),
        secondary_detection_weight=f("BIOHUB_SECONDARY_DETECTION_WEIGHT", 0.475),
        secondary_link_mode=_raw("BIOHUB_SECONDARY_LINK_MODE", "low_margin_consensus"),
        secondary_mix_temperature=f("BIOHUB_SECONDARY_MIX_TEMPERATURE", 1.0),
        secondary_low_margin_max=f("BIOHUB_SECONDARY_LOW_MARGIN_MAX", 0.35),
        dual_seed_edge_threshold=f("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", 0.48),
        run_output_diagnostics=_boolean("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", False),
        allow_pip_install=_boolean("BIOHUB_ALLOW_PIP_INSTALL", False),
        allow_artifact_fallback=_boolean("BIOHUB_ALLOW_ARTIFACT_FALLBACK", False),
    )
