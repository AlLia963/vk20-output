"""biohub_tracking v9.0 — behavior-preserving restructure of the v8.2 pipeline.

The package is developed here as plain Python modules and inlined into the
Kaggle notebook by build_v90.py at build time. Nothing here depends on
/kaggle paths except where explicitly documented, so the foundation layers
(config / resources / deps) are fully testable on a local machine.
"""

from .config import RunConfig, make_config, MODE_VERIFY, MODE_OPTIMIZE

__all__ = ["RunConfig", "make_config", "MODE_VERIFY", "MODE_OPTIMIZE"]
