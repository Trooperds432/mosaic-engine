"""One-off evaluation of the filing-text features after the data_text.py backfill: rebuild the broad panel with them and
run the discovery-only feature test (feature_test.py: pre-holdout dates, joint BH-FDR with every earlier test).
They enter the models only through a full re-validation with a new frozen recipe."""
import json
import sys

import config as C
from features import build_panel_chunked
import feature_test

TEXT = ["text_sim_cos", "text_sim_jac", "text_sim_rf", "text_len_chg", "text_neg", "text_d_neg", "text_d_unc", "text_d_lit",
        "wiki_attn_abn", "wiki_attn_level"]

if __name__ == "__main__":
    _lock = C.hold_lock()
    prof = sys.argv[1] if len(sys.argv) > 1 else "broad"
    panel, meta = build_panel_chunked(profile=prof)
    print({f: round(float(panel[f].notna().mean()), 3) for f in TEXT}, flush=True)
    panel.to_parquet(C.PROC / f"panel_{prof}.parquet", index=False)
    (C.PROC / f"panel_meta_{prof}.json").write_text(json.dumps(meta, indent=1))
    del panel
    out = feature_test.run(prof, TEXT)
    (C.REPORTS / f"text_features_{prof}.json").write_text(json.dumps(out, indent=1, default=float))
