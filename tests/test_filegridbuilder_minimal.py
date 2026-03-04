from __future__ import annotations

import pandas as pd

from filegrids import FileGridBuilder


def test_build_minimal_synthetic_grid(make_synthetic_cepdiag_files):
    base = make_synthetic_cepdiag_files()
    pattern = "{exp_id}_ens_{number=[1-2]}_{data_var}_{init_date=20090[1-3]??}.nc"

    fgb = FileGridBuilder(
        str(base),
        pattern,
        data_var_tags=["data_var"],
        parse_dates=["init_date"],
    )
    grid = fgb.build()

    assert list(grid) == ["sav300"]
    meta = grid["sav300"]
    assert meta["dims"] == ["exp_id", "number", "init_date"]
    assert meta["coords"]["exp_id"] == ["iwm0", "iwm1"]
    assert meta["coords"]["number"] == [1, 2]
    assert meta["coords"]["init_date"] == [
        pd.Timestamp("2009-01-01"),
        pd.Timestamp("2009-02-01"),
        pd.Timestamp("2009-03-01"),
    ]
    assert len(meta["files"]) == 12
