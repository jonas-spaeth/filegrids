from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def make_synthetic_cepdiag_files(tmp_path: Path):
    """Create tiny synthetic CEPDIAG-like NetCDF files for FileGridBuilder tests."""

    def _make(
        *,
        exp_ids: tuple[str, ...] = ("iwm0", "iwm1"),
        numbers: tuple[int, ...] = (1, 2),
        data_vars: tuple[str, ...] = ("sav300",),
        init_dates: tuple[str, ...] = ("20090101", "20090201", "20090301"),
    ) -> Path:
        # Keep files small but preserve expected structure and coordinate names.
        time = np.array(["2009-01-01", "2009-01-02"], dtype="datetime64[ns]")
        latitude = np.array([-0.5, 0.5], dtype=np.float32)
        longitude = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        for exp_id in exp_ids:
            for number in numbers:
                for data_var in data_vars:
                    for init_date in init_dates:
                        path = (
                            tmp_path
                            / f"{exp_id}_ens_{number}_{data_var}_{init_date}.nc"
                        )
                        data = np.full(
                            (time.size, latitude.size, longitude.size),
                            fill_value=float(number),
                            dtype=np.float32,
                        )
                        ds = xr.Dataset(
                            data_vars={
                                data_var: (
                                    ("time", "latitude", "longitude"),
                                    data,
                                )
                            },
                            coords={
                                "time": time,
                                "latitude": latitude,
                                "longitude": longitude,
                            },
                        )
                        ds.to_netcdf(path, engine="h5netcdf")
        return tmp_path

    return _make
