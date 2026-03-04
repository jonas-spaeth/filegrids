from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import filegrids.cli as cli


def test_nc_to_virtualizarr_cli_calls_builder(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class FakeVZ:
        def to_kerchunk(self, target: str, format: str) -> None:
            captured["target"] = target
            captured["format"] = format

    class FakeVDS:
        def __init__(self) -> None:
            self.vz = FakeVZ()

    class FakeBuilder:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def build(self):
            captured["build_called"] = True
            return {"sav300": {"files": ["f1.nc"]}}

        def to_virtual_dataset(self, *, rename_data_vars: bool):
            captured["rename_data_vars"] = rename_data_vars
            return FakeVDS()

    monkeypatch.setattr(cli, "FileGridBuilder", FakeBuilder)

    base = tmp_path / "stage"
    base.mkdir()
    target = tmp_path / "out.parquet"
    pattern = "{exp_id}_ens_{number}_{data_var}_{init_date}.nc"

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "nc-to-virtualizarr",
            str(base),
            pattern,
            str(target),
            "--rename-data-vars",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["build_called"] is True
    assert captured["rename_data_vars"] is True
    assert captured["target"] == str(target)
    assert captured["format"] == "parquet"

    init_kwargs = captured["init_kwargs"]
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["base"] == str(base)
    assert init_kwargs["pattern"] == pattern
    assert init_kwargs["data_var_tags"] == ("data_var",)
    assert init_kwargs["parse_dates"] == ("init_date",)
