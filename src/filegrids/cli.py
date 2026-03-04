from __future__ import annotations

import click

from .core import FileGridBuilder


@click.group()
def main() -> None:
    """filegrids command line tools."""


@main.command("nc-to-virtualizarr")
@click.argument("base", type=click.Path(exists=True, file_okay=False, path_type=str))
@click.argument("pattern", type=str)
@click.argument("target", type=click.Path(path_type=str))
@click.option(
    "--data-var-tag",
    "data_var_tags",
    multiple=True,
    default=("data_var",),
    show_default=True,
    help="Tag(s) used to split files by variable name.",
)
@click.option(
    "--parse-date",
    "parse_dates",
    multiple=True,
    default=("init_date",),
    show_default=True,
    help="Tag(s) parsed to datetime.",
)
@click.option(
    "--rename-data-vars/--no-rename-data-vars",
    default=False,
    show_default=True,
    help="Rename opened variable to the data-var bucket name when needed.",
)
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["parquet", "json"], case_sensitive=False),
    default="parquet",
    show_default=True,
    help="Output format for kerchunk index.",
)
def nc_to_virtualizarr(
    base: str,
    pattern: str,
    target: str,
    data_var_tags: tuple[str, ...],
    parse_dates: tuple[str, ...],
    rename_data_vars: bool,
    out_format: str,
) -> None:
    """Convert NetCDF files matched by pattern into a virtualizarr kerchunk index."""
    fgb = FileGridBuilder(
        base=base,
        pattern=pattern,
        data_var_tags=data_var_tags,
        parse_dates=parse_dates,
    )
    grid = fgb.build()
    vds = fgb.to_virtual_dataset(rename_data_vars=rename_data_vars)
    vds.vz.to_kerchunk(target, format=out_format)
    click.echo(f"Wrote {len(grid)} variable bucket(s) to {target}")
