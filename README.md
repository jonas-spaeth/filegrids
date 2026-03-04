filegrids

Example usage
=============

```python
from filegrids import FileGridBuilder

BASE = "/path/to/files"
INIT_DATES = "200[5-9]????"
PATTERN = f"ensemble_forecast_{{init_date={INIT_DATES}}}_{{member}}_{{data_var}}.nc"
TARGET = "/path/to/virtual_dataset.parquet"
fgb = FileGridBuilder(
    BASE,
    PATTERN,
    parse_dates=["init_date"],
)
grid = fgb.build()
vds = fgb.to_virtual_dataset(preprocess=accipy.cepdiag.preprocess_leadtime)
vds.vz.to_kerchunk(TARGET, format="parquet")
```

Docs
====

Local docs build:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

GitHub Pages deployment is handled by `.github/workflows/docs.yml`:

- pull requests: docs are built (validation only)
- pushes to `main`: docs are built, uploaded as Pages artifacts, and deployed
