Quickstart
==========

Install with docs and runtime extras:

.. code-block:: bash

   pip install -e ".[xarray,virtual,docs]"

Build a file grid:

.. code-block:: python

   from filegrids import FileGridBuilder

   fgb = FileGridBuilder(
       base="/path/to/stage",
       pattern="{exp_id}_ens_{number=[1-2]}_{data_var}_{init_date=20090[1-3]??}.nc",
       data_var_tags=["data_var"],
       parse_dates=["init_date"],
   )

   grid = fgb.build()
   vds = fgb.to_virtual_dataset(rename_data_vars=True)
   vds.vz.to_kerchunk("/path/to/out.parquet", format="parquet")

Build docs locally:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html
