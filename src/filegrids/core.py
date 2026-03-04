from __future__ import annotations
import os, re, glob, fnmatch
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Iterable, Optional, Callable
import html as _html

import numpy as np
import pandas as pd


class FileGridBuilder:
    """
    Build nested grids of file paths from a filename pattern, e.g.:
        "{exp_id=iwm*}_ens_{number}_{data_var}_{date}.nc"

    Supports:
    - glob defaults per tag, e.g. {exp_id=iwm*}
    - automatic number/date parsing
    - optional transform functions per tag
    - nested list output suitable for xarray.open_mfdataset(combine="nested")

    If you do NOT include a data-var tag in the pattern (e.g. no {data_var}),
    the builder will NOT try to infer variable names; it will open files as-is
    and let xarray expose whatever variables they contain.
    """

    def __init__(
        self,
        base: str,
        pattern: str,
        data_var_tags: Iterable[str] = ("data_var",),
        parse_dates: Iterable[str] = (),
        concat_dim_tags: Optional[List[str]] = None,
        transform_funcs: Optional[Dict[str, Callable[[Any], Any]]] = None,
    ) -> None:
        self.base = base
        self.pattern = pattern
        self.data_var_tags = tuple(data_var_tags)
        self.parse_dates = set(parse_dates)
        self.transform_funcs = transform_funcs or {}

        self._last_grid = None  # cache of most recent grid
        self._last_stats = None  # cache of most recent stats

        # Parse literal parts and tags (with optional default glob)
        self.tags: List[str] = []
        self.tag_globs: Dict[str, Optional[str]] = {}
        self._lits: List[str] = []
        i = 0
        for m in re.finditer(r"{(\w+)(?:=([^}]+))?}", pattern):
            self._lits.append(pattern[i : m.start()])
            name, g = m.group(1), m.group(2)
            self.tags.append(name)
            self.tag_globs[name] = g  # may be None
            i = m.end()
        self._lits.append(pattern[i:])

        # Keep only data_var tags that actually appear in the pattern
        tag_set = set(self.tags)
        self.data_var_tags = tuple(t for t in self.data_var_tags if t in tag_set)

        # Default concat dims = all non–data-var tags in pattern order
        auto_dims = [t for t in self.tags if t not in self.data_var_tags]
        self.dim_tags = (
            list(concat_dim_tags) if concat_dim_tags is not None else auto_dims
        )

        self.regex = self._compile_regex()
        self.glob_pattern = self._compile_glob()

    # -------------------------------------------------------------------------
    # pattern compilation
    # -------------------------------------------------------------------------
    def _compile_glob(self) -> str:
        parts = []
        for lit, tag in zip(self._lits[:-1], self.tags):
            parts.append(lit)
            parts.append(
                self.tag_globs[tag] if self.tag_globs[tag] is not None else "*"
            )
        parts.append(self._lits[-1])
        return "".join(parts)

    def _compile_regex(self) -> re.Pattern:
        chunks = []
        for lit, tag in zip(self._lits[:-1], self.tags):
            chunks.append(re.escape(lit))
            g = self.tag_globs[tag]
            if g is None:
                chunks.append(rf"(?P<{tag}>.+?)")
            else:
                rgx = fnmatch.translate(g)
                # strip anchors added by fnmatch.translate
                rgx = rgx.replace("(?s:", "").rstrip(")\\Z").rstrip("$")
                chunks.append(rf"(?P<{tag}>{rgx})")
        chunks.append(re.escape(self._lits[-1]))
        return re.compile("^" + "".join(chunks) + "$")

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _maybe_number(x: Any) -> Any:
        if isinstance(x, (int, float, np.number)):
            return x
        for t in (int, float):
            try:
                return t(x)
            except Exception:
                continue
        return x

    def _coerce(self, tag: str, val: str) -> Any:
        """Convert strings to numbers, dates, or apply custom transform functions."""
        if tag in self.parse_dates:
            val = pd.to_datetime(str(val), errors="raise")
        else:
            val = self._maybe_number(val)

        # Apply custom transformation if provided
        if tag in self.transform_funcs:
            try:
                val = self.transform_funcs[tag](val)
            except Exception as e:
                raise ValueError(
                    f"Error transforming value for tag '{tag}': {val!r} -> {e}"
                ) from e

        return val

    @staticmethod
    def _sorted_unique(vals: Iterable[Any]) -> List[Any]:
        vals = list({v for v in vals})

        def key(v):
            if isinstance(v, (pd.Timestamp, np.datetime64, datetime)):
                return (0, pd.Timestamp(v).to_datetime64())
            if isinstance(v, (int, float, np.number)):
                return (1, float(v))
            return (2, str(v))

        return sorted(vals, key=key)

    # -------------------------------------------------------------------------
    # tree building
    # -------------------------------------------------------------------------
    @staticmethod
    def _tree_insert(tree: Dict, keys: List[Any], path: str) -> None:
        cur = tree
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = path

    def _tree_to_nested_list(
        self, tree: Dict, dim_values: List[List[Any]], depth: int = 0
    ):
        if depth == len(dim_values):
            return tree
        return [
            self._tree_to_nested_list(tree[v], dim_values, depth + 1)
            for v in dim_values[depth]
        ]

    # -------------------------------------------------------------------------
    # build
    # -------------------------------------------------------------------------
    def build(
        self, select: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build file grid.

        Returns:
          {
            <bucket_name>: {
              "paths": nested list of paths,
              "dims": list of dimension names,
              "coords": {dim: list of values},
              "files": flat list of all files
            },
            ...
          }

        If no data_var tag is present in the pattern, <bucket_name> will be "__all__"
        and the dataset will include whatever variables the files contain.
        """
        files = glob.glob(os.path.join(self.base, self.glob_pattern))
        matches = []
        for f in files:
            m = self.regex.match(os.path.basename(f))
            if not m:
                continue
            info = {k: self._coerce(k, v) for k, v in m.groupdict().items()}

            # optional filtering
            if select:
                ok = True
                for tag, cond in select.items():
                    if tag not in info:
                        continue
                    v = info[tag]
                    ok = cond(v) if callable(cond) else v in cond
                    if not ok:
                        break
                if not ok:
                    continue

            info["_file"] = f
            matches.append(info)

        by_var = defaultdict(list)
        for info in matches:
            if self.data_var_tags:
                var_key = tuple(info[tag] for tag in self.data_var_tags)
            else:
                var_key = ("__all__",)  # single bucket, no data_var tag in filenames
            by_var[var_key].append(info)

        result: Dict[str, Dict[str, Any]] = {}
        for var_key, items in by_var.items():
            coords = {
                d: self._sorted_unique([it[d] for it in items]) for d in self.dim_tags
            }
            tree: Dict = {}
            for it in items:
                self._tree_insert(tree, [it[d] for d in self.dim_tags], it["_file"])
            dim_values = [coords[d] for d in self.dim_tags]
            nested = self._tree_to_nested_list(tree, dim_values)

            name = var_key[0] if len(var_key) == 1 else "_".join(map(str, var_key))
            result[name] = {
                "paths": nested,
                "dims": list(self.dim_tags),
                "coords": coords,
                "files": [it["_file"] for it in items],
            }

        # cache & summarize for reprs
        self._last_grid = result
        self._last_stats = self._summarize(result)
        return result

    # -------------------------------------------------------------------------
    # xarray integration
    # -------------------------------------------------------------------------
    def to_dataset(
        self,
        grid: Optional[Dict[str, Dict[str, Any]]] = None,
        rename_data_vars: bool = False,
        **open_kwargs,
    ):
        try:
            import xarray as xr
        except Exception as e:
            raise ImportError(
                "xarray is required for to_dataset(). Install filegrids[xarray]."
            ) from e

        grid = self.build() if grid is None else grid
        datasets = []
        for bucket_name, meta in grid.items():
            ds = xr.open_mfdataset(
                meta["paths"], combine="nested", concat_dim=meta["dims"], **open_kwargs
            )
            # Only try to rename if we actually split by data-var tags
            if self.data_var_tags and rename_data_vars:
                if bucket_name not in ds.data_vars and len(ds.data_vars) == 1:
                    ds = ds.rename({list(ds.data_vars)[0]: bucket_name})
            # assign coords from grid
            coords = {
                d: (d, meta["coords"][d])
                for d in meta["dims"]
                if d in meta.get("coords", {})
            }
            if coords:
                ds = ds.assign_coords(**coords)
            datasets.append(ds)
        return xr.combine_by_coords(datasets)

    # -------------------------------------------------------------------------
    # virtualizarr integration
    # -------------------------------------------------------------------------
    def to_virtual_dataset(
        self,
        grid: Optional[Dict[str, Dict[str, Any]]] = None,
        rename_data_vars: bool = False,
        **vkws,
    ):
        """
        Open a virtual multi-file dataset using virtualizarr/obstore.
        """
        try:
            import xarray as xr
            import virtualizarr.xarray  # registers chunk manager
            from virtualizarr import open_virtual_mfdataset
            from virtualizarr.parsers import HDFParser
            from virtualizarr.registry import ObjectStoreRegistry
            from obstore.store import LocalStore
        except Exception as e:
            raise ImportError(
                "virtualizarr/obstore/xarray required. Install filegrids[virtual]."
            ) from e

        grid = self.build() if grid is None else grid
        parser = vkws.pop("parser", None) or HDFParser()
        registry = vkws.pop("registry", None)

        def to_urls(node):
            if isinstance(node, list):
                return [to_urls(n) for n in node]
            return "file://" + os.path.abspath(node)

        if registry is None:
            store = LocalStore(prefix=self.base)
            url_map = {}
            for meta in grid.values():
                stack = [meta["paths"]]
                while stack:
                    n = stack.pop()
                    if isinstance(n, list):
                        stack.extend(n)
                    else:
                        url_map["file://" + os.path.abspath(n)] = store
            registry = ObjectStoreRegistry(url_map)

        vds_list = []
        for bucket_name, meta in grid.items():
            nested_urls = to_urls(meta["paths"])
            vds = open_virtual_mfdataset(
                urls=nested_urls,
                parser=parser,
                registry=registry,
                combine="nested",
                concat_dim=meta["dims"],
                **vkws,
            )
            coords = {
                d: (d, meta["coords"][d])
                for d in meta["dims"]
                if d in meta.get("coords", {})
            }
            if coords:
                vds = vds.assign_coords(**coords)
            # Only try to rename when we split by data-var tags
            if self.data_var_tags and rename_data_vars:
                if bucket_name not in vds.data_vars and len(vds.data_vars) == 1:
                    vds = vds.rename({list(vds.data_vars)[0]: bucket_name})
            vds_list.append(vds)

        return xr.combine_by_coords(vds_list)

    # ---------------- nice reprs ----------------
    def __repr__(self) -> str:
        base = self.base
        pat = self.pattern
        dims = self.dim_tags
        dvt = self.data_var_tags if self.data_var_tags else "(none)"
        tags = self.tags
        if not self._last_stats:
            return f"FileGridBuilder(base='{base}', pattern='{pat}', dims={dims}, data_var_tags={dvt}, tags={tags})"
        s = self._last_stats
        return (
            f"FileGridBuilder(base='{base}', pattern='{pat}', dims={dims}, "
            f"data_var_tags={dvt}, tags={tags}, last_build: vars={s['n_vars']}, files={s['n_files']})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML repr for notebooks with expandable coords."""
        base = _html.escape(str(self.base))
        pat = _html.escape(str(self.pattern))
        dims = ", ".join(_html.escape(str(d)) for d in self.dim_tags)
        dvt = ", ".join(_html.escape(str(t)) for t in self.data_var_tags) or "(none)"
        tags = ", ".join(_html.escape(str(t)) for t in self.tags)

        if not self._last_stats or not self._last_grid:
            return f"""
            <div style="font-family:system-ui,Segoe UI,Arial;line-height:1.35;
                        border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;">
              <div style="font-weight:600;margin-bottom:6px;">FileGridBuilder</div>
              <table style="font-size:13px;">
                <tr><td style="padding-right:8px;color:#6b7280;">base</td><td>{base}</td></tr>
                <tr><td style="padding-right:8px;color:#6b7280;">pattern</td><td><code>{pat}</code></td></tr>
                <tr><td style="padding-right:8px;color:#6b7280;">dims</td><td>{dims}</td></tr>
                <tr><td style="padding-right:8px;color:#6b7280;">data_var_tags</td><td>{dvt}</td></tr>
                <tr><td style="padding-right:8px;color:#6b7280;">tags</td><td>{tags}</td></tr>
              </table>
              <div style="margin-top:6px;color:#6b7280;font-size:12px;">(call <code>.build()</code> to see stats)</div>
            </div>
            """

        s = self._last_stats
        g = self._last_grid

        def format_coords(coords: dict) -> str:
            rows = []
            for k, v in coords.items():
                vals = ", ".join(_html.escape(str(x)) for x in v[:8])
                if len(v) > 8:
                    vals += " …"
                rows.append(
                    f"<tr><td style='padding-right:6px;color:#6b7280;'>{_html.escape(str(k))}</td>"
                    f"<td>{vals}</td></tr>"
                )
            return (
                "<table style='font-size:12px;margin-top:4px;'>"
                + "".join(rows)
                + "</table>"
            )

        vars_html = "".join(
            f"<details style='margin-bottom:4px;'><summary><code>{_html.escape(str(var))}</code> "
            f"— {meta['n_files'] if 'n_files' in meta else len(meta.get('files', []))} files</summary>"
            f"{format_coords(g[var]['coords'])}</details>"
            for var, meta in s["vars"].items()
        )

        return f"""
        <div style="font-family:system-ui,Segoe UI,Arial;line-height:1.35;
                    border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;">
          <div style="font-weight:600;margin-bottom:6px;">FileGridBuilder</div>
          <table style="font-size:13px;margin-bottom:8px;">
            <tr><td style="padding-right:8px;color:#6b7280;">base</td><td>{base}</td></tr>
            <tr><td style="padding-right:8px;color:#6b7280;">pattern</td><td><code>{pat}</code></td></tr>
            <tr><td style="padding-right:8px;color:#6b7280;">dims</td><td>{dims}</td></tr>
            <tr><td style="padding-right:8px;color:#6b7280;">data_var_tags</td><td>{dvt}</td></tr>
            <tr><td style="padding-right:8px;color:#6b7280;">tags</td><td>{tags}</td></tr>
          </table>
          <div style="font-size:13px;margin-bottom:4px;">
            <strong>Last build:</strong> {s["n_vars"]} variables, {s["n_files"]} unique files
          </div>
          <div style="margin-top:6px;font-size:13px;">
            <strong>Variables and coords</strong>
            <div style="margin-top:4px;">{vars_html}</div>
          </div>
        </div>
        """

    # ---------------- helper for stats ----------------
    @staticmethod
    def _summarize(grid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        vars_stats = {}
        all_files = set()
        for var, meta in grid.items():
            fset = set(meta.get("files", []))
            vars_stats[var] = {"n_files": len(fset)}
            all_files.update(fset)
        return {"n_vars": len(grid), "n_files": len(all_files), "vars": vars_stats}
