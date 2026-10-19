"""The `compute` tool: a short Python snippet, written by the model, run on this study's cells in an isolated process.

It fills the gaps no fixed tool covers (arithmetic across results, gating cells, custom gene sets, finer per-state
statistics, small models) without giving up provenance: the snippet runs here, its code is kept in the call's args, and
its output becomes a tool result with an id like any other, so a number in a report can only come from code that ran.

Isolation (bubblewrap): no network, nothing writable but a private /tmp, and only the study data, Python and its
libraries visible. If the sandbox cannot start, the tool refuses; it never runs a snippet unprotected.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TIMEOUT_S = 180
MAX_CODE = 6000
MAX_OUT = 6000
MEM_BYTES = 24 * 1024 ** 3

RUNNER = r'''
import io, json, sys, contextlib, traceback
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
D = sys.argv[1]
obs = pd.read_pickle(f"{D}/obs.pkl")
X = sp.load_npz(f"{D}/X.npz").tocsc()
genes = {g: i for i, g in enumerate(json.load(open(f"{D}/genes.json")))}
umi = obs["umi"].to_numpy(float)
KB = json.load(open(f"{D}/meta.json")).get("kb")

def col(g):
    """Counts of one gene per cell."""
    return np.asarray(X[:, genes[g]].todense()).ravel()

def expr(g):
    """log1p counts per 10k per cell."""
    return np.log1p(col(g) / np.maximum(umi, 1) * 1e4)

def mask(**kw):
    """Cells matching every column=value (a list/set matches any of its values), e.g. mask(sample="PG4L", state=...)."""
    m = np.ones(len(obs), bool)
    for k, v in kw.items():
        m &= obs[k].isin(list(v)).to_numpy() if isinstance(v, (list, tuple, set)) else (obs[k].to_numpy() == v)
    return m

def pct(g, m=None):
    """% of cells expressing a gene (count > 0), optionally within a mask."""
    v = col(g) > 0
    return float(v[m].mean() * 100) if m is not None else float(v.mean() * 100)

def kb(name):
    """A curated table: progeny, collectri, lr_consensus, omnipath_signaling, membrane_receptors."""
    if not KB:
        raise ValueError("no curated tables in this study")
    return pd.read_csv(f"{KB}/{name}.csv")

def tojson(x, depth=0):
    if isinstance(x, pd.DataFrame):
        return {"columns": [str(c) for c in x.columns], "index": [str(i) for i in x.index[:200]],
                "rows": [[tojson(v, depth + 1) for v in r] for r in x.head(200).itertuples(index=False)]}
    if isinstance(x, pd.Series):
        return {str(k): tojson(v, depth + 1) for k, v in list(x.items())[:200]}
    if isinstance(x, dict):
        return {str(k): tojson(v, depth + 1) for k, v in list(x.items())[:200]}
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [tojson(v, depth + 1) for v in list(x)[:200]]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    return str(x)[:500]

code = open(f"{D}/snippet.py").read()
ns = {"obs": obs, "X": X, "genes": genes, "umi": umi, "np": np, "pd": pd, "sp": sp, "stats": stats,
      "col": col, "expr": expr, "mask": mask, "pct": pct, "kb": kb}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<snippet>", "exec"), ns)
    out = {"ok": True, "stdout": buf.getvalue(), "result": tojson(ns.get("result"))}
except Exception:
    out = {"ok": False, "stdout": buf.getvalue(), "error": traceback.format_exc(limit=3)[-1500:]}
sys.__stdout__.write("\n@@MECHANIST-COMPUTE@@" + json.dumps(out))
'''


class ComputeSandbox:
    """Exports the cells once to a private folder, then runs each snippet in its own isolated process."""

    def __init__(self, cells, kb_dir: str | None = None):
        import threading
        self.cells = cells
        self.kb_dir = kb_dir
        self.dir: Path | None = None
        self._lock = threading.Lock()

    def _export(self) -> Path:
        with self._lock:
            return self._export_locked()

    def _export_locked(self) -> Path:
        if self.dir is not None:
            return self.dir
        import scipy.sparse as sp
        d = Path(tempfile.mkdtemp(prefix="mechanist-compute-"))
        atexit.register(shutil.rmtree, d, True)
        c = self.cells
        X = c.X if not isinstance(c.X, np.ndarray) else sp.csc_matrix(c.X)
        obs = c.obs.copy()
        obs["umi"] = np.asarray(X.sum(axis=1)).ravel()
        obs.to_pickle(d / "obs.pkl")
        sp.save_npz(d / "X.npz", sp.csc_matrix(X), compressed=False)
        (d / "genes.json").write_text(json.dumps(list(c.gene_names)))
        (d / "meta.json").write_text(json.dumps({"kb": self.kb_dir}))
        (d / "runner.py").write_text(RUNNER)
        self.dir = d
        return d

    def _command(self, d: Path, work: Path) -> list[str]:
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("the compute sandbox (bubblewrap) is not installed on this node")
        py = Path(sys.executable)
        binds = {"/usr", "/lib", "/lib64", "/bin", "/etc", str(Path(sys.prefix)), str(Path(sys.base_prefix)),
                 str(py.resolve().parent.parent), str(d)}
        if self.kb_dir:
            binds.add(str(Path(self.kb_dir).resolve()))
        cmd = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp"]
        for b in sorted(binds):
            cmd += ["--ro-bind-try", b, b]
        cmd += ["--bind", str(work), str(work), "--chdir", str(work), "--setenv", "HOME", str(work),
                "--setenv", "OMP_NUM_THREADS", "2", str(py), "-I", str(d / "runner.py"), str(work)]
        return cmd

    def run(self, code: str) -> dict:
        d = self._export()
        work = Path(tempfile.mkdtemp(prefix="snippet-", dir=d))
        try:
            for f in ("obs.pkl", "X.npz", "genes.json", "meta.json"):
                os.link(d / f, work / f)  # hard links: no copy of the data
            (work / "snippet.py").write_text(code)

            def limit():
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
            p = subprocess.run(self._command(d, work), capture_output=True, text=True, timeout=TIMEOUT_S,
                               preexec_fn=limit, check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"the snippet ran longer than {TIMEOUT_S} s and was stopped"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out = p.stdout.rsplit("@@MECHANIST-COMPUTE@@", 1)
        if len(out) < 2:
            err = (p.stderr or p.stdout)[-800:]
            if "bwrap" in err or "namespace" in err.lower():
                raise RuntimeError(f"the compute sandbox could not start: {err[-300:]}")
            return {"ok": False, "error": f"the snippet produced no result: {err}"}
        return json.loads(out[1])
