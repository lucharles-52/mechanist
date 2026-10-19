"""Build Mechanist's offline knowledge base (run once with the pipeline env, which has liana + decoupler):

    ../Projects/.cl_analysis/envs/cl-analysis-llm-2026-09-rapids/.venv/bin/python scripts/build_kb.py

Writes kb/lr_consensus.csv (LIANA consensus ligand-receptor resource), kb/collectri.csv (CollecTRI TF -> target,
signed), kb/progeny.csv (PROGENy pathway -> responsive gene weights, top 500 per pathway) and kb/manifest.json
(versions, row counts, sha256). Only public resources are fetched; nothing from the study leaves the cluster.
"""

import datetime as dt
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import decoupler as dc
import liana as li
import pandas as pd

KB = Path(__file__).resolve().parents[1] / "kb"
KB.mkdir(exist_ok=True)

OMNIPATH = ("https://omnipathdb.org/interactions?datasets=omnipath&genesymbols=1&organisms=9606&directed=1"
            "&fields=sources,references&license=academic&format=tsv")
with urllib.request.urlopen(OMNIPATH, timeout=300) as r:
    sig = pd.read_csv(io.StringIO(r.read().decode()), sep="\t")
sig = sig[sig.consensus_direction.astype(bool)]
sig = pd.DataFrame({"source": sig.source_genesymbol, "target": sig.target_genesymbol,
                    "stimulation": sig.consensus_stimulation.astype(bool),
                    "inhibition": sig.consensus_inhibition.astype(bool),
                    "n_refs": sig.references.fillna("").str.count(";") + sig.references.notna().astype(int)})

INTERCELL = "https://omnipathdb.org/intercell?categories=receptor&plasma_membrane_transmembrane=yes&format=tsv"
with urllib.request.urlopen(INTERCELL, timeout=300) as r:
    ic = pd.read_csv(io.StringIO(r.read().decode()), sep="\t")
receptors = (ic.groupby("genesymbol").size().rename("n_resources").reset_index()
             .rename(columns={"genesymbol": "gene"}))

tables = {
    "membrane_receptors.csv": receptors,
    "omnipath_signaling.csv": sig,
    "lr_consensus.csv": li.resource.select_resource("consensus")[["ligand", "receptor"]],
    "collectri.csv": dc.op.collectri(organism="human")[["source", "target", "weight"]],
    "progeny.csv": dc.op.progeny(organism="human", top=500)[["source", "target", "weight"]],
}
manifest = {"built": dt.datetime.now().isoformat(timespec="seconds"), "liana": li.__version__,
            "decoupler": dc.__version__, "files": {}}
for name, df in tables.items():
    path = KB / name
    df.to_csv(path, index=False)
    manifest["files"][name] = {"rows": len(df), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(KB / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(manifest, indent=1))
