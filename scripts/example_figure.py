"""Example: label agreement against RVEF change, and the convention envelope of the test patients.
Usage: python scripts/example_figure.py  (writes figures/example.png)"""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
table = root / "data" / "acdc_lcia_per_patient.csv"
if not table.exists():
    sys.exit("data/acdc_lcia_per_patient.csv not found; build it first with scripts/build_table.py (see README)")
t = pd.read_csv(table)
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
dmin = t[["dice_ed", "dice_es"]].min(axis=1); big = t.delta_rvef.abs() > 5
a.scatter(dmin[~big], t.delta_rvef.abs()[~big], s=10, c="#bdbdbd"); a.scatter(dmin[big], t.delta_rvef.abs()[big], s=10, c="#222222")
a.axhline(5, ls="--", c="k", lw=0.7); a.set_xlim(0.8, 1.0)
a.set_xlabel("Dice between the two label sets (min of ED, ES)"); a.set_ylabel("|RVEF change| (points)")
te = t[t.split == "test"].sort_values(by="envelope_low").reset_index(drop=True)
for i, r in te.iterrows():
    b.plot([i, i], [r.envelope_low, r.envelope_high], c="#b03a2e" if r.flag_40 else "#b5b5b5", lw=1.6)
b.scatter(te.index, te.nnunet_rvef, s=6, c="k", zorder=3); b.axhline(40, ls="--", c="k", lw=0.7)
b.set_xlabel("Test patients"); b.set_ylabel("RVEF (%)"); b.set_xticks([])
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
(root / "figures").mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(root / "figures" / "example.png", dpi=200)
print("flagged at 40 %:", int(te.flag_40.sum()), "of", len(te))
