"""Build data/acdc_lcia_per_patient.csv from the study's analysis tables (provenance script).
Usage: python scripts/build_table.py <analysis_package_dir> data/acdc_lcia_per_patient.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from lcia import convention_envelope, decision_flag, envelope_error
P, OUT = sys.argv[1], sys.argv[2]
rm = pd.read_csv(f"{P}/01_reference_geometry/tables/reference_measurements.csv")
cg = pd.read_csv(f"{P}/01_reference_geometry/tables/convention_label_geometry.csv")
an = pd.read_csv(f"{P}/05_c2_reliability_evaluation/results/reference_policy_analysis.csv")
an = an[an.model == "c2_ensemble"]
dice = cg.pivot(index="patient", columns="phase", values="reference_rv_dice")
group = cg.groupby("patient").group.first()
t = pd.DataFrame({"patient": rm.patient})
t["split"] = np.where(t.patient.str[-3:].astype(int) <= 100, "development", "test")
t["group"] = t.patient.map(group)
t["ed_frame"], t["es_frame"] = rm.ed_frame, rm.es_frame
for c in ["edv_ml", "esv_ml"]:
    t[f"original_{c}"] = rm[f"original_{c}"]; t[f"refined_{c}"] = rm[f"refined_{c}"]
t["original_rvef"], t["refined_rvef"] = rm.original_rvef_pct, rm.refined_rvef_pct
t["delta_rvef"] = t.refined_rvef - t.original_rvef
t["dice_ed"], t["dice_es"] = t.patient.map(dice["ED"]), t.patient.map(dice["ES"])
t["rv_labels_identical"] = ((t.dice_ed == 1.0) & (t.dice_es == 1.0)).astype(int)
lo, hi = convention_envelope(t.original_rvef, t.refined_rvef)
t["envelope_low"], t["envelope_high"] = lo, hi; t["envelope_width"] = hi - lo
for tau in (40, 45):
    t[f"flag_{tau}"] = decision_flag(t.original_rvef, t.refined_rvef, tau)
pred = an.set_index(an.patient_number.astype(int)).predicted_rvef_points
num = t.patient.str[-3:].astype(int)
t["nnunet_rvef"] = num.map(pred)
t["nnunet_error_original"] = t.nnunet_rvef - t.original_rvef
t["nnunet_error_refined"] = t.nnunet_rvef - t.refined_rvef
m = t.nnunet_rvef.notna()
t.loc[m, "nnunet_envelope_error"] = envelope_error(t.nnunet_rvef[m], t.original_rvef[m], t.refined_rvef[m])
dec = {c: 3 for c in ["dice_ed", "dice_es"]}
t = t.round({c: 2 for c in t.columns if t[c].dtype.kind == "f"} | dec)
t.to_csv(OUT, index=False)
# checks against the paper
te = t[t.split == "test"]
print("changed >5:", int((t.delta_rvef.abs() > 5).sum()), "| flags test 40/45:", int(te.flag_40.sum()), int(te.flag_45.sum()),
      "| nnU-Net MAE orig/ref/env:", round(te.nnunet_error_original.abs().mean(), 2), round(te.nnunet_error_refined.abs().mean(), 2), round(te.nnunet_envelope_error.mean(), 2),
      "| identical RV:", int(t.rv_labels_identical.sum()))
