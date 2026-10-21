# 1. Label comparison
Compares both RV label policies for all 150 ACDC patients, without model inference.
Run from the repository root:
`python analysis/01_label_comparison/label_comparison.py --acdc-dir ACDC --refined-dir RefinedLabelsACDC --out-dir outputs/labels`
Outputs: `convention_label_geometry.csv`, its JSON summary and diagnostic plots.
LV/myocardium changes or unexpected class transitions fail the comparison.
Native reference volumes and processed model metrics must not be conflated.
Stored-index anatomical names are historical fields, not resolved anatomy.
The regional anatomical plot is excluded from the main manuscript.
