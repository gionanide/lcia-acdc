# 5. Frozen ensemble evaluation
Applies all five final checkpoints to ACDC 101-150 and the 320 labelled M&Ms patients.
Run from the repository root with user-owned cohort folders:
`python analysis/05_nnunet_evaluation/evaluate_ensemble.py --stage all --acdc-dir ACDC --refined-dir RefinedLabelsACDC --mnms-dir MNM --mnms-metadata MNM/metadata.csv --nnunet-root nnunet --historical-checkpoint checkpoints/last.pt --qualification-dir outputs/oof --out-dir outputs/evaluation`
Stages `prepare`, `infer`, `analyze` can instead be called separately with the same paths.
Outputs: native masks/probabilities, per-patient tables, segmentation/measurement batteries and figures.
Uses official probability averaging, not hard-mask majority voting.
The M&Ms validation gate and bounded header-only geometry policy are preserved.
No training, adaptation, calibration, postprocessing selection or M&Ms-2 access is included.
The convention envelope is not a confidence interval or a claim of anatomical truth.
