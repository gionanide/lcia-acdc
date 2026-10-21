# 2. Fixed historical models
Holds predictions fixed while changing references for ACDC patients 101-150.
Run from the repository root:
`python analysis/02_deployed_models/evaluate_fixed_models.py --acdc-dir ACDC --refined-dir RefinedLabelsACDC --runs-root checkpoints --out-dir outputs/fixed --architectures unet segresnet attnunet swinunetr --n-boot 5000`
Inputs: externally supplied last-named checkpoints; `best_by_ef` is rejected.
Outputs: reference-fragility case/summary/ranking CSVs and JSON metadata.
With U-Net, also writes the synchronized-study reproduction prerequisite.
Prediction volumes use the processed grid; references use the native grid.
Dice uses processed masks; architecture ordering is descriptive.
Historical checkpoint epoch counts are not established by their filenames.
