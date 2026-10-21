# 3. Synchronized convention training
Both arms use the same 85/15 patient split, initialization and stochastic choices.
Five-epoch control (not executed for this source port):
`python analysis/03_synchronized_retraining/control_runs.py --acdc-dir ACDC --refined-dir RefinedLabelsACDC --out-dir outputs/paired --runs-root checkpoints/paired --reproduction-gate outputs/fixed/reproduction_gate.json`
Main command: the same arguments to `train_synchronized_pair.py --mode train-convention --seeds 0 1 2`.
Outputs: terminal checkpoints, training metadata and same-label control comparisons.
`voxel_reproduction.py` produces native endpoints and processed-grid spatial comparisons.
`label_free_scores.py` and `control_report.py` consume explicitly supplied supporting tables.
Calibration residual and reference-weighted spatial diagnostics are labelled-audit comparators, not deployable scores.
Use seed correlations individually; the mixed-row mean is not an across-seed mean.
