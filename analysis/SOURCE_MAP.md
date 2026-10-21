# Paper Analysis Source Map

Sources are from the accepted `reference_convention_writing_package_v1`.
Numerical functions were retained; paths and internal orchestration were ported.
The analysis scripts are not re-executed in this distribution (see the validation status in the README).

| Public path | Source under the accepted analysis package |
|---|---|
| `01_label_comparison/label_comparison.py` | `01_reference_geometry/methods/00_label_geometry_audit.py` |
| `01_label_comparison/utils.py`, `_shared/convention_utils.py` | Used exports and shared definitions from `01_reference_geometry/methods/convention_utils.py` |
| `01_label_comparison/config.yaml` | `01_reference_geometry/methods/protocol_convention_v1.yaml` |
| `02_deployed_models/evaluate_fixed_models.py` | `02_fixed_prediction_reference_fragility/methods/reference_fragility_experiment.py` |
| `02_deployed_models/{config,dataset,model_zoo}.py` | Same-named historical methods |
| `03_synchronized_retraining/train_synchronized_pair.py` | `03_paired_convention_training/methods/train_paired_v2.py` |
| `03_synchronized_retraining/{common,rng_pairing}.py` | Same-named paired-study methods |
| `03_synchronized_retraining/voxel_reproduction.py` | `03_paired_convention_training/methods/03_spatial_transmission.py` |
| `03_synchronized_retraining/label_free_scores.py` | `03_paired_convention_training/methods/05_monitor_reassessment.py` |
| `03_synchronized_retraining/control_report.py` | `03_paired_convention_training/methods/06_final_control_report.py` |
| `03_synchronized_retraining/control_runs.py` | Control-only entry point for the retained paired-training implementation |
| `03_synchronized_retraining/config.yaml`, `SYNCHRONIZED_RETRAINING.md` | Paired scientific protocol and RNG specification |
| `04_nnunet_training/training_core.py` | `04_c2_development_qualification/methods/training/training_core.py`, without compute-accounting fields/checks |
| `04_nnunet_training/train_fold.py`, `training_io.py` | Portable entry point and path helpers around the retained official trainer instrumentation |
| `04_nnunet_training/nnUNetPlans.json`, `development_splits.json` | Frozen study plans and development case assignments; no image data |
| `04_nnunet_training/out_of_fold_qualification.py` | Portable OOF export and qualification entry point replacing internal dispatch/namespace wrappers |
| `04_nnunet_training/oof_{predictor,metrics,selection}.py` | Actual predictor, metrics and selection implementations delegated to by the original OOF wrappers |
| `04_nnunet_training/config.yaml` | OOF scientific protocol, without source-run paths, authorizations or compute accounting |
| `05_nnunet_evaluation/evaluate_ensemble.py` | `05_c2_reliability_evaluation/methods/03_analyze.py`, with one portable stage CLI |
| `05_nnunet_evaluation/reliability_{data,inference,metrics}.py` | Same-named evaluation methods |
| `05_nnunet_evaluation/reliability_common.py` | Portable configuration/file helpers replacing internal deployment bookkeeping |
| `05_nnunet_evaluation/config.yaml` | Frozen v1.3 scientific evaluation contract without operational amendment history |

The original OOF files are dispatch wrappers, not the numerical implementation.
Their delegated predictor/metric/selection definitions are retained as helper
modules rather than shipping the operational wrappers. The source for those
definitions is the original `c2_development_qualification_v1` implementation.

The paired control report still consumes existing supporting calibration tables;
the exploratory calibration script and its results are not redistributed.
The report's mixed-row correlation mean must not be described as an average
over the three seeds. Historical regional names do not resolve anatomy.

Excluded: operational launch/recovery scripts, internal audit scripts, approval
documents, accounting files, historical provenance bundles, crossed decomposition,
raw cohorts, checkpoints, predictions and result tables, including
the per-patient table. Figure scripts are omitted at the author's request.
