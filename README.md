# LCIA-ACDC: label-change impact assessment for the ACDC right ventricular labels

This repository accompanies the paper *When Reference Labels Change: A Framework for Assessing Deep Learning Measurement of Right Ventricular Ejection Fraction* (Gionanidis, Miltiadous, Giannakeas, Tzallas, 2026).

The ACDC dataset has two expert label sets for the right ventricle (RV): the original labels of the challenge and the refined labels released by Zhao et al. (MICCAI 2024). The two label sets agree closely in terms of Dice, but they can give a different right ventricular ejection fraction (RVEF) for the same patient. This repository provides the code which compares the two label sets for every patient, the code of the convention envelope and the decision-stability flag, and the analysis source of the paper.

## Contents

| Path | Content |
|---|---|
| `lcia/` | Python code: convention envelope, decision-stability flag, envelope error, volumes, RVEF and Dice |
| `lcia/compare.py` | Level 1 of LCIA computed directly from the two folders of label files |
| `scripts/example_figure.py` | Example plot from the per-patient table (generated locally) |
| `scripts/build_table.py` | Provenance script which builds the per-patient table from the analysis outputs of the paper |
| `tests/` | Tests of the code (and of the per-patient table, when it is generated locally) |
| `analysis/` | Five paper-analysis steps, with inputs supplied as arguments |

## Analysis source

Each analysis folder has a short README with its primary command. The five
steps cover label comparison, fixed historical models, synchronized retraining,
nnU-Net training/OOF qualification, and native-grid ensemble evaluation.
`analysis/SOURCE_MAP.md` records the source mapping and portability changes.
No images, labels, weights, probability caches, per-patient tables or analysis
results are included. Paper figure scripts are not included in this distribution.

The historical four-model comparison uses processed-grid predictions against
native-grid reference measurements, with processed-grid Dice. Synchronized
training uses native-projected endpoint measurements and processed-grid Dice.
The qualified nnU-Net analysis uses native export for measurements and overlap.
These are not one uniform native-grid model leaderboard.

Historical models are fixed non-endpoint-selected checkpoints identified by
SHA-256; their completed epoch counts are unproved. This limitation does not
apply to the documented 100-epoch synchronized training or the five 1,000-epoch
nnU-Net folds. The separate 200-epoch fold-0 continuation is nonselectable.

### Validation status

The tests of the code (`python -m pytest tests`) pass, and `scripts/build_table.py` reproduces the per-patient table of the paper exactly from its analysis outputs. The table itself is not redistributed (see Data sources). The Level 1 comparison from the raw ACDC label files (`python -m lcia.compare`) and the five-epoch identical-label control of the synchronized retraining require the ACDC data, and they are not run for this distribution.

## The per-patient table

The table is not included in this repository. It is written to `data/acdc_lcia_per_patient.csv` by `scripts/build_table.py`, and its Level 1 columns can be computed from the ACDC data and the refined labels with `python -m lcia.compare`. All volumes are in mL and all RVEF values in percent. The RVEF is calculated as 100 (EDV − ESV) / EDV from the voxel counts on the native image grid. The Dice is computed for the RV between the original and the refined label of the same frame.

| Column | Description |
|---|---|
| `patient` | ACDC patient identifier |
| `split` | `development` (patients 001–100) or `test` (patients 101–150), as used in the paper |
| `group` | ACDC diagnostic group (DCM, HCM, MINF, NOR, RV) |
| `ed_frame`, `es_frame` | End-diastolic and end-systolic frame |
| `original_edv_ml`, `refined_edv_ml` | End-diastolic RV volume under the original and the refined labels |
| `original_esv_ml`, `refined_esv_ml` | End-systolic RV volume under the original and the refined labels |
| `original_rvef`, `refined_rvef` | RVEF under the original and the refined labels |
| `delta_rvef` | Refined minus original RVEF (percentage points) |
| `dice_ed`, `dice_es` | Dice between the two RV labels at end diastole and end systole |
| `rv_labels_identical` | 1 if the RV labels of both frames are identical |
| `envelope_low`, `envelope_high`, `envelope_width` | Convention envelope, i.e., the RVEF interval spanned by the two label sets |
| `flag_40`, `flag_45` | Decision-stability flag at the 40 % and 45 % thresholds, i.e., 1 if the two label sets place the RVEF on different sides of the threshold |
| `nnunet_rvef` | RVEF predicted by the nnU-Net ensemble of the paper, trained on the original labels (test patients only) |
| `nnunet_error_original`, `nnunet_error_refined` | Predicted minus reference RVEF against each label set (test patients only) |
| `nnunet_envelope_error` | Distance of the prediction from the convention envelope (test patients only) |

## Usage

```python
import pandas as pd
from lcia import decision_flag, envelope_error

t = pd.read_csv("data/acdc_lcia_per_patient.csv")   # generated locally
flags = decision_flag(t.original_rvef, t.refined_rvef, tau=45)   # same as t.flag_45
```

To evaluate your own model against both label sets, compute its RVEF for each patient and use `envelope_error(your_rvef, t.original_rvef, t.refined_rvef)`.

To recompute Level 1 from the label files:

```bash
pip install -r requirements.txt
python -m lcia.compare <ACDC_folder> <RefinedLabelsACDC_folder> level1.csv
python -m pytest tests
```

## Data sources

No images, labels or per-patient values derived from them are redistributed here. The ACDC dataset is available from its organizers (https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and should be cited as Bernard et al., IEEE TMI 2018. The refined labels were obtained from the repository of Zhao et al. as the archive `RefinedLabelsACDC.zip` at commit `604de3ed1a718e1488c767c5a0c52836da687fb8`.

## Citation

If you use the code, please cite the paper (see `CITATION.cff`), the ACDC dataset and the refined labels of Zhao et al.

## License

The code is released under the MIT License.
