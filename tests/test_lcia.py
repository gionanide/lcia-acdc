import numpy as np, pandas as pd, pathlib, sys, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lcia import convention_envelope, decision_flag, envelope_error, dice, rvef

def test_envelope_and_flag():
    lo, hi = convention_envelope([45.15, 50.0], [39.70, 52.0])
    assert np.allclose(lo, [39.70, 50.0]) and np.allclose(hi, [45.15, 52.0])
    assert list(decision_flag([45.15, 50.0], [39.70, 52.0], 40)) == [1, 0]
    assert list(decision_flag([45.15, 50.0], [39.70, 52.0], 45)) == [1, 0]

def test_envelope_error():
    assert np.allclose(envelope_error([42.0, 46.72, 30.0], [45.15] * 3, [39.70] * 3), [0.0, 1.57, 9.70])

def test_dice_and_rvef():
    a = np.zeros((4, 4)); a[:2] = 1; b = np.zeros((4, 4)); b[:3] = 1
    assert abs(dice(a, b) - 0.8) < 1e-9 and dice(np.zeros(3), np.zeros(3)) == 1.0
    assert abs(rvef(100, 40) - 60) < 1e-9

TABLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "acdc_lcia_per_patient.csv"

@pytest.mark.skipif(not TABLE.exists(), reason="per-patient table not generated (see README)")
def test_table_is_consistent():
    t = pd.read_csv(TABLE)
    assert len(t) == 150 and (t.split == "test").sum() == 50
    assert (decision_flag(t.original_rvef, t.refined_rvef, 40) == t.flag_40).all()
    assert (decision_flag(t.original_rvef, t.refined_rvef, 45) == t.flag_45).all()
    te = t[t.split == "test"]
    assert te.flag_40.sum() == 10 and te.flag_45.sum() == 11
    assert (t.delta_rvef.abs() > 5).sum() == 63
    assert abs(te.nnunet_envelope_error.mean() - 3.82) < 0.01
