# RNG synchronisation and sham-control specification

## Why this is required

The previous paired loop executed the original-label model first and the refined-label model second. Both models contained dropout. Because the torch RNG state was not restored, the two arms received different dropout masks. The observed pair difference therefore cannot be attributed solely to the labels.

## Required deterministic settings

At process start, record and set:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

Use `torch.use_deterministic_algorithms(True)` when supported. If a required operation is nondeterministic, fail or record the exact warning and use a validated deterministic alternative.

Record TF32 and AMP settings. Both arms must use identical settings.

## RNG state helper

Implement helpers equivalent to:

```python
def capture_torch_rng():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_torch_rng(state):
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
```

## Paired batch step

For each batch:

1. zero both optimizers;
2. capture RNG state immediately before arm A forward;
3. run arm A forward/loss/backward/update;
4. restore the captured state;
5. run arm B forward/loss/backward/update;
6. confirm the global RNG state after B equals the expected post-forward state from A when the operations are otherwise identical.

Do not regenerate the image or augmentation separately for each arm. Construct the paired batch once.

## Same-label one-batch sham

Use the same labels in A and B.

Required diagnostics:

- maximum logit difference;
- loss difference;
- per-parameter maximum gradient difference;
- maximum parameter difference after update;
- optimizer-state difference;
- GradScaler state difference if AMP is enabled;
- CPU and CUDA RNG-state hashes before/after each arm.

The test must fail loudly if tolerances are exceeded.

## Five-epoch sham

Train two same-label arms for five epochs from the same initialization with synchronized RNG.

Required checks:

- identical sample-order hashes;
- identical augmentation hashes;
- identical epoch losses;
- checkpoint tensor comparison;
- prediction equality on all 50 ACDC evaluation patients;
- identical EDV, ESV, RVEF, and Dice.

A same-label sham that diverges invalidates the paired-label experiment.

## Optional independent-stochastic null

A separate null may deliberately use independent dropout streams while holding labels and initialization fixed. It must be clearly named `independent_stochastic_null` and never confused with the synchronized sham.
