# Design: HDF5 Save/Reload for Saliency Models

**Date:** 2026-03-21
**Branch:** enh-save-baseline-model-to-hdf5
**Status:** Approved

## Background

`pysaliency.export_model_to_hdf5` serializes model *predictions* (saliency maps) for every stimulus into an HDF5 file. For large stimulus sets this produces huge files. For models that are fast to compute (e.g. kernel density baseline models), it is preferable to serialize the model *parameters* instead, so the model can be reconstructed and run on demand.

The datasets module already has a mature `to_hdf5` / `read_hdf5` pattern per class, with a unified `pysaliency.datasets.read_hdf5` dispatcher. This design extends that pattern to models.

## Scope

- `BaselineModel`: already implemented in the current branch (no changes needed)
- `CrossvalidatedBaselineModel`: new, covered by this spec
- `pysaliency.read_hdf5`: unified dispatcher (already partially implemented); document `**kwargs` passthrough contract

Out of scope: other model types (`GoldModel`, `KDEGoldModel`, etc.) — follow the same pattern incrementally.

## Key Constraints

- **Stimulus IDs are not stable across systems.** SHA1 hashes of pixel data differ between libjpeg versions. Saving only `stimulus_ids` (Option B) was considered and rejected: it silently breaks when a system update changes JPEG decoding. Full stimulus data must be embedded, or the caller must supply stimuli explicitly at load time.
- **Stimuli can be large.** Embedding raw pixel arrays for large datasets is expensive. A flag controls whether stimuli are embedded, analogous to `BaselineModel`'s `include_shape_cache`.

## Design

### `BaselineModel` (existing — no changes)

Saves to HDF5:
- `attrs['type'] = 'pysaliency.baseline_utils.BaselineModel'`
- `attrs['version'] = '1.0'`
- `attrs['bandwidth']`, `attrs['eps']`, `attrs['keep_aspect']`
- dataset `xs`, dataset `ys` — normalized fixation coordinates
- optional group `shape_cache` — precomputed log-density maps per image shape (controlled by `include_shape_cache=True`, default `True`)

No reference to stimuli or fixations is stored; the model is self-contained.

### `CrossvalidatedBaselineModel`

#### `to_hdf5(target, include_stimuli=True)`

Uses the existing `@hdf5_wrapper(mode='w')` decorator for transparent file/group dispatch.

Saves:
- `attrs['type'] = 'pysaliency.baseline_utils.CrossvalidatedBaselineModel'`
- `attrs['version'] = '1.0'`
- `attrs['bandwidth']`, `attrs['eps']`
- dataset `xs`, dataset `ys` — normalized fixation coordinates
- dataset `fixations_n` — `self.fixations.n` (integer array mapping each fixation to its stimulus index)
- group `stimuli` — embedded via `self.stimuli.to_hdf5(target.create_group('stimuli'))` — **only when `include_stimuli=True`**

When `include_stimuli=False`, the `stimuli` group is omitted entirely; the file is not self-contained and requires `stimuli=` at load time.

#### `read_hdf5(source, *, stimuli=None, caching=True, memory_cache_size=None, cache_location=None)`

Classmethod, uses `@hdf5_wrapper(mode='r')`.

Load order:
1. Validate `type` and `version` attrs
2. Resolve stimuli:
   - If `stimuli` kwarg is provided → use it (ignore embedded group even if present)
   - Else if `'stimuli'` group exists in file → load via `pysaliency.datasets.read_hdf5`
   - Else → raise `ValueError("No stimuli found in HDF5 file. Pass `stimuli=` explicitly.")`
3. Reconstruct model via `cls.__new__(cls)` + `Model.__init__(...)` (same pattern as `BaselineModel`)
4. Restore `bandwidth`, `eps`, `xs`, `ys`, `fixations.n` (stored as a plain struct with `.n` attribute, or just as an array assigned to `self.fixations` with a `.n` property — see Implementation Notes)

#### Registration

`pysaliency/hdf5.py` `_MODEL_READERS`:

```python
_MODEL_READERS = {
    'pysaliency.baseline_utils.BaselineModel': _read_baseline_model,
    'pysaliency.baseline_utils.CrossvalidatedBaselineModel': _read_crossvalidated_baseline_model,
}
```

where `_read_crossvalidated_baseline_model` is a thin wrapper that delegates to `CrossvalidatedBaselineModel.read_hdf5`.

### `pysaliency.read_hdf5` dispatcher

The dispatcher already accepts and passes through `**kwargs` to the class-specific reader. This is the correct contract — `read_hdf5` remains a thin dispatcher with no knowledge of model-specific parameters.

Usage:
```python
# Self-contained file — no kwargs needed
model = pysaliency.read_hdf5('model.hdf5')

# File saved without stimuli — pass them through **kwargs
model = pysaliency.read_hdf5('model.hdf5', stimuli=my_stimuli)

# Canonical class-level API (always available)
model = CrossvalidatedBaselineModel.read_hdf5('model.hdf5', stimuli=my_stimuli)
```

### `pysaliency.__init__` exports

`read_hdf5` is added to the top-level `pysaliency` namespace (already in progress on this branch).

## Implementation Notes

- `CrossvalidatedBaselineModel._log_density` currently references `self.stimuli` (for `stimulus_ids`) and `self.fixations.n`. After deserialization, `self.stimuli` is the resolved stimuli object and `self.fixations` needs a `.n` attribute. The simplest approach is a small namedtuple or dataclass `_FixationsN` with just a `.n` field, avoiding a full `Fixations` object dependency. Alternatively, store `fixations_n` directly as `self._fixations_n` and update `_log_density` accordingly — this is a minor internal refactor, not a public API change.
- `hdf5_wrapper` currently handles `self` as the first argument (instance methods). For classmethods, `BaselineModel` uses `@classmethod` + `@hdf5_wrapper` stacked — the same pattern applies to `CrossvalidatedBaselineModel`.
- When stimuli are embedded as raw numpy arrays (via `Stimuli.to_hdf5`), hashes computed at reload time are stable. When the caller passes `FileStimuli` and JPEG decoding differs across systems, they must provide `stimuli=` explicitly to ensure hash consistency.

## Tests

All tests go in `tests/test_baseline_utils.py` (existing file) and `tests/test_hdf5_io.py`:

| Test | File |
|------|------|
| Roundtrip `CrossvalidatedBaselineModel` with `include_stimuli=True` | `test_baseline_utils.py` |
| Roundtrip `CrossvalidatedBaselineModel` with `include_stimuli=False` + explicit `stimuli=` | `test_baseline_utils.py` |
| `ValueError` when `include_stimuli=False` and no `stimuli=` on reload | `test_baseline_utils.py` |
| `type` attr is `'pysaliency.baseline_utils.CrossvalidatedBaselineModel'` | `test_baseline_utils.py` |
| `pysaliency.read_hdf5` dispatcher round-trips `CrossvalidatedBaselineModel` | `test_hdf5_io.py` |
| `pysaliency.read_hdf5` with `stimuli=` kwarg passthrough | `test_hdf5_io.py` |
