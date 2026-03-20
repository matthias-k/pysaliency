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

#### Internal refactor: store `fixations_n` directly

As part of this work, `CrossvalidatedBaselineModel.__init__` is refactored to store only the fixation index array rather than the full `fixations` object:

```python
# Before
self.fixations = fixations
self.shape_cache = {}   # unused — remove

# After
self.fixations_n = fixations.n.copy()
```

`_log_density` is updated accordingly: `self.fixations.n` → `self.fixations_n`. This makes the model's stored state minimal and eliminates any need for workarounds during deserialization.

#### `to_hdf5(target, include_stimuli=True)`

Uses the existing `@hdf5_wrapper(mode='w')` decorator for transparent file/group dispatch.

Saves:
- `attrs['type'] = 'pysaliency.baseline_utils.CrossvalidatedBaselineModel'`
- `attrs['version'] = '1.0'`
- `attrs['bandwidth']`, `attrs['eps']` — note: no `keep_aspect`, no `shape_cache`
- dataset `xs`, dataset `ys` — normalized fixation coordinates
- dataset `fixations_n` — `self.fixations_n` (integer array mapping each fixation to its stimulus index)
- group `stimuli` — embedded via `self.stimuli.to_hdf5(target.create_group('stimuli'))` — **only when `include_stimuli=True`**

When `include_stimuli=False`, the `stimuli` group is omitted entirely; the file is not self-contained and requires `stimuli=` at load time.

#### `read_hdf5(source, *, stimuli=None, caching=True, memory_cache_size=None, cache_location=None)`

Classmethod. Decorator stacking must match `BaselineModel` exactly: `@classmethod` outermost, `@hdf5_wrapper(mode='r')` inner.

Load order:
1. Validate `type` and `version` attrs. Raise `ValueError` if `type` is not `'pysaliency.baseline_utils.CrossvalidatedBaselineModel'` or `version` is not `'1.0'`.
2. Resolve stimuli:
   - If `stimuli` kwarg is provided → use it (ignore embedded group even if present)
   - Else if `'stimuli'` group exists in file → load via `pysaliency.hdf5.read_hdf5(source['stimuli'])`
   - Else → raise `ValueError("No stimuli found in HDF5 file. Pass stimuli= explicitly.")`
3. Reconstruct model via `cls.__new__(cls)` + `Model.__init__(...)` (same pattern as `BaselineModel`)
4. Restore fields:
   - `model.bandwidth = source.attrs['bandwidth']`
   - `model.eps = source.attrs['eps']`
   - `model.xs = source['xs'][...]`
   - `model.ys = source['ys'][...]`
   - `model.fixations_n = source['fixations_n'][...]`
   - `model.stimuli = stimuli` (resolved above)

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

- **`CrossvalidatedBaselineModel` refactor** is an internal change. The public constructor signature `__init__(self, stimuli, fixations, bandwidth, eps)` is unchanged; only the stored attributes change (`self.fixations_n` replaces `self.fixations`, `self.shape_cache` is removed). This is not a breaking change.
- **`hdf5_wrapper` + `@classmethod` ordering:** `@classmethod` must be the outermost decorator and `@hdf5_wrapper(mode='r')` the inner one — exactly as in `BaselineModel.read_hdf5`. Reversing the order will fail.
- **WeakValueDictionary cache in `pysaliency.read_hdf5`:** The top-level `_read_hdf5_from_file` is cached via `boltons.cacheutils.cached`. Since `boltons` does not include `**kwargs` in the cache key, passing `stimuli=` via `pysaliency.read_hdf5('model.hdf5', stimuli=X)` on a second call would return the cached model from the first call. The existing fallback path (lines 43–45 of `hdf5.py`) bypasses the cache when `kwargs` are not hashable — but a numpy array for `stimuli` is not hashable, so the fallback triggers correctly. This is worth an explicit test to guard against regressions.
- **Stimuli hash stability:** Hash consistency requires that the embedded stimuli and the stimuli passed to `_log_density` are decoded via the same code path. If both are `FileStimuli` pointing to the same JPEG files, hashes will match even across libjpeg updates (both sides re-decode with the same updated library). A mismatch only arises when the two sides use different decoding paths — e.g., stimuli embedded as raw numpy arrays in HDF5 but evaluated with freshly JPEG-decoded `FileStimuli`, or vice versa. In such cases `stimuli=` must be passed explicitly on reload.
- **Unknown stimulus at inference time:** If a stimulus is passed to `_log_density` that was not in the training set, `self.stimuli.stimulus_ids.index(stimulus_id)` raises `ValueError`. This is existing behavior and is not changed by this design.

## Tests

All tests go in `tests/test_baseline_utils.py` (existing file) and `tests/test_hdf5_io.py`:

| Test | File |
|------|------|
| Roundtrip `CrossvalidatedBaselineModel` with `include_stimuli=True` — numerical equality of log-density | `test_baseline_utils.py` |
| Roundtrip `CrossvalidatedBaselineModel` with `include_stimuli=False` + explicit `stimuli=` — numerical equality | `test_baseline_utils.py` |
| `ValueError` when `include_stimuli=False` and no `stimuli=` on reload | `test_baseline_utils.py` |
| `type` attr is `'pysaliency.baseline_utils.CrossvalidatedBaselineModel'` | `test_baseline_utils.py` |
| `pysaliency.read_hdf5` dispatcher round-trips `CrossvalidatedBaselineModel` | `test_hdf5_io.py` |
| `pysaliency.read_hdf5` with `stimuli=` kwarg passthrough bypasses WeakValueDictionary cache | `test_hdf5_io.py` |
