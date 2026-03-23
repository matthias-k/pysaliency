# Design: Compact HDF5 Export for Model Predictions

**Date:** 2026-03-23
**Status:** Approved

## Background

`pysaliency.export_model_to_hdf5` serializes model predictions (saliency maps or log-density maps) for every stimulus into an HDF5 file. For large stimulus sets these files reach 20–30 GB. Two independently usable optimizations can reduce this significantly with minimal impact on downstream metric scores:

1. **Dtype reduction** — storing predictions as float32 or float16 instead of float64. Pilot experiments show float16 has essentially zero effect on average metric values and only negligible effects on worst-case per-stimulus changes. Float32 is even safer. (Sub-byte float formats such as float8 are out of scope as they are not natively supported by numpy.)
2. **Spatial downsampling** — storing predictions at 1/2, 1/4, or 1/8 of the original resolution. Most saliency models produce smooth outputs. Even 2× downsampling has a noticeably larger effect than dtype reduction, though in practice it is often still acceptable; 4× is larger still; 8× introduces degradation that becomes clearly visible in metric scores.

Both optimizations are independent and can be combined.

## Scope

- `export_model_to_hdf5`: two new parameters (`dtype`, `downscale_factor`)
- `HDF5SaliencyMapModel`: transparent upsampling and upcast on load
- `HDF5Model`: transparent upsampling and renormalization for downsampled or float16 files; strict legacy behavior preserved for float32/float64 non-downsampled files
- New tests in `tests/test_precomputed_models.py`

Out of scope: per-stimulus dtype/downscale selection; formats other than HDF5.

## Design

### Export API

```python
def export_model_to_hdf5(
    model, stimuli, filename,
    compression=9, overwrite=True, flush=False,
    dtype=None,          # e.g. np.float32, np.float16; None = preserve native dtype
    downscale_factor=1,  # integer >= 1; 1 = no downsampling
):
```

Both new parameters default to today's behavior — fully backward compatible.

#### Export pseudocode

The sequencing below is normative; it resolves the ordering of root-attr writing versus stimulus processing:

```python
mode = 'w' if overwrite else 'a'
with h5py.File(filename, mode=mode) as f:

    # Determine which stimuli to process
    if overwrite:
        indices = list(range(len(stimuli)))
    else:
        # append mode is for resuming an interrupted export of the same job,
        # not for combining data from different export runs
        if 'type' in f.attrs:
            # Validate consistency with existing compact file
            _validate_append_consistency(f, downscale_factor)  # raises ValueError on mismatch
        indices = [i for i in range(len(stimuli)) if names[i] not in f]

    if not indices:
        return  # nothing to do

    # Process first stimulus to determine effective_stored_dtype (needed for root attrs)
    first_smap = _compute_smap(model, stimuli[indices[0]])
    effective_stored_dtype = _effective_dtype(first_smap, dtype, downscale_factor)

    # Emit uint8/size guard warning based on first stimulus (once only)
    _check_size_guard(first_smap, effective_stored_dtype)

    # Write root attrs if this is a new file (overwrite=True or file had no attrs)
    if overwrite or 'type' not in f.attrs:
        f.attrs['type'] = 'pysaliency.precomputed_models.predictions'
        f.attrs['version'] = '1.0'
        f.attrs['downscale_factor'] = downscale_factor
        f.attrs['dtype'] = str(effective_stored_dtype)  # informational; based on first stimulus

    # Process all stimuli (reuse already-computed first_smap)
    for i, k in enumerate(indices):
        smap = first_smap if i == 0 else _compute_smap(model, stimuli[k])
        smap = _downsample(smap, downscale_factor)       # step 3; no-op if downscale_factor == 1
        smap = smap.astype(dtype) if dtype is not None else smap  # step 4
        ds = f.create_dataset(names[k], data=smap, compression=compression)
        if downscale_factor > 1:
            ds.attrs['original_shape'] = np.array([H, W], dtype=np.int64)  # pre-padding H, W from step 3
        if flush:
            f.flush()
```

Note on the root-attrs `dtype` field: it is **purely informational**, computed from the first stimulus only. If stimuli have heterogeneous native dtypes (rare in practice), the field may not reflect later stimuli — this is acceptable given it is informational and the loader does not depend on it.

#### `_effective_dtype(smap, dtype, downscale_factor)`

```python
def _effective_dtype(smap, dtype, downscale_factor):
    if dtype is not None:
        return np.dtype(dtype)
    if downscale_factor > 1:
        # numpy .mean() returns float64 for integer inputs, preserves float types
        if np.issubdtype(smap.dtype, np.integer):
            return np.dtype(np.float64)
        else:
            return smap.dtype  # float32 stays float32, float64 stays float64
    return smap.dtype
```

#### `_downsample(smap, downscale_factor)`

```python
def _downsample(smap, k):
    if k == 1:
        return smap
    H, W = smap.shape
    H_pad = int(np.ceil(H / k)) * k
    W_pad = int(np.ceil(W / k)) * k
    smap = np.pad(np.ascontiguousarray(smap),
                  ((0, H_pad - H), (0, W_pad - W)),
                  mode='edge')
    # np.ascontiguousarray ensures C-order before reshape; np.pad preserves C-order
    return smap.reshape(H_pad // k, k, W_pad // k, k).mean(axis=(1, 3))
    # Note: when H is not divisible by k, the last bin is biased toward the
    # border value (edge-padded rows are copies). This is a minor, acceptable
    # artefact for the 2× and 4× factors targeted by this feature.
```

#### `_validate_append_consistency(f, downscale_factor)`

Checks only `int(f.attrs['downscale_factor']) == downscale_factor`. Raises `ValueError` with a descriptive message on mismatch.

The `dtype` root attr is purely informational and is intentionally excluded from this check: determining `effective_stored_dtype` requires computing the first stimulus, which has not happened yet at the point the validator is called. The `downscale_factor` check is sufficient to catch the most dangerous mismatch (spatial layout of stored data).

#### Uint8 / size guard

Implemented in `_check_size_guard(smap, effective_stored_dtype)`:

If `effective_stored_dtype.itemsize > np.dtype(smap.dtype).itemsize`, emit `warnings.warn` explaining:
- The native dtype and its size in bytes
- The effective stored dtype and its size in bytes
- A suggestion to pass `dtype=np.uint8` (if native is integer) or to omit compact options
- Additionally, if native is integer and `downscale_factor > 1` and `dtype` is None: note that casting the float result back to the original integer type would be lossy

### HDF5 File Format

```
/ (root)
  attrs:
    type = 'pysaliency.precomputed_models.predictions'
    version = '1.0'
    downscale_factor = 2
    dtype = 'float32'               # informational; from first stimulus

  images/cat.jpg   (dataset shape ceil(H/2) × ceil(W/2), dtype float32)
    attrs:
      original_shape = np.array([H, W], dtype=np.int64)   # pre-padding; present only when downscale_factor > 1

  images/dog.jpg   (dataset shape ceil(H'/2) × ceil(W'/2), dtype float32)
    attrs:
      original_shape = np.array([H', W'], dtype=np.int64)
```

**Overwritten files** (`overwrite=True`): h5py opens with `mode='w'`, which truncates the file completely before writing begins. The resulting file is always internally consistent.

**Append-mode files** (`overwrite=False`): intended only for resuming an interrupted export. All datasets in the file originate from the same export job with the same settings; mixing data from different jobs is not a supported use case.

**Legacy files** (written by current code) have no root attrs and no `original_shape` on datasets. The loader handles them identically to today.

**Non-compact new files** (`dtype=None`, `downscale_factor=1`) have root attrs but no `original_shape` on datasets — the loader skips upsampling.

Root attrs are purely informational. The loader derives behavior entirely from per-dataset `original_shape` (whether to upsample) and the dataset's native HDF5 dtype (which normalization path to use in `HDF5Model`).

### Load Side

#### `HDF5SaliencyMapModel`

`__init__` reads and stores `f.attrs.get('version')` for future compat; no other init changes.

A new private helper is added:

```python
def _key_for_stimulus(self, stimulus):
    stimulus_id = get_image_hash(stimulus)
    stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)  # raises ValueError if not found
    return self.names[stimulus_index]
```

`_saliency_map` is refactored to call `_key_for_stimulus` internally, replacing the inline index lookup currently at lines 328–329.

`_saliency_map` updated logic:

1. `stimulus_key = self._key_for_stimulus(stimulus)`
2. `dataset = self.hdf5_file[stimulus_key]`; `smap = dataset[:]`
3. If `'original_shape'` in `dataset.attrs`: cast `smap` to float64, then upsample:
   ```python
   target_shape = tuple(dataset.attrs['original_shape'])
   zoom_factors = (target_shape[0] / smap.shape[0], target_shape[1] / smap.shape[1])
   smap = scipy.ndimage.zoom(smap.astype(np.float64), zoom_factors, order=1, mode='nearest')
   ```
4. Otherwise: return `smap` with native dtype unchanged.

Shape check (`check_shape`) runs against the post-upsampling shape, as today. When using downsampled exports with resized stimuli and `check_shape=False`, upsampling targets `original_shape` (the shape at export time), not the (resized) stimulus shape.

**Dtype behavior:**

| File type | `original_shape` present? | `_saliency_map` returns |
|---|---|---|
| Legacy / uint8 / non-compact | no | native dtype (unchanged) |
| Compact, dtype-reduced only | no | native dtype (float16/32) |
| Compact, downsampled | yes | float64 |

The dtype-reduced-only path intentionally returns the native stored dtype. Downstream code that requires float64 (e.g. `HDF5Model`) is responsible for casting.

#### `HDF5Model`

Constructor gains a new parameter and explicitly stores it:

```python
class HDF5Model(Model):
    def __init__(self, stimuli, filename, check_shape=True,
                 max_normalization_error=np.log(1.1), **kwargs):
        super().__init__(**kwargs)
        self.parent_model = HDF5SaliencyMapModel(
            stimuli=stimuli, filename=filename,
            caching=False, check_shape=check_shape,
        )
        self.max_normalization_error = max_normalization_error  # on HDF5Model, not parent_model
```

`max_normalization_error` can be updated after construction. Setting it to `None` disables the tolerance guard on the relaxed path only — the strict ±0.01 check on the legacy path is always applied regardless.

`_log_density` updated logic:

```python
def _log_density(self, stimulus):
    key = self.parent_model._key_for_stimulus(stimulus)
    dataset = self.parent_model.hdf5_file[key]
    use_relaxed_path = (
        'original_shape' in dataset.attrs                                     # spatially downsampled
        or dataset.dtype.itemsize < np.dtype(np.float32).itemsize             # float16 or narrower
    )
    # Note: this attr lookup is cheap (HDF5 metadata); parent_model has caching=False
    # so the subsequent saliency_map call also reads from disk each time.

    smap = self.parent_model.saliency_map(stimulus).astype(np.float64)

    if use_relaxed_path:
        if self.max_normalization_error is not None:
            if abs(logsumexp(smap)) >= self.max_normalization_error:
                raise ValueError(
                    f'Log density normalization error {abs(logsumexp(smap)):.4f} '
                    f'exceeds threshold {self.max_normalization_error:.4f}'
                )
        smap -= logsumexp(smap)
    else:
        # Legacy path: strict check, no renormalization
        if not -0.01 <= logsumexp(smap) <= 0.01:
            raise ValueError('Not a correct log density!')

    return smap
```

### Interpolation Choices

Motivated by pilot experiments comparing several strategies (always using bilinear upsampling in log space):

- **Downsampling**: area averaging in log/value space via edge-pad then reshape + mean. Non-divisible shapes are handled by padding to the next multiple of `k` with edge values; `original_shape` stores the pre-padding shape so the padding region is implicitly removed during upsampling. Known minor limitation: the last bin in non-divisible dimensions is biased toward the border value (edge-padded rows). This is acceptable for the 2× and 4× factors targeted. Empirically outperforms subsampling and probability-space pooling.
- **Upsampling**: bilinear interpolation in log/value space via `scipy.ndimage.zoom(order=1, mode='nearest')`. `mode='nearest'` matches the existing project convention and avoids reflect-padding edge artefacts. The upsampling strategy has negligible empirical effect on metric scores.
- **Renormalization**: not applied at export; applied at load time only on the relaxed path in `HDF5Model`, after verifying the residual is within the configured tolerance.

## Tests

All new tests in `tests/test_precomputed_models.py`. Roundtrip tests are parametrized over `(dtype, downscale_factor)` combinations: `(None, 1)`, `(np.float32, 1)`, `(np.float16, 1)`, `(np.float32, 2)`, `(np.float16, 4)` for both `HDF5SaliencyMapModel` and `HDF5Model`. The `(None, 1)` case is a regression test (no new code paths). For `HDF5Model`:
- `(np.float32, 1)` and `(None, 1)`: exercise the strict legacy path — no renormalization, strict ±0.01 check
- `(np.float16, 1)`: exercises the relaxed path triggered by dtype (float16 itemsize < float32 itemsize)
- `(np.float32, 2)` and `(np.float16, 4)`: exercise the relaxed path triggered by `original_shape`

| Test | Description |
|------|-------------|
| Root attrs | Correct `type`, `version`, `dtype` (using `effective_stored_dtype`), `downscale_factor` written |
| Dataset dtype | Stored dtype matches `effective_stored_dtype` |
| Dataset shape | Stored shape is `ceil(original_shape / downscale_factor)` |
| `original_shape` attr | Present iff `downscale_factor > 1`; `np.int64` array with correct pre-padding shape |
| Non-divisible shape — stored | Stored shape is `ceil(H/k) × ceil(W/k)` |
| Non-divisible shape — loaded | Upsampled output shape matches `original_shape` exactly |
| Append mode — consistent | Appending with matching settings succeeds |
| Append mode — mismatch (downscale only) | `ValueError` on mismatched `downscale_factor` |
| Append mode — mismatch (dtype only) | `ValueError` on mismatched `dtype` |
| Append mode — new file created | Root attrs written correctly when `overwrite=False` and file is new |
| Append mode — legacy file | Append to legacy file succeeds; no root attrs written |
| uint8 + `downscale_factor>1` — warning | `UserWarning` emitted |
| uint8 + `downscale_factor>1` — stored dtype | Actual stored dtype is float64 |
| uint8 + `dtype=np.float32` — warning | `UserWarning` emitted |
| uint8 + `dtype=np.uint8` | No warning |
| float32 + `downscale_factor>1` + `dtype=None` | Stored dtype is float32 (not float64); root attr `dtype` is `'float32'` |
| Legacy file load | Native dtype returned, shape matches stimulus — no behavioral change |
| float32 load, no downsampling | float32 returned (native dtype preserved) |
| Downsampled load | float64 returned, shape matches `original_shape` |
| Resized stimuli + `check_shape=False` | Stimulus presented at size S ≠ `original_shape`; upsamples to `original_shape`, not S |
| `HDF5Model` float16 relaxed path | float16 file triggers relaxed path; output float64, logsumexp ≈ 0 |
| `HDF5Model` float32 strict path | float32 non-downsampled file uses strict path; logsumexp outside ±0.01 raises `ValueError` |
| `HDF5Model` non-compact no renorm | Non-compact float32/float64 output is bit-identical to stored values (cast to float64) |
| `HDF5Model` downsampled roundtrip | float64 output, logsumexp ≈ 0 after renorm (parametrized) |
| `HDF5Model` threshold | logsumexp before renorm within `log(1.1)` for float16+4× |
| `HDF5Model` corrupted file | `ValueError` raised when logsumexp exceeds `max_normalization_error` |
| Custom `max_normalization_error` | Tighter threshold triggers `ValueError`; looser does not |
| `max_normalization_error=None` | Relaxed path skips guard; output is float64, logsumexp ≈ 0 (renorm still applied) |
