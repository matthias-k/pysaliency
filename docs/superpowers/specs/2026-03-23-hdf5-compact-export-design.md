# Design: Compact HDF5 Export for Model Predictions

**Date:** 2026-03-23
**Status:** Approved

## Background

`pysaliency.export_model_to_hdf5` serializes model predictions (saliency maps or log-density maps) for every stimulus into an HDF5 file. For large stimulus sets these files reach 20–30 GB. Two independently usable optimizations can reduce this significantly with minimal impact on downstream metric scores:

1. **Dtype reduction** — storing predictions as float32 or float16 instead of float64. Pilot experiments show barely any change in model performance metrics.
2. **Spatial downsampling** — storing predictions at 1/2, 1/4, or 1/8 of the original resolution. Most saliency models produce smooth outputs. 2× and 4× downsampling have negligible metric impact; 8× shows noticeable but sometimes acceptable degradation.

Both optimizations are independent and can be combined.

## Scope

- `export_model_to_hdf5`: two new parameters (`dtype`, `downscale_factor`)
- `HDF5SaliencyMapModel`: transparent upsampling and upcast on load
- `HDF5Model`: transparent upsampling, upcast, renormalization with configurable error threshold
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

#### Export logic per stimulus

1. Compute `smap` as today (native dtype, full resolution).
2. **Uint8 guard** (checked once on first stimulus): if `itemsize(effective_stored_dtype) > itemsize(native_dtype)`, emit `warnings.warn` with a message explaining the size increase and suggesting `dtype=np.uint8` or omitting compact options. If `downscale_factor > 1` and `dtype=None`, note additionally that zoom requires float conversion and casting back to uint8 is lossy.
3. If `downscale_factor > 1`: downsample using `scipy.ndimage.zoom(smap, 1/downscale_factor, order=1, mode='nearest', grid_mode=True)` (area averaging in value/log space). **No renormalization at export** — the small residual is preserved as signal; renormalization happens at load time for log-density models.
4. If `dtype` is not None: cast to target dtype.
5. Store dataset. If `downscale_factor > 1`, store `original_shape` as a dataset-level attribute.

#### File-level attributes (written once at file open)

```python
f.attrs['type'] = 'pysaliency.precomputed_models.predictions'
f.attrs['version'] = '1.0'
f.attrs['downscale_factor'] = downscale_factor   # informational
f.attrs['dtype'] = str(np.dtype(dtype or native_dtype))  # informational
```

In `overwrite=False` (append) mode, if the file already exists, validate that its `type`, `version`, `downscale_factor`, and `dtype` attributes are consistent with the current call and raise `ValueError` if not.

### HDF5 File Format

```
/ (root)
  attrs:
    type = 'pysaliency.precomputed_models.predictions'
    version = '1.0'
    downscale_factor = 2
    dtype = 'float32'

  images/cat.jpg   (dataset shape H/2 × W/2, dtype float32)
    attrs:
      original_shape = (H, W)    # present only when downscale_factor > 1

  images/dog.jpg   (dataset shape H'/2 × W'/2, dtype float32)
    attrs:
      original_shape = (H', W')
```

**Legacy files** (written by current code) have no root attrs and no `original_shape` on datasets. The loader handles them identically to today.

**Non-compact new files** (`dtype=None`, `downscale_factor=1`) have root attrs but no `original_shape` — loader skips upsampling.

The `downscale_factor` and `dtype` root attrs are purely informational. The loader derives behavior from per-dataset `original_shape` (whether to upsample) and the dataset's native HDF5 dtype (whether to upcast).

### Load Side

#### `HDF5SaliencyMapModel`

`__init__` reads and stores `f.attrs.get('version')` for future compat; no other init changes.

`_saliency_map` updated logic:

1. Load raw dataset → `smap` (native HDF5 dtype, stored resolution).
2. If dataset has `original_shape` attr: cast to float64, then upsample via `scipy.ndimage.zoom` with bilinear interpolation (`order=1`) to `original_shape`.
3. Otherwise: return native dtype unchanged (preserves existing behavior for uint8 and legacy files).

Shape check (`check_shape`) runs against the post-upsampling shape, as today.

**Dtype summary:**

| File type | `original_shape` present? | `_saliency_map` returns |
|---|---|---|
| Legacy / uint8 / non-compact | no | native dtype (unchanged) |
| Compact, dtype-reduced only | no | native dtype (float16/32) |
| Compact, downsampled | yes | float64 |

#### `HDF5Model`

Constructor gains a new parameter:

```python
class HDF5Model(Model):
    def __init__(self, stimuli, filename, check_shape=True,
                 max_normalization_error=np.log(1.2), **kwargs):
```

`max_normalization_error` is stored as an instance attribute and can be updated after construction. `None` disables the check entirely.

`_log_density` updated logic:

1. Get `smap` from `parent_model.saliency_map(stimulus)`.
2. Cast to float64.
3. If `max_normalization_error` is not None: check `abs(logsumexp(smap)) < max_normalization_error`; raise `ValueError` if violated (indicates file corruption or unsupported normalization convention — not a soft warning, since a large mass error is a real problem).
4. Renormalize: `smap -= logsumexp(smap)`.
5. Return.

For non-compact files the logsumexp is already ≈ 0, so step 4 is a near-no-op — no behavioral change.

### Interpolation Choices

Motivated by pilot experiments:

- **Downsampling**: area averaging in log/value space (`scipy.ndimage.zoom` with `order=1`, `grid_mode=True`). Empirically outperforms subsampling and probability-space pooling.
- **Upsampling**: bilinear in log/value space (`order=1`). Upsampling strategy has negligible empirical effect; bilinear is fast and avoids the overshoot risk of bicubic.
- No renormalization at export; renormalize only at load (log-density models only), after checking the residual is small.

## Tests

All new tests in `tests/test_precomputed_models.py`. Roundtrip tests are parametrized over `(dtype, downscale_factor)` combinations: `(None, 1)`, `(np.float32, 1)`, `(np.float16, 1)`, `(np.float32, 2)`, `(np.float16, 4)` — for both `SaliencyMapModel` and `Model`.

| Test | Description |
|------|-------------|
| Root attrs | Correct `type`, `version`, `dtype`, `downscale_factor` written |
| Dataset dtype | Stored dtype matches requested `dtype` |
| Dataset shape | Stored shape is `original_shape / downscale_factor` |
| `original_shape` attr | Present iff `downscale_factor > 1`, correct value |
| Append mode mismatch | `ValueError` on inconsistent `downscale_factor` or `dtype` |
| uint8 + `downscale_factor>1` | `UserWarning` emitted |
| uint8 + `dtype=np.float32` | `UserWarning` emitted |
| uint8 + `dtype=np.uint8` | No warning |
| Legacy file load | Native dtype returned, shape matches stimulus — no behavioral change |
| float32 load, no downsampling | float32 returned (native dtype preserved) |
| Downsampled load | float64 returned, shape matches `original_shape` |
| Resized stimuli + `check_shape=False` | Upsamples to `original_shape`, not stimulus shape |
| `HDF5Model` roundtrip | float64 output, logsumexp ≈ 0 after renorm |
| `HDF5Model` threshold | logsumexp before renorm within `log(1.2)` for float16+4× |
| `HDF5Model` corrupted file | `ValueError` raised when logsumexp exceeds threshold |
| Custom `max_normalization_error` | Tighter threshold triggers; looser does not |
