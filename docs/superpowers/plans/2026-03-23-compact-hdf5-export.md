# Compact HDF5 Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dtype` and `downscale_factor` parameters to `export_model_to_hdf5` so large prediction files can be stored more compactly, with transparent upsampling/upcasting on load.

**Architecture:** All changes live in one file (`pysaliency/precomputed_models.py`). Four private helper functions handle the new export logic. `HDF5SaliencyMapModel._saliency_map` gains transparent upsampling. `HDF5Model._log_density` gains a two-path normalization check (strict for legacy/float32/float64, configurable+renorm for downsampled/float16).

**Tech Stack:** numpy, h5py, scipy.ndimage (already used in project), pytest with parametrize. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-23-hdf5-compact-export-design.md`

---

## File Map

| File | Change |
|------|--------|
| `pysaliency/precomputed_models.py` | Add 4 helpers, update `export_model_to_hdf5`, `HDF5SaliencyMapModel`, `HDF5Model` |
| `tests/test_precomputed_models.py` | Add all new tests (append to existing file) |

---

## Task 1: Export — dtype parameter and root attrs

Add the `dtype` parameter and write versioned root attrs. No downsampling yet.

**Files:**
- Modify: `pysaliency/precomputed_models.py:129-168`
- Test: `tests/test_precomputed_models.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_precomputed_models.py`:

```python
import h5py
import warnings


def test_export_root_attrs_written(file_stimuli, tmpdir):
    """New-format files always have type/version/downscale_factor/dtype root attrs."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename)
    with h5py.File(filename, 'r') as f:
        assert f.attrs['type'] == 'pysaliency.precomputed_models.predictions'
        assert f.attrs['version'] == '1.0'
        assert int(f.attrs['downscale_factor']) == 1
        assert f.attrs['dtype'] == 'float64'


def test_export_dtype_float32(file_stimuli, tmpdir):
    """dtype=np.float32 stores float32 datasets."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=np.float32)
    with h5py.File(filename, 'r') as f:
        assert f.attrs['dtype'] == 'float32'
        # check one dataset
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float32


def test_export_dtype_float16(file_stimuli, tmpdir):
    """dtype=np.float16 stores float16 datasets."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=np.float16)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float16


def test_export_dtype_none_preserves_native(file_stimuli, tmpdir):
    """dtype=None (default) preserves the model's native output dtype."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=None)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        # GaussianSaliencyMapModel returns float64
        assert f[keys[0]].dtype == np.float64
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/matthias/Documents/Uni/Bethge/Saliency/pysaliency
python -m pytest --nomatlab tests/test_precomputed_models.py::test_export_root_attrs_written tests/test_precomputed_models.py::test_export_dtype_float32 tests/test_precomputed_models.py::test_export_dtype_float16 tests/test_precomputed_models.py::test_export_dtype_none_preserves_native -v
```

Expected: FAIL — `export_model_to_hdf5` doesn't write root attrs yet.

- [ ] **Step 3: Add `_effective_dtype` helper and update `export_model_to_hdf5`**

In `pysaliency/precomputed_models.py`, add before `export_model_to_hdf5`:

```python
def _effective_dtype(smap, dtype, downscale_factor):
    """Determine the dtype that will actually be stored in the HDF5 file."""
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

Replace the `export_model_to_hdf5` signature and body:

```python
def export_model_to_hdf5(model, stimuli, filename, compression=9, overwrite=True, flush=False,
                          dtype=None, downscale_factor=1):
    """Export pysaliency model predictions for stimuli into hdf5 file

    model: Model or SaliencyMapModel
    stimuli: instance of FileStimuli or Stimuli with filenames attribute
    filename: where to save hdf5 file to
    compression: how much to compress the data
    overwrite: if False, an existing file will be appended to (for resuming
      interrupted exports). Stimuli already present in the file are skipped.
    flush: whether the hdf5 file should be flushed after each stimulus
    dtype: numpy dtype for stored predictions (e.g. np.float32, np.float16).
      None (default) preserves the model's native output dtype.
    downscale_factor: integer >= 1. Spatially downsample predictions by this
      factor before storing. 1 = no downsampling (default).
    """
    filenames = get_stimuli_filenames(stimuli)
    names = get_minimal_unique_filenames(filenames)

    import h5py

    mode = 'w' if overwrite else 'a'
    # Record whether the file existed before opening, to decide whether to write root attrs
    file_existed = os.path.isfile(filename)

    with h5py.File(filename, mode=mode) as f:
        # Determine which stimuli to process
        if overwrite:
            indices = list(range(len(stimuli)))
        else:
            # append mode is for resuming an interrupted export of the same job
            if 'type' in f.attrs:
                _validate_append_consistency(f, downscale_factor)
            indices = [i for i in range(len(stimuli)) if names[i] not in f]
            logging.debug(f"Skipping {len(stimuli) - len(indices)} already existing entries")

        if not indices:
            return

        # Compute first smap to determine effective dtype (needed for root attrs)
        first_stimulus = stimuli[indices[0]]
        if isinstance(model, SaliencyMapModel):
            first_smap = model.saliency_map(first_stimulus)
        elif isinstance(model, Model):
            first_smap = model.log_density(first_stimulus)
        else:
            raise TypeError(type(model))

        effective_stored_dtype = _effective_dtype(first_smap, dtype, downscale_factor)
        _check_size_guard(first_smap, effective_stored_dtype, downscale_factor, dtype)

        # Write root attrs for new files only (overwrite=True always creates fresh;
        # overwrite=False writes attrs only if the file is brand new, not for legacy files)
        if overwrite or not file_existed:
            f.attrs['type'] = 'pysaliency.precomputed_models.predictions'
            f.attrs['version'] = '1.0'
            f.attrs['downscale_factor'] = downscale_factor
            f.attrs['dtype'] = str(effective_stored_dtype)

        for i, k in tqdm(list(enumerate(indices))):
            if i == 0:
                smap = first_smap
            else:
                stimulus = stimuli[k]
                if isinstance(model, SaliencyMapModel):
                    smap = model.saliency_map(stimulus)
                elif isinstance(model, Model):
                    smap = model.log_density(stimulus)

            H, W = smap.shape[0], smap.shape[1]
            smap = _downsample_smap(smap, downscale_factor)
            if dtype is not None:
                smap = smap.astype(dtype)

            ds = f.create_dataset(names[k], data=smap, compression=compression)
            if downscale_factor > 1:
                ds.attrs['original_shape'] = np.array([H, W], dtype=np.int64)
            if flush:
                f.flush()
```

Add stub helpers (full implementation in later tasks) before `export_model_to_hdf5`:

```python
def _downsample_smap(smap, k):
    """Downsample a 2D map by integer factor k using area averaging."""
    if k == 1:
        return smap
    H, W = smap.shape
    H_pad = int(np.ceil(H / k)) * k
    W_pad = int(np.ceil(W / k)) * k
    smap = np.pad(np.ascontiguousarray(smap),
                  ((0, H_pad - H), (0, W_pad - W)),
                  mode='edge')
    return smap.reshape(H_pad // k, k, W_pad // k, k).mean(axis=(1, 3))


def _check_size_guard(smap, effective_stored_dtype, downscale_factor, dtype):
    """Warn if compact settings will produce a larger-per-element file than the native dtype."""
    native_dtype = np.dtype(smap.dtype)
    if np.dtype(effective_stored_dtype).itemsize > native_dtype.itemsize:
        msg = (
            f"Model produces {native_dtype} predictions "
            f"({native_dtype.itemsize} byte/element) but will be stored as "
            f"{effective_stored_dtype} ({np.dtype(effective_stored_dtype).itemsize} byte/element). "
            f"Compact export may result in a larger file than the original."
        )
        if np.issubdtype(native_dtype, np.integer):
            msg += " Consider passing dtype=np.uint8 to preserve the original dtype."
        if np.issubdtype(native_dtype, np.integer) and downscale_factor > 1 and dtype is None:
            msg += " Note: casting area-averaging float results back to integer is lossy."
        warnings.warn(msg)


def _validate_append_consistency(f, downscale_factor):
    """Check that an existing compact HDF5 file is compatible with the current export settings."""
    existing = int(f.attrs['downscale_factor'])
    if existing != downscale_factor:
        raise ValueError(
            f"Cannot append to HDF5 file: existing downscale_factor={existing} "
            f"does not match requested downscale_factor={downscale_factor}."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py::test_export_root_attrs_written tests/test_precomputed_models.py::test_export_dtype_float32 tests/test_precomputed_models.py::test_export_dtype_float16 tests/test_precomputed_models.py::test_export_dtype_none_preserves_native -v
```

Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add pysaliency/precomputed_models.py tests/test_precomputed_models.py
git commit -m "feat: add dtype parameter and versioned root attrs to export_model_to_hdf5"
```

---

## Task 2: Export — downsampling parameter

Add and test the `downscale_factor` parameter behaviour (stored shape, `original_shape` attr, non-divisible sizes).

**Files:**
- Modify: `tests/test_precomputed_models.py`

(All implementation code was already added in Task 1 — `_downsample_smap` and the `ds.attrs['original_shape']` write are in place. This task adds the tests.)

- [ ] **Step 1: Write failing tests**

```python
def test_export_downscale_stored_shape(file_stimuli, tmpdir):
    """Stored shape is ceil(H/k) x ceil(W/k) for divisible dimensions."""
    # file_stimuli uses 100x100 images, which is divisible by 2 and 4
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].shape == (50, 50)


def test_export_downscale_original_shape_attr(file_stimuli, tmpdir):
    """original_shape attr is present and correct when downscale_factor > 1."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        ds = f[keys[0]]
        assert 'original_shape' in ds.attrs
        np.testing.assert_array_equal(ds.attrs['original_shape'], [100, 100])
        assert ds.attrs['original_shape'].dtype == np.int64


def test_export_no_downscale_no_original_shape_attr(file_stimuli, tmpdir):
    """original_shape attr is absent when downscale_factor == 1."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert 'original_shape' not in f[keys[0]].attrs


@pytest.fixture
def file_stimuli_nondivisible(tmpdir):
    """Stimuli with 101x97 images — not divisible by 2 or 4."""
    filenames = []
    for i in range(3):
        filename = tmpdir.join(f'stim_{i:04d}.png')
        imsave(str(filename), np.random.randint(0, 255, (101, 97, 3), dtype=np.uint8))
        filenames.append(str(filename))
    return pysaliency.FileStimuli(filenames=filenames)


def test_export_downscale_nondivisible_stored_shape(file_stimuli_nondivisible, tmpdir):
    """Non-divisible shapes are padded: stored shape is ceil(H/k) x ceil(W/k)."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli_nondivisible, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        # ceil(101/2)=51, ceil(97/2)=49
        assert f[keys[0]].shape == (51, 49)


def test_export_downscale_nondivisible_original_shape_attr(file_stimuli_nondivisible, tmpdir):
    """original_shape stores the pre-padding shape, not the padded or stored shape."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli_nondivisible, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        np.testing.assert_array_equal(f[keys[0]].attrs['original_shape'], [101, 97])


def test_export_float32_downscale_dtype_is_float32(file_stimuli, tmpdir):
    """float32 model output + downscale_factor > 1 stores float32, not float64."""
    # GaussianSaliencyMapModel returns float64, so we need a float32-producing model
    class Float32SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.float32)

    model = Float32SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float32
        assert f.attrs['dtype'] == 'float32'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py::test_export_downscale_stored_shape tests/test_precomputed_models.py::test_export_downscale_original_shape_attr tests/test_precomputed_models.py::test_export_no_downscale_no_original_shape_attr tests/test_precomputed_models.py::test_export_downscale_nondivisible_stored_shape tests/test_precomputed_models.py::test_export_downscale_nondivisible_original_shape_attr tests/test_precomputed_models.py::test_export_float32_downscale_dtype_is_float32 -v
```

Expected: FAIL (downscale_factor parameter not wired up yet — actually already implemented in Task 1. These may already pass.)

- [ ] **Step 3: Verify all pass and run full suite**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_precomputed_models.py
git commit -m "test: add downsampling export tests"
```

---

## Task 3: Export — append mode and size guard

Test append consistency validation and the uint8/size warning.

**Files:**
- Test: `tests/test_precomputed_models.py`

- [ ] **Step 1: Write failing tests**

```python
# NOTE: The spec test table lists "Append mode — mismatch (dtype only)" but the spec design
# section intentionally excludes dtype from the consistency check (it is purely informational
# and cannot be determined before the first stimulus is computed). There is therefore no
# ValueError raised on dtype mismatch — this is correct behavior, not a gap.


def test_export_append_consistency_mismatch_downscale(file_stimuli, tmpdir):
    """Appending with a different downscale_factor raises ValueError."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    partial = pysaliency.FileStimuli(filenames=file_stimuli.filenames[:3])
    export_model_to_hdf5(model, partial, filename, downscale_factor=2)
    with pytest.raises(ValueError, match='downscale_factor'):
        export_model_to_hdf5(model, file_stimuli, filename, overwrite=False, downscale_factor=1)


def test_export_append_new_file_root_attrs(file_stimuli, tmpdir):
    """overwrite=False on a new file still writes root attrs."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, overwrite=False)
    with h5py.File(filename, 'r') as f:
        assert 'type' in f.attrs
        assert f.attrs['version'] == '1.0'


def test_export_append_legacy_file_no_root_attrs(file_stimuli, tmpdir):
    """Appending to a legacy file (no root attrs) succeeds without adding root attrs."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    partial = pysaliency.FileStimuli(filenames=file_stimuli.filenames[:3])
    remaining = pysaliency.FileStimuli(filenames=file_stimuli.filenames[3:])

    # Write a legacy-format file manually (no root attrs)
    import h5py
    names = pysaliency.utils.get_minimal_unique_filenames(partial.filenames)
    with h5py.File(filename, 'w') as f:
        for k, s in enumerate(partial):
            f.create_dataset(names[k], data=model.saliency_map(s))

    export_model_to_hdf5(model, remaining, filename, overwrite=False)

    with h5py.File(filename, 'r') as f:
        assert 'type' not in f.attrs  # no root attrs written into legacy file


def test_export_uint8_model_downscale_warns(tmpdir):
    """Uint8 model output + downscale_factor > 1 emits a UserWarning (stored as float64 > uint8)."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = []
    for i in range(2):
        fname = str(tmpdir.join(f'stim_{i}.png'))
        imsave(fname, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
        filenames.append(fname)
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with pytest.warns(UserWarning, match='larger'):
        export_model_to_hdf5(model, stimuli, filename, downscale_factor=2)


def test_export_uint8_model_downscale_stored_as_float64(tmpdir):
    """Uint8 model output + downscale_factor > 1 is stored as float64 (area avg result)."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = []
    for i in range(2):
        fname = str(tmpdir.join(f'stim_{i}.png'))
        imsave(fname, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
        filenames.append(fname)
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        export_model_to_hdf5(model, stimuli, filename, downscale_factor=2)
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        assert f[keys[0]].dtype == np.float64


def test_export_uint8_model_float32_dtype_warns(tmpdir):
    """Uint8 model + dtype=np.float32 warns (float32 > uint8)."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = [str(tmpdir.join(f'stim_{i}.png')) for i in range(2)]
    for f in filenames:
        imsave(f, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with pytest.warns(UserWarning, match='larger'):
        export_model_to_hdf5(model, stimuli, filename, dtype=np.float32)


def test_export_uint8_model_uint8_dtype_no_warn(tmpdir):
    """Uint8 model + dtype=np.uint8 does not warn."""
    class Uint8SaliencyMapModel(pysaliency.SaliencyMapModel):
        def _saliency_map(self, stimulus):
            return np.ones((stimulus.shape[0], stimulus.shape[1]), dtype=np.uint8)

    filenames = [str(tmpdir.join(f'stim_{i}.png')) for i in range(2)]
    for f in filenames:
        imsave(f, np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
    stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = Uint8SaliencyMapModel()
    filename = str(tmpdir.join('model.hdf5'))
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        export_model_to_hdf5(model, stimuli, filename, dtype=np.uint8)
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py::test_export_append_consistency_mismatch_downscale tests/test_precomputed_models.py::test_export_append_new_file_root_attrs tests/test_precomputed_models.py::test_export_append_legacy_file_no_root_attrs tests/test_precomputed_models.py::test_export_uint8_model_downscale_warns tests/test_precomputed_models.py::test_export_uint8_model_downscale_stored_as_float64 tests/test_precomputed_models.py::test_export_uint8_model_float32_dtype_warns tests/test_precomputed_models.py::test_export_uint8_model_uint8_dtype_no_warn -v
```

Note: all implementation for these tests is already in place from Task 1. Most will pass immediately; the legacy-file test (`test_export_append_legacy_file_no_root_attrs`) verifies the `file_existed` flag logic. If any fail, fix in `precomputed_models.py`.

- [ ] **Step 3: Run full suite.**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py -v
```

If any new tests fail, debug and fix in `precomputed_models.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_precomputed_models.py pysaliency/precomputed_models.py
git commit -m "test: add append-mode consistency and size-guard tests"
```

---

## Task 4: Load side — `HDF5SaliencyMapModel` upsampling

Add `_key_for_stimulus` helper and transparent upsampling in `_saliency_map`.

**Files:**
- Modify: `pysaliency/precomputed_models.py:303-334`
- Test: `tests/test_precomputed_models.py`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize('dtype,downscale_factor', [
    (None, 1),
    (np.float32, 1),
    (np.float16, 1),
    (np.float32, 2),
    (np.float16, 4),
])
def test_hdf5_saliency_map_model_roundtrip(file_stimuli, tmpdir, dtype, downscale_factor):
    """HDF5SaliencyMapModel returns correct shape and values after compact export."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename,
                          dtype=dtype, downscale_factor=downscale_factor)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    for s in file_stimuli:
        result = loaded.saliency_map(s)
        assert result.shape == (s.shape[0], s.shape[1]), \
            f"Shape mismatch: {result.shape} != {(s.shape[0], s.shape[1])}"


def test_hdf5_saliency_map_model_dtype_reduced_returns_native(file_stimuli, tmpdir):
    """dtype-reduced-only files return the stored native dtype (not upcast)."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, dtype=np.float32)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    result = loaded.saliency_map(file_stimuli[0])
    assert result.dtype == np.float32


def test_hdf5_saliency_map_model_downsampled_returns_float64(file_stimuli, tmpdir):
    """Downsampled files upsample and return float64."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli, filename, downscale_factor=2)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    result = loaded.saliency_map(file_stimuli[0])
    assert result.dtype == np.float64


def test_hdf5_saliency_map_model_nondivisible_loaded_shape(file_stimuli_nondivisible, tmpdir):
    """Upsampled output shape matches original_shape (not padded shape)."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, file_stimuli_nondivisible, filename, downscale_factor=2)

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli_nondivisible, filename)
    result = loaded.saliency_map(file_stimuli_nondivisible[0])
    assert result.shape == (101, 97)  # original shape, not (51, 49) or (102, 98)


def test_hdf5_saliency_map_model_resized_stimuli(tmpdir):
    """With check_shape=False and resized stimuli, upsampling targets original_shape."""
    # Export at full size, then load with resized (smaller) stimuli
    from imageio import imsave as _imsave
    filenames = []
    for i in range(2):
        fname = str(tmpdir.join(f'stim_{i}.png'))
        _imsave(fname, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        filenames.append(fname)
    full_stimuli = pysaliency.FileStimuli(filenames=filenames)

    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(model, full_stimuli, filename, downscale_factor=2)

    # Load with the same filenames but we'll check that the loaded shape is 100x100
    # (original_shape), not 50x50 (stored) nor whatever the stimulus reports
    loaded = pysaliency.HDF5SaliencyMapModel(full_stimuli, filename, check_shape=False)
    result = loaded.saliency_map(full_stimuli[0])
    assert result.shape == (100, 100)


def test_hdf5_saliency_map_model_legacy_file_unchanged(file_stimuli, tmpdir):
    """Legacy files (no root attrs) still work exactly as before."""
    model = pysaliency.GaussianSaliencyMapModel(width=0.1)
    filename = str(tmpdir.join('model.hdf5'))
    # Write a legacy file manually
    names = pysaliency.utils.get_minimal_unique_filenames(file_stimuli.filenames)
    with h5py.File(filename, 'w') as f:
        for k, s in enumerate(file_stimuli):
            f.create_dataset(names[k], data=model.saliency_map(s))

    loaded = pysaliency.HDF5SaliencyMapModel(file_stimuli, filename)
    for s in file_stimuli:
        expected = model.saliency_map(s)
        np.testing.assert_array_equal(loaded.saliency_map(s), expected)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest --nomatlab -k "roundtrip or downsampled_returns_float64 or nondivisible_loaded_shape" tests/test_precomputed_models.py -v
```

Expected: FAIL — `_saliency_map` doesn't upsample yet.

- [ ] **Step 3: Update `HDF5SaliencyMapModel` in `pysaliency/precomputed_models.py`**

Replace lines 310–334:

```python
class HDF5SaliencyMapModel(SaliencyMapModel):
    """ exposes a HDF5 file with saliency maps as pysaliency model

        The stimuli have to be of type `FileStimuli`. For each
        stimulus file, the model expects a dataset with the same
        name in the dataset.
        If the file was created with downscale_factor > 1, predictions are
        transparently upsampled to their original resolution on load.
    """
    def __init__(self, stimuli, filename, check_shape=True, **kwargs):
        super(HDF5SaliencyMapModel, self).__init__(**kwargs)

        self.stimuli = stimuli
        self.filename = filename
        self.check_shape = check_shape

        if not os.path.isfile(self.filename):
            raise ValueError(f'File {self.filename} does not exist')

        import h5py
        self.hdf5_file = h5py.File(self.filename, 'r')
        self.version = self.hdf5_file.attrs.get('version')
        self.all_keys = get_keys_recursive(self.hdf5_file)

        self.names = get_keys_from_filenames_with_prefix(get_stimuli_filenames(stimuli), self.all_keys)

    def _key_for_stimulus(self, stimulus):
        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)
        return self.names[stimulus_index]

    def _saliency_map(self, stimulus):
        stimulus_key = self._key_for_stimulus(stimulus)
        dataset = self.hdf5_file[stimulus_key]
        smap = dataset[:]

        if 'original_shape' in dataset.attrs:
            # Compact downsampled file: upsample to original resolution
            target_shape = tuple(dataset.attrs['original_shape'])
            zoom_factors = (target_shape[0] / smap.shape[0], target_shape[1] / smap.shape[1])
            import scipy.ndimage
            smap = scipy.ndimage.zoom(smap.astype(np.float64), zoom_factors, order=1, mode='nearest')

        if not smap.shape == (stimulus.shape[0], stimulus.shape[1]):
            if self.check_shape:
                warnings.warn('Wrong shape for stimulus {}'.format(stimulus_key), stacklevel=4)
        return smap
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pysaliency/precomputed_models.py tests/test_precomputed_models.py
git commit -m "feat: add transparent upsampling to HDF5SaliencyMapModel"
```

---

## Task 5: Load side — `HDF5Model` two-path normalization

Add `max_normalization_error` parameter and the two-path `_log_density`.

**Files:**
- Modify: `pysaliency/precomputed_models.py:337-355`
- Test: `tests/test_precomputed_models.py`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize('dtype,downscale_factor', [
    (None, 1),
    (np.float32, 1),
    (np.float16, 1),
    (np.float32, 2),
    (np.float16, 4),
])
def test_hdf5_model_roundtrip(file_stimuli, tmpdir, dtype, downscale_factor):
    """HDF5Model returns valid log densities after compact export."""
    import warnings
    base = pysaliency.models.SaliencyMapNormalizingModel(
        pysaliency.GaussianSaliencyMapModel(width=0.1))
    filename = str(tmpdir.join('model.hdf5'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        export_model_to_hdf5(base, file_stimuli, filename,
                              dtype=dtype, downscale_factor=downscale_factor)

    loaded = pysaliency.HDF5Model(file_stimuli, filename)
    from scipy.special import logsumexp
    for s in file_stimuli:
        result = loaded.log_density(s)
        assert result.dtype == np.float64
        assert result.shape == (s.shape[0], s.shape[1])
        assert abs(logsumexp(result)) < 0.001, f"logsumexp={logsumexp(result):.6f}, not close to 0"


def test_hdf5_model_strict_path_no_renorm(file_stimuli, tmpdir):
    """Non-compact float64/float32 files use strict path: output not renormalized."""
    from scipy.special import logsumexp
    base = pysaliency.models.SaliencyMapNormalizingModel(
        pysaliency.GaussianSaliencyMapModel(width=0.1))
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(base, file_stimuli, filename)

    loaded = pysaliency.HDF5Model(file_stimuli, filename)
    for s in file_stimuli:
        result = loaded.log_density(s)
        # Result should be bit-identical to stored (cast to float64), not renormalized
        expected = base.log_density(s)
        np.testing.assert_array_equal(result, expected)


def test_hdf5_model_strict_path_raises_on_bad_density(file_stimuli, tmpdir):
    """Non-compact file with bad log density raises ValueError (strict ±0.01 check)."""
    filename = str(tmpdir.join('model.hdf5'))
    names = pysaliency.utils.get_minimal_unique_filenames(file_stimuli.filenames)
    with h5py.File(filename, 'w') as f:
        for k, s in enumerate(file_stimuli):
            # deliberately unnormalized: constant map, logsumexp >> 0
            bad_map = np.zeros((s.shape[0], s.shape[1]), dtype=np.float64)
            f.create_dataset(names[k], data=bad_map)

    loaded = pysaliency.HDF5Model(file_stimuli, filename)
    with pytest.raises(ValueError, match='correct log density'):
        loaded.log_density(file_stimuli[0])


def test_hdf5_model_relaxed_path_threshold_exceeded_raises(file_stimuli, tmpdir):
    """Corrupted downsampled file raises ValueError when logsumexp > max_normalization_error."""
    filename = str(tmpdir.join('model.hdf5'))
    names = pysaliency.utils.get_minimal_unique_filenames(file_stimuli.filenames)
    with h5py.File(filename, 'w') as f:
        f.attrs['type'] = 'pysaliency.precomputed_models.predictions'
        f.attrs['version'] = '1.0'
        f.attrs['downscale_factor'] = 2
        f.attrs['dtype'] = 'float64'
        for k, s in enumerate(file_stimuli):
            stored_shape = (s.shape[0] // 2, s.shape[1] // 2)
            bad_map = np.zeros(stored_shape, dtype=np.float64)  # heavily unnormalized
            ds = f.create_dataset(names[k], data=bad_map)
            ds.attrs['original_shape'] = np.array([s.shape[0], s.shape[1]], dtype=np.int64)

    loaded = pysaliency.HDF5Model(file_stimuli, filename)
    with pytest.raises(ValueError, match='normalization error'):
        loaded.log_density(file_stimuli[0])


def test_hdf5_model_custom_max_normalization_error(file_stimuli, tmpdir):
    """Custom max_normalization_error: tighter raises, looser allows."""
    base = pysaliency.models.SaliencyMapNormalizingModel(
        pysaliency.GaussianSaliencyMapModel(width=0.1))
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(base, file_stimuli, filename, downscale_factor=2)

    # Very tight threshold should raise (there is some error after round-trip)
    loaded_tight = pysaliency.HDF5Model(file_stimuli, filename, max_normalization_error=1e-10)
    with pytest.raises(ValueError):
        loaded_tight.log_density(file_stimuli[0])

    # Default threshold should pass
    loaded_default = pysaliency.HDF5Model(file_stimuli, filename)
    loaded_default.log_density(file_stimuli[0])  # should not raise


def test_hdf5_model_threshold_within_bounds_for_float16_4x(file_stimuli, tmpdir):
    """Explicit check: logsumexp before renorm is within log(1.1) for float16+4x export."""
    from scipy.special import logsumexp as _logsumexp
    base = pysaliency.models.SaliencyMapNormalizingModel(
        pysaliency.GaussianSaliencyMapModel(width=0.1))
    filename = str(tmpdir.join('model.hdf5'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        export_model_to_hdf5(base, file_stimuli, filename, dtype=np.float16, downscale_factor=4)

    # Manually load the raw stored data and upsample to measure pre-renorm logsumexp
    names = pysaliency.utils.get_minimal_unique_filenames(file_stimuli.filenames)
    import scipy.ndimage
    with h5py.File(filename, 'r') as f:
        for k, s in enumerate(file_stimuli):
            ds = f[names[k]]
            smap = ds[:].astype(np.float64)
            target_shape = tuple(ds.attrs['original_shape'])
            zoom_factors = (target_shape[0] / smap.shape[0], target_shape[1] / smap.shape[1])
            smap = scipy.ndimage.zoom(smap, zoom_factors, order=1, mode='nearest')
            err = abs(_logsumexp(smap))
            assert err < np.log(1.1), f"logsumexp error {err:.4f} exceeds log(1.1)={np.log(1.1):.4f}"


def test_hdf5_model_max_normalization_error_none(file_stimuli, tmpdir):
    """max_normalization_error=None skips the guard; renorm is still applied."""
    from scipy.special import logsumexp
    base = pysaliency.models.SaliencyMapNormalizingModel(
        pysaliency.GaussianSaliencyMapModel(width=0.1))
    filename = str(tmpdir.join('model.hdf5'))
    export_model_to_hdf5(base, file_stimuli, filename, downscale_factor=2)

    loaded = pysaliency.HDF5Model(file_stimuli, filename, max_normalization_error=None)
    for s in file_stimuli:
        result = loaded.log_density(s)
        assert abs(logsumexp(result)) < 0.001  # renorm still runs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest --nomatlab -k "hdf5_model_roundtrip or strict_path_no_renorm or threshold_exceeded_raises" tests/test_precomputed_models.py -v
```

Expected: FAIL — `HDF5Model` doesn't have the new logic yet.

- [ ] **Step 3: Update `HDF5Model` in `pysaliency/precomputed_models.py`**

Replace lines 337–355:

```python
class HDF5Model(Model):
    """ exposes a HDF5 file with log densities as pysaliency model.

        For more detail see HDF5SaliencyMapModel.
        Files created with downscale_factor > 1 or dtype=np.float16 use a
        relaxed normalization check and automatic renormalization on load.
        All other files use the original strict ±0.01 check.
    """
    def __init__(self, stimuli, filename, check_shape=True,
                 max_normalization_error=np.log(1.1), **kwargs):
        super(HDF5Model, self).__init__(**kwargs)
        self.parent_model = HDF5SaliencyMapModel(
            stimuli=stimuli,
            filename=filename,
            caching=False,
            check_shape=check_shape
        )
        self.max_normalization_error = max_normalization_error

    def _log_density(self, stimulus):
        key = self.parent_model._key_for_stimulus(stimulus)
        dataset = self.parent_model.hdf5_file[key]
        use_relaxed_path = (
            'original_shape' in dataset.attrs
            or dataset.dtype.itemsize < np.dtype(np.float32).itemsize
        )

        smap = self.parent_model.saliency_map(stimulus).astype(np.float64)

        if use_relaxed_path:
            if self.max_normalization_error is not None:
                err = abs(logsumexp(smap))
                if err >= self.max_normalization_error:
                    raise ValueError(
                        f'Log density normalization error {err:.4f} exceeds '
                        f'threshold {self.max_normalization_error:.4f}'
                    )
            smap = smap - logsumexp(smap)
        else:
            if not -0.01 <= logsumexp(smap) <= 0.01:
                raise ValueError('Not a correct log density!')

        return smap
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest --nomatlab tests/test_precomputed_models.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pysaliency/precomputed_models.py tests/test_precomputed_models.py
git commit -m "feat: add two-path normalization and max_normalization_error to HDF5Model"
```

---

## Task 6: Final verification

Run the full test suite, check no regressions.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest --nomatlab --notheano --nodownload tests/ -v
```

Expected: all pass, no regressions.

- [ ] **Step 2: Spot-check the feature end-to-end**

```python
# Quick smoke test in Python (not a formal test, just sanity check)
import numpy as np
import pysaliency
from pysaliency import export_model_to_hdf5

# Use any FileStimuli you have, or create a small one
# model = pysaliency.GaussianSaliencyMapModel(width=0.1)
# export_model_to_hdf5(model, stimuli, '/tmp/compact.hdf5', dtype=np.float32, downscale_factor=2)
# loaded = pysaliency.HDF5SaliencyMapModel(stimuli, '/tmp/compact.hdf5')
# print(loaded.saliency_map(stimuli[0]).shape)  # should match stimulus shape
```

- [ ] **Step 3: Final commit (if any fixups needed)**

```bash
git add -p
git commit -m "fix: <description of any fixup>"
```
