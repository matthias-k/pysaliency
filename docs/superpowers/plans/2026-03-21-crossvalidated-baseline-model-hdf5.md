# CrossvalidatedBaselineModel HDF5 Save/Reload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `to_hdf5`/`read_hdf5` to `CrossvalidatedBaselineModel` so it can be serialized by parameters (not predictions), wired into the `pysaliency.read_hdf5` dispatcher.

**Architecture:** Refactor `CrossvalidatedBaselineModel` to store only `fixations_n` (not the full fixations object), then add `to_hdf5`/`read_hdf5` following the exact same pattern as the already-complete `BaselineModel`. Register in the `_MODEL_READERS` dispatch table in `pysaliency/hdf5.py`.

**Tech Stack:** Python, h5py, numpy, pytest. Build Cython extensions with `make cython` before running tests.

---

## File Map

| File | Change |
|------|--------|
| `pysaliency/baseline_utils.py` | Refactor `CrossvalidatedBaselineModel.__init__` and `_log_density`; add `to_hdf5` and `read_hdf5` |
| `pysaliency/hdf5.py` | Add `_read_crossvalidated_baseline_model` helper; extend existing `_MODEL_READERS` dict |
| `tests/test_baseline_utils.py` | Extend top-level import; add 6 new tests |
| `tests/test_hdf5_io.py` | Extend top-level import; add 3 new tests for dispatcher, kwarg override, and cache bypass |

---

## Task 1: Refactor `CrossvalidatedBaselineModel` internals

Replace `self.fixations` + `self.shape_cache` with `self.fixations_n`. Public constructor signature is unchanged.

**Files:**
- Modify: `pysaliency/baseline_utils.py:559-591`
- Modify: `tests/test_baseline_utils.py`

- [ ] **Step 1: Update the top-level import in `tests/test_baseline_utils.py`**

The existing import block (lines 9–17) imports from `pysaliency.baseline_utils`. Add `CrossvalidatedBaselineModel` to it:

```python
from pysaliency.baseline_utils import (
    BaselineModel,
    CrossvalidatedBaselineModel,
    CrossvalMultipleRegularizations,
    GeneralMixtureKernelDensityEstimator,
    KDEGoldModel,
    MixtureKernelDensityEstimator,
    ScikitLearnImageCrossValidationGenerator,
    fill_fixation_map,
)
```

- [ ] **Step 2: Write a failing test that asserts the new internal state**

Add to `tests/test_baseline_utils.py`:

```python
def test_crossvalidated_baseline_model_stores_fixations_n(stimuli, scanpath_fixations):
    model = CrossvalidatedBaselineModel(stimuli, scanpath_fixations, bandwidth=0.1)
    assert hasattr(model, 'fixations_n')
    assert not hasattr(model, 'fixations')
    assert not hasattr(model, 'shape_cache')
    np.testing.assert_array_equal(model.fixations_n, scanpath_fixations.n)
```

- [ ] **Step 3: Run the test to confirm it fails**

```bash
python -m pytest --nomatlab tests/test_baseline_utils.py::test_crossvalidated_baseline_model_stores_fixations_n -v
```

Expected: FAIL — `model` has no `fixations_n`, still has `fixations` and `shape_cache`.

- [ ] **Step 4: Implement the refactor**

In `pysaliency/baseline_utils.py`, replace the `CrossvalidatedBaselineModel` class body (lines 559–591) with:

```python
class CrossvalidatedBaselineModel(Model):
    def __init__(self, stimuli, fixations, bandwidth, eps=1e-20, **kwargs):
        super(CrossvalidatedBaselineModel, self).__init__(**kwargs)
        self.stimuli = stimuli
        self.bandwidth = bandwidth
        self.eps = eps
        self.xs, self.ys = normalize_fixations(stimuli, fixations)
        self.fixations_n = fixations.n.copy()

    def _log_density(self, stimulus):
        shape = stimulus.shape[0], stimulus.shape[1]

        stimulus_id = get_image_hash(stimulus)
        stimulus_index = self.stimuli.stimulus_ids.index(stimulus_id)

        inds = self.fixations_n != stimulus_index

        ZZ = np.zeros(shape)

        _fixations = np.array([self.ys[inds]*shape[0], self.xs[inds]*shape[1]]).T
        fill_fixation_map(ZZ, _fixations)
        ZZ = gaussian_filter(ZZ, [self.bandwidth*shape[0], self.bandwidth*shape[1]])
        ZZ *= (1-self.eps)
        ZZ += self.eps * 1.0/(shape[0]*shape[1])
        ZZ = np.log(ZZ)

        ZZ -= logsumexp(ZZ)

        return ZZ
```

- [ ] **Step 5: Run the new test plus the full baseline_utils suite**

```bash
python -m pytest --nomatlab tests/test_baseline_utils.py -v
```

Expected: all pass. If any existing test relied on `model.fixations` or `model.shape_cache`, it will fail here — fix it in the same commit.

- [ ] **Step 6: Commit**

```bash
git add pysaliency/baseline_utils.py tests/test_baseline_utils.py
git commit -m "refactor: CrossvalidatedBaselineModel stores fixations_n, removes shape_cache"
```

---

## Task 2: Add `to_hdf5` to `CrossvalidatedBaselineModel`

**Files:**
- Modify: `pysaliency/baseline_utils.py`
- Modify: `tests/test_baseline_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_baseline_utils.py`. Note: `import h5py` goes inside the test function body, matching the style of the existing `test_baseline_model_hdf5_type` test:

```python
def test_crossvalidated_baseline_model_hdf5_type(tmp_path, stimuli, scanpath_fixations):
    model = CrossvalidatedBaselineModel(stimuli, scanpath_fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_type.hdf5'
    model.to_hdf5(path)

    import h5py
    with h5py.File(path, 'r') as f:
        value = f.attrs['type']
        if not isinstance(value, str):
            value = value.decode('utf8')
        assert value == 'pysaliency.baseline_utils.CrossvalidatedBaselineModel'
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest --nomatlab tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_type -v
```

Expected: FAIL — `CrossvalidatedBaselineModel` has no `to_hdf5`.

- [ ] **Step 3: Implement `to_hdf5`**

Add the method to `CrossvalidatedBaselineModel` in `pysaliency/baseline_utils.py`, immediately after `_log_density`. `hdf5_wrapper` is already imported from `.datasets.utils` at the top of the file:

```python
    @hdf5_wrapper(mode='w')
    def to_hdf5(self, target, include_stimuli=True):
        target.attrs['type'] = np.bytes_('pysaliency.baseline_utils.CrossvalidatedBaselineModel')
        target.attrs['version'] = np.bytes_('1.0')
        target.attrs['bandwidth'] = self.bandwidth
        target.attrs['eps'] = self.eps
        target.create_dataset('xs', data=self.xs)
        target.create_dataset('ys', data=self.ys)
        target.create_dataset('fixations_n', data=self.fixations_n)

        if include_stimuli:
            stimuli_group = target.create_group('stimuli')
            self.stimuli.to_hdf5(stimuli_group)
```

- [ ] **Step 4: Run the type test**

```bash
python -m pytest --nomatlab tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_type -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pysaliency/baseline_utils.py tests/test_baseline_utils.py
git commit -m "feat: add CrossvalidatedBaselineModel.to_hdf5"
```

---

## Task 3: Add `read_hdf5` to `CrossvalidatedBaselineModel` with full roundtrip tests

**Files:**
- Modify: `pysaliency/baseline_utils.py`
- Modify: `tests/test_baseline_utils.py`

- [ ] **Step 1: Write all four failing tests at once**

Add to `tests/test_baseline_utils.py`:

```python
def test_crossvalidated_baseline_model_hdf5_roundtrip_with_stimuli(tmp_path, stimuli, scanpath_fixations):
    model = CrossvalidatedBaselineModel(stimuli, scanpath_fixations, bandwidth=0.1)
    first_prediction = model.log_density(stimuli[0]).copy()

    path = tmp_path / 'cv_baseline.hdf5'
    model.to_hdf5(path)

    reloaded = CrossvalidatedBaselineModel.read_hdf5(path)
    np.testing.assert_allclose(reloaded.log_density(stimuli[0]), first_prediction)


def test_crossvalidated_baseline_model_hdf5_roundtrip_without_stimuli(tmp_path, stimuli, scanpath_fixations):
    model = CrossvalidatedBaselineModel(stimuli, scanpath_fixations, bandwidth=0.1)
    first_prediction = model.log_density(stimuli[1]).copy()

    path = tmp_path / 'cv_baseline_no_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=False)

    reloaded = CrossvalidatedBaselineModel.read_hdf5(path, stimuli=stimuli)
    np.testing.assert_allclose(reloaded.log_density(stimuli[1]), first_prediction)


def test_crossvalidated_baseline_model_hdf5_error_without_stimuli(tmp_path, stimuli, scanpath_fixations):
    model = CrossvalidatedBaselineModel(stimuli, scanpath_fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_no_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=False)

    with pytest.raises(ValueError, match='stimuli'):
        CrossvalidatedBaselineModel.read_hdf5(path)


def test_crossvalidated_baseline_model_hdf5_stimuli_kwarg_overrides_embedded(tmp_path, stimuli, scanpath_fixations):
    """stimuli= kwarg takes precedence over the embedded stimuli group."""
    model = CrossvalidatedBaselineModel(stimuli, scanpath_fixations, bandwidth=0.1)
    first_prediction = model.log_density(stimuli[0]).copy()

    path = tmp_path / 'cv_baseline_with_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=True)

    # Pass the same stimuli explicitly — should still work (kwarg wins)
    reloaded = CrossvalidatedBaselineModel.read_hdf5(path, stimuli=stimuli)
    assert reloaded.stimuli is stimuli
    np.testing.assert_allclose(reloaded.log_density(stimuli[0]), first_prediction)
```

- [ ] **Step 2: Run to confirm all four fail**

```bash
python -m pytest --nomatlab \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_roundtrip_with_stimuli \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_roundtrip_without_stimuli \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_error_without_stimuli \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_stimuli_kwarg_overrides_embedded \
  -v
```

Expected: all FAIL — `CrossvalidatedBaselineModel` has no `read_hdf5`.

- [ ] **Step 3: Implement `read_hdf5`**

Add to `CrossvalidatedBaselineModel` in `pysaliency/baseline_utils.py`. Decorator stack: `@classmethod` outermost, `@hdf5_wrapper(mode='r')` inner — this matches `BaselineModel.read_hdf5` exactly. `Model` is already available at module level via `from . import Model`.

```python
    @classmethod
    @hdf5_wrapper(mode='r')
    def read_hdf5(
        cls,
        source,
        *,
        stimuli=None,
        caching=True,
        memory_cache_size=None,
        cache_location=None,
    ):
        from .hdf5 import read_hdf5 as _read_hdf5

        data_type = decode_string(source.attrs['type'])
        data_version = decode_string(source.attrs['version'])

        if data_type != 'pysaliency.baseline_utils.CrossvalidatedBaselineModel':
            raise ValueError("Invalid type! Expected 'pysaliency.baseline_utils.CrossvalidatedBaselineModel', got", data_type)
        if data_version != '1.0':
            raise ValueError("Invalid version! Expected '1.0', got", data_version)

        if stimuli is None:
            if 'stimuli' not in source:
                raise ValueError(
                    "No stimuli found in HDF5 file. Pass stimuli= explicitly."
                )
            stimuli = _read_hdf5(source['stimuli'])

        model = cls.__new__(cls)
        Model.__init__(model, cache_location=cache_location, caching=caching, memory_cache_size=memory_cache_size)
        model.bandwidth = source.attrs['bandwidth']
        model.eps = source.attrs['eps']
        model.xs = source['xs'][...]
        model.ys = source['ys'][...]
        model.fixations_n = source['fixations_n'][...]
        model.stimuli = stimuli

        return model
```

- [ ] **Step 4: Run all five tests (type + 4 roundtrip/override)**

```bash
python -m pytest --nomatlab \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_type \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_roundtrip_with_stimuli \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_roundtrip_without_stimuli \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_error_without_stimuli \
  tests/test_baseline_utils.py::test_crossvalidated_baseline_model_hdf5_stimuli_kwarg_overrides_embedded \
  -v
```

Expected: all PASS.

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
python -m pytest --nomatlab tests/test_baseline_utils.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pysaliency/baseline_utils.py tests/test_baseline_utils.py
git commit -m "feat: add CrossvalidatedBaselineModel.read_hdf5 with optional stimuli embedding"
```

---

## Task 4: Register in `pysaliency.read_hdf5` dispatcher and add integration tests

**Files:**
- Modify: `pysaliency/hdf5.py`
- Modify: `tests/test_hdf5_io.py`

- [ ] **Step 1: Update the import in `tests/test_hdf5_io.py`**

The existing file imports `BaselineModel` from `pysaliency.baseline_utils`. Extend it to also import `CrossvalidatedBaselineModel`:

```python
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
```

- [ ] **Step 2: Write the three failing dispatcher tests**

Add to `tests/test_hdf5_io.py`:

```python
def test_unified_read_hdf5_reads_crossvalidated_baseline_model(tmp_path):
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_model.hdf5'
    model.to_hdf5(path)

    loaded = pysaliency.read_hdf5(path)
    assert isinstance(loaded, CrossvalidatedBaselineModel)
    np.testing.assert_allclose(loaded.log_density(stimuli[0]), model.log_density(stimuli[0]))


def test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg(tmp_path):
    """stimuli= kwarg is forwarded correctly through the dispatcher."""
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_no_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=False)

    loaded = pysaliency.read_hdf5(str(path), stimuli=stimuli)
    assert isinstance(loaded, CrossvalidatedBaselineModel)
    assert loaded.stimuli is stimuli
    np.testing.assert_allclose(loaded.log_density(stimuli[0]), model.log_density(stimuli[0]))


def test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg_bypasses_cache(tmp_path):
    """Regression: stimuli= kwarg must bypass WeakValueDictionary cache in _read_hdf5_from_file."""
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_no_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=False)

    loaded = pysaliency.read_hdf5(str(path), stimuli=stimuli)
    assert loaded.stimuli is stimuli

    # Second call with a different stimuli object — must not return the cached first result
    stimuli2 = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    loaded2 = pysaliency.read_hdf5(str(path), stimuli=stimuli2)
    assert loaded2.stimuli is stimuli2
    np.testing.assert_allclose(loaded2.log_density(stimuli2[0]), CrossvalidatedBaselineModel(stimuli2, fixations, bandwidth=0.1).log_density(stimuli2[0]))
```

- [ ] **Step 3: Run to confirm all three fail**

```bash
python -m pytest --nomatlab \
  tests/test_hdf5_io.py::test_unified_read_hdf5_reads_crossvalidated_baseline_model \
  tests/test_hdf5_io.py::test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg \
  tests/test_hdf5_io.py::test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg_bypasses_cache \
  -v
```

Expected: all FAIL — `CrossvalidatedBaselineModel` not in `_MODEL_READERS`.

- [ ] **Step 4: Update `pysaliency/hdf5.py`**

Add `_read_crossvalidated_baseline_model` after `_read_baseline_model`, then **replace** the existing `_MODEL_READERS` dict definition with the extended version:

```python
def _read_crossvalidated_baseline_model(source, **kwargs):
    from .baseline_utils import CrossvalidatedBaselineModel
    return CrossvalidatedBaselineModel.read_hdf5(source, **kwargs)


_MODEL_READERS = {
    'pysaliency.baseline_utils.BaselineModel': _read_baseline_model,
    'pysaliency.baseline_utils.CrossvalidatedBaselineModel': _read_crossvalidated_baseline_model,
}
```

- [ ] **Step 5: Run all three dispatcher tests**

```bash
python -m pytest --nomatlab \
  tests/test_hdf5_io.py::test_unified_read_hdf5_reads_crossvalidated_baseline_model \
  tests/test_hdf5_io.py::test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg \
  tests/test_hdf5_io.py::test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg_bypasses_cache \
  -v
```

Expected: all PASS.

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest --nomatlab tests/ -v
```

Expected: all pass. Fix any regressions before committing.

- [ ] **Step 7: Commit**

```bash
git add pysaliency/hdf5.py tests/test_hdf5_io.py
git commit -m "feat: register CrossvalidatedBaselineModel in pysaliency.read_hdf5 dispatcher"
```
