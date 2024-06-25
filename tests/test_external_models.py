from __future__ import absolute_import, print_function, division

import os

import pytest

import numpy as np

import pysaliency
import pysaliency.external_datasets


@pytest.fixture(scope='module')
def color_stimulus():
    return np.load(os.path.join('tests', 'external_models', 'color_stimulus.npy'))


@pytest.fixture(scope='module')
def grayscale_stimulus():
    return np.load(os.path.join('tests', 'external_models', 'grayscale_stimulus.npy'))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_AIM(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.AIM(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))),
                               rtol=1e-5)


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_SUN(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.SUN(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))),
                               rtol=1e-5)


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_ContextAwareSaliency(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.ContextAwareSaliency(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_GBVS(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.GBVS(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_GBVSIttiKoch(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.GBVSIttiKoch(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_Judd(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.Judd(location=str(tmpdir), saliency_toolbox_archive='SaliencyToolbox2.3.zip')
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_IttiKoch(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.IttiKoch(location=str(tmpdir), saliency_toolbox_archive='SaliencyToolbox2.3.zip')
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_RARE2007(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.external_models.RARE2007(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_RARE2012(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.RARE2012(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


@pytest.mark.skip_octave
@pytest.mark.matlab
def test_CovSal(tmpdir, matlab, color_stimulus, grayscale_stimulus):
    model = pysaliency.CovSal(location=str(tmpdir))
    print('Testing color')
    saliency_map = model.saliency_map(color_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
    print('Testing Grayscale')
    saliency_map = model.saliency_map(grayscale_stimulus)
    np.testing.assert_allclose(saliency_map,
                               np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))
