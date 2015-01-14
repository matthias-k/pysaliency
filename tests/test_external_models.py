from __future__ import absolute_import, print_function, division

import os
import shutil
import unittest
from itertools import product

from nose.tools import assert_equal, nottest

import numpy as np

import pysaliency
import pysaliency.external_datasets


@nottest
def create_test_stimuli():
    color_stimulus = np.load(os.path.join('tests', 'external_models', 'color_stimulus.npy'))
    grayscale_stimulus = np.load(os.path.join('tests', 'external_models', 'grayscale_stimulus.npy'))
    return color_stimulus, grayscale_stimulus


class DataStore(object):
    def __init__(self):
        self.color_stimulus, self.grayscale_stimulus = create_test_stimuli()
        self.old_matlab_names = pysaliency.utils.MatlabOptions.matlab_names
        self.old_octave_names = pysaliency.utils.MatlabOptions.octave_names


def setUp():
    DataStore.ds = DataStore()


class ModelTemplate(object):
    data_path = 'test_data'
    matlab_names = ['matlab', 'octave']

    def setUp(self):
        if os.path.isdir(self.data_path):
            shutil.rmtree(self.data_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def tearDown(self):
        shutil.rmtree(self.data_path)

    def get_model(self, location, matlab):
        if matlab == 'octave':
            pysaliency.utils.MatlabOptions.matlab_names = []
            pysaliency.utils.MatlabOptions.octave_names = DataStore.ds.old_octave_names
        else:
            pysaliency.utils.MatlabOptions.matlab_names = DataStore.ds.old_matlab_names
            pysaliency.utils.MatlabOptions.octave_names = []
        model = self.create_model(location)
        return model

    def create_model(self, location):
        raise NotImplementedError()

    @property
    def params(self):
        return product([None, self.data_path], self.matlab_names)

    def test_saliency_maps(self):
        for location, matlab in self.params:
            yield self.check_saliency_maps, location, matlab

    def check_saliency_maps(self, location, matlab):
        model = self.get_model(location, matlab)
        print('Testing color')
        color_stimulus = DataStore.ds.color_stimulus
        saliency_map = model.saliency_map(color_stimulus)
        np.testing.assert_allclose(saliency_map,
                                   np.load(os.path.join('tests', 'external_models', '{}_color_stimulus.npy'.format(model.__modelname__))))
        print('Testing Grayscale')
        grayscale_stimulus = DataStore.ds.grayscale_stimulus
        saliency_map = model.saliency_map(grayscale_stimulus)
        np.testing.assert_allclose(saliency_map,
                                   np.load(os.path.join('tests', 'external_models', '{}_grayscale_stimulus.npy'.format(model.__modelname__))))


class TestAIM(ModelTemplate):
    def create_model(self, location):
        return pysaliency.AIM(location=location)


class TestSUN(ModelTemplate):
    matlab_names = ['matlab']

    def create_model(self, location):
        return pysaliency.SUN(location=location)


class TestContextAwareSaliency(ModelTemplate):
    def create_model(self, location):
        return pysaliency.ContextAwareSaliency(location=location)


#class TestBMS(ModelTemplate):
#    def create_model(self, location):
#        return pysaliency.BMS(location=location)

class TestGBVS(ModelTemplate):
    matlab_names = ['matlab']

    def create_model(self, location):
        return pysaliency.GBVS(location=location)


class TestGBVSIttiKoch(ModelTemplate):
    matlab_names = ['matlab']

    def create_model(self, location):
        return pysaliency.GBVSIttiKoch(location=location)


class TestJudd(ModelTemplate):
    matlab_names = ['matlab']

    def create_model(self, location):
        return pysaliency.Judd(location=location, saliency_toolbox_archive='SaliencyToolbox2.3.zip')


class TestIttiKoch(ModelTemplate):
    matlab_names = ['matlab']

    def create_model(self, location):
        return pysaliency.IttiKoch(location=location, saliency_toolbox_archive='SaliencyToolbox2.3.zip')

if __name__ == '__main__':
    unittest.main()
