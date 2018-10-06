from __future__ import absolute_import, division, print_function, unicode_literals

#import .datasets as datasets
#from . import models
#import .utils

from . import datasets
from . import saliency_map_models
from . import models
from . import external_models
from . import external_datasets

from .datasets import (Fixations, FixationTrains, Stimuli, FileStimuli, create_nonfixations, create_subset, remove_out_of_stimulus_fixations,
                       concatenate_datasets, read_hdf5)
from .saliency_map_models import (SaliencyMapModel, GeneralSaliencyMapModel, FixationMap, CachedSaliencyMapModel, ExpSaliencyMapModel,
                                  DisjointUnionSaliencyMapModel, SubjectDependentSaliencyMapModel, ResizingSaliencyMapModel,
                                  BluringSaliencyMapModel, DigitizeMapModel, HistogramNormalizedSaliencyMapModel,
                                  export_model_to_hdf5)
from .models import (GeneralModel, Model, UniformModel, CachedModel, MixtureModel,
                     DisjointUnionModel, SubjectDependentModel, ShuffledAUCSaliencyMapModel, ResizingModel,
                     StimulusDependentModel
                     )
from .saliency_map_conversion import SaliencyMapConvertor, JointSaliencyMapConvertor, optimize_for_information_gain
from .precomputed_models import (SaliencyMapModelFromFiles,
                                 SaliencyMapModelFromDirectory,
                                 SaliencyMapModelFromFile,
                                 ModelFromDirectory,
                                 HDF5SaliencyMapModel,
                                 HDF5Model)

#from .stationary_models import StationarySaliencyModel
from .external_models import AIM, SUN, ContextAwareSaliency, BMS, GBVS, GBVSIttiKoch, Judd, IttiKoch, RARE2012, CovSal
from .external_datasets import (get_mit1003, get_mit1003_onesize,
                                get_cat2000_train, get_cat2000_test,
                                get_toronto,
                                get_iSUN_training, get_iSUN_validation, get_iSUN_testing,
                                get_SALICON, get_SALICON_train, get_SALICON_val, get_SALICON_test,
                                get_mit300,
                                get_koehler,
                                get_FIGRIM,
                                get_OSIE,
                                get_NUSEF_public)

from .metric_optimization import maximize_expected_sim, SIMSaliencyMapModel
