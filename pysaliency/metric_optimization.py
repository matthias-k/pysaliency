from __future__ import print_function, division, absolute_import, unicode_literals

from .saliency_map_models import SaliencyMapModel


class SIMSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model,
                 kernel_size,
                 train_samples_per_epoch=1000, val_samples=1000,
                 train_seed=43, val_seed=42,
                 fixation_count=100, batch_size=50,
                 max_batch_size=None,
                 initial_learning_rate=1e-7,
                 backlook=1,
                 min_iter=0,
                 max_iter=1000,
                 truncate_gaussian=3,
                 learning_rate_decay_samples=None,
                 learning_rate_decay_scheme=None,
                 learning_rate_decay_ratio=0.333333333,
                 minimum_learning_rate=1e-11,
                 initial_model=None,
                 verbose=True,
                 session_config=None,
                 library='torch',
                 **kwargs
                 ):
        super(SIMSaliencyMapModel, self).__init__(**kwargs)
        self.parent_model = parent_model

        self.kernel_size = kernel_size
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples = val_samples
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.fixation_count = fixation_count
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.initial_learning_rate = initial_learning_rate
        self.backlook = backlook
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.truncate_gaussian = truncate_gaussian
        self.learning_rate_decay_samples = learning_rate_decay_samples
        self.learning_rate_decay_scheme = learning_rate_decay_scheme
        self.learning_rate_decay_ratio = learning_rate_decay_ratio
        self.minimum_learning_rate = minimum_learning_rate
        self.initial_model = initial_model
        self.verbose = verbose
        self.session_config = session_config
        self.library = library

    def _saliency_map(self, stimulus):
        log_density = self.parent_model.log_density(stimulus)

        if self.initial_model:
            initial_saliency_map = self.initial_model.saliency_map(stimulus)
        else:
            initial_saliency_map = None

        if self.library.lower() == 'tensorflow':
            from .metric_optimization_tf import maximize_expected_sim
        elif self.library.lower() == 'torch':
            from .metric_optimization_torch import maximize_expected_sim
        else:
            raise ValueError(self.library)



        saliency_map, val_scores = maximize_expected_sim(
            log_density,
            kernel_size=self.kernel_size,
            train_samples_per_epoch=self.train_samples_per_epoch,
            val_samples=self.val_samples,
            train_seed=self.train_seed,
            val_seed=self.val_seed,
            fixation_count=self.fixation_count,
            batch_size=self.batch_size,
            max_batch_size=self.max_batch_size,
            verbose=self.verbose,
            session_config=self.session_config,
            initial_learning_rate=self.initial_learning_rate,
            backlook=self.backlook,
            min_iter=self.min_iter,
            max_iter=self.max_iter,
            truncate_gaussian=self.truncate_gaussian,
            learning_rate_decay_samples=self.learning_rate_decay_samples,
            initial_saliency_map=initial_saliency_map,
            learning_rate_decay_scheme=self.learning_rate_decay_scheme,
            learning_rate_decay_ratio=self.learning_rate_decay_ratio,
            minimum_learning_rate=self.minimum_learning_rate
        )
        return saliency_map
