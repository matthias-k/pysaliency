import numpy as np
import torch

from ..models import Model, ScanpathModel
from ..datasets import as_stimulus



class StaticDeepGazeModel(Model):
    def __init__(self, centerbias_model, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centerbias_model = centerbias_model
        self.torch_model = self._load_model()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model.to(self.device)

    def _load_model(self):
        raise NotImplementedError()

    def _log_density(self, stimulus):
        stimulus = as_stimulus(stimulus)
        stimulus_data = stimulus.stimulus_data

        if stimulus_data.ndim == 2:
            stimulus_data = np.dstack((stimulus_data, stimulus_data, stimulus_data))

        stimulus_data = stimulus_data.transpose(2, 0, 1)

        centerbias_data = self.centerbias_model.log_density(stimulus)

        image_tensor = torch.tensor(np.array([stimulus_data]), dtype=torch.float32).to(self.device)
        centerbias_tensor = torch.tensor(np.array([centerbias_data]), dtype=torch.float32).to(self.device)

        log_density_prediction = self.torch_model.forward(image_tensor, centerbias_tensor)

        return log_density_prediction.detach().cpu().numpy()[0].astype(np.float64)


class DeepGazeI(StaticDeepGazeModel):
    """DeepGaze I model

    see https://github.com/matthias-k/DeepGaze and

    DeepGaze I: Kümmerer, M., Theis, L., & Bethge, M. (2015).
    Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet.
    ICLR Workshop Track (http://arxiv.org/abs/1411.1045)
    """
    def __init__(self, centerbias_model, device=None, *args, **kwargs):
        super().__init__(centerbias_model=centerbias_model, *args, **kwargs)

    def _load_model(self):
        return torch.hub.load('matthias-k/DeepGaze', 'DeepGazeI', pretrained=True)


class DeepGazeIIE(StaticDeepGazeModel):
    """DeepGaze IIE model

    see https://github.com/matthias-k/DeepGaze and

    DeepGaze IIE: Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021).
    Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling.
    ICCV 2021 (http://arxiv.org/abs/2105.12441)
    """
    def __init__(self, centerbias_model, device=None, *args, **kwargs):
        super().__init__(centerbias_model=centerbias_model, *args, **kwargs)

    def _load_model(self):
        return torch.hub.load('matthias-k/DeepGaze', 'DeepGazeIIE', pretrained=True)

    def _log_density(self, stimulus):
        return super()._log_density(stimulus)[0]
