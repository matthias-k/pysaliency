from .models import ScanpathModel
from .saliency_map_models import ScanpathSaliencyMapModel
from PIL import Image
from io import BytesIO
import requests
import json
import numpy as np
import orjson

from .datasets import as_stimulus

class HTTPScanpathModel(ScanpathModel):
    """
    A scanpath model that uses a HTTP server to make predictions.
    
    The model is provided with an URL where it expects a server with the following API:
    
    /conditional_log_density: expects a POST request with a file attachtment `stimulus` 
    containing the stimulus and a json body containing x_hist, y_hist, t_hist and a dictionary with other attributes
    /type: returns the model type and version
    """
    def __init__(self, url):
        self.url = url
        self.check_type()

    @property
    def log_density_url(self):
        return self.url + "/conditional_log_density"
    
    @property
    def type_url(self):
        return self.url + "/type"
    
    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        # build request
        stimulus_object = as_stimulus(stimulus)

        # TODO: check for file stimuli, in this case use original file to save encoding time
        pil_image = Image.fromarray(stimulus_object.stimulus_data)
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='png')

        def _convert_attribute(attribute):
            if isinstance(attribute, np.ndarray):
                return attribute.tolist()
            if isinstance(attribute, (np.int64, np.int32)):
                return int(attribute)
            if isinstance(attribute, (np.float64, np.float32)):
                return float(attribute)
            return attribute

        json_data = {
            "x_hist": x_hist.tolist(),
            "y_hist": y_hist.tolist(),
            "t_hist": t_hist.tolist(),
            "attributes": {key: _convert_attribute(value) for key, value in (attributes or {}).items()}
        }
        # send request
        response = requests.post(f"{self.log_density_url}", data={'json_data': orjson.dumps(json_data)}, files={'stimulus': image_bytes.getvalue()})

        # parse response
        if response.status_code != 200:
            raise ValueError(f"Server returned status code {response.status_code}")

        json_data = orjson.loads(response.text)
        prediction = np.array(json_data['log_density'])
        return prediction

    def check_type(self):
        response = requests.get(f"{self.type_url}").json()
        if not response['type'] == 'ScanpathModel':
            raise ValueError(f"invalid Model type: {response['type']}. Expected 'ScanpathModel'")
        if not response['version'] in ['v1.0.0']:
            raise ValueError(f"invalid Model type: {response['version']}. Expected 'v1.0.0'")


class HTTPScanpathSaliencyMapModel(ScanpathSaliencyMapModel):
    """
    A scanpath saliency map model that uses a HTTP server to make predictions.
    
    The model is provided with an URL where it expects a server with the following API:
    
    /conditional_saliency_map: expects a POST request with a file attachtment `stimulus` 
    containing the stimulus and a json body containing x_hist, y_hist, t_hist and a dictionary with other attributes
    /type: returns the model type and version
    """
    def __init__(self, url):
        self.url = url
        self.check_type()

    @property
    def saliency_map_url(self):
        return self.url + "/conditional_saliency_map"
    
    @property
    def type_url(self):
        return self.url + "/type"
    
    def conditional_saliency_map(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        # build request
        stimulus_object = as_stimulus(stimulus)

        # TODO: check for file stimuli, in this case use original file to save encoding time
        pil_image = Image.fromarray(stimulus_object.stimulus_data)
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='png')

        def _convert_attribute(attribute):
            if isinstance(attribute, np.ndarray):
                return attribute.tolist()
            if isinstance(attribute, (np.int64, np.int32)):
                return int(attribute)
            if isinstance(attribute, (np.float64, np.float32)):
                return float(attribute)
            return attribute

        json_data = {
            "x_hist": x_hist.tolist(),
            "y_hist": y_hist.tolist(),
            "t_hist": t_hist.tolist(),
            "attributes": {key: _convert_attribute(value) for key, value in (attributes or {}).items()}
        }
        # send request
        response = requests.post(f"{self.saliency_map_url}", data={'json_data': orjson.dumps(json_data)}, files={'stimulus': image_bytes.getvalue()})

        # parse response
        if response.status_code != 200:
            raise ValueError(f"Server returned status code {response.status_code}")

        json_data = orjson.loads(response.text)
        prediction = np.array(json_data['saliency_map'])
        return prediction

    def check_type(self):
        response = requests.get(f"{self.type_url}").json()
        if not response['type'] == 'ScanpathSaliencyMapModel':
            raise ValueError(f"invalid Model type: {response['type']}. Expected 'ScanpathSaliencyMapModel'")
        if not response['version'] in ['v1.0.0']:
            raise ValueError(f"invalid Model type: {response['version']}. Expected 'v1.0.0'")
