import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import json

import pysaliency
from pysaliency.http_models import HTTPScanpathModel, HTTPScanpathSaliencyMapModel


class TestHTTPScanpathModel(unittest.TestCase):
    """Test the existing HTTPScanpathModel"""
    
    def test_init(self):
        """Test that HTTPScanpathModel initializes correctly"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathModel('http://example.com')
            self.assertEqual(model.url, 'http://example.com')
            self.assertEqual(model.log_density_url, 'http://example.com/conditional_log_density')
            self.assertEqual(model.type_url, 'http://example.com/type')
            
            mock_get.assert_called_once_with('http://example.com/type')
    
    def test_init_invalid_type(self):
        """Test that HTTPScanpathModel raises error for invalid type"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'WrongType', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            with self.assertRaises(ValueError):
                HTTPScanpathModel('http://example.com')
    
    def test_conditional_log_density(self):
        """Test conditional_log_density method"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathModel('http://example.com')
        
        # Mock the POST request for log density
        with patch('pysaliency.http_models.requests.post') as mock_post:
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            expected_log_density = [-2.1, -1.8, -2.5, -1.2]
            mock_post_response.text = json.dumps({'log_density': expected_log_density})
            mock_post.return_value = mock_post_response
            
            # Create test stimulus
            stimulus = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
            x_hist = np.array([1, 2, 3, 4])
            y_hist = np.array([4, 5, 6, 7])
            t_hist = np.array([0.1, 0.2, 0.3, 0.4])
            
            result = model.conditional_log_density(stimulus, x_hist, y_hist, t_hist)
            
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected_log_density)
            
            # Verify the POST request was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertEqual(call_args[0][0], 'http://example.com/conditional_log_density')
            self.assertIn('data', call_args[1])
            self.assertIn('files', call_args[1])


class TestHTTPScanpathSaliencyMapModel(unittest.TestCase):
    """Test the new HTTPScanpathSaliencyMapModel"""
    
    def test_init(self):
        """Test that HTTPScanpathSaliencyMapModel initializes correctly"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathSaliencyMapModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathSaliencyMapModel('http://example.com')
            self.assertEqual(model.url, 'http://example.com')
            self.assertEqual(model.saliency_map_url, 'http://example.com/conditional_saliency_map')
            self.assertEqual(model.type_url, 'http://example.com/type')
            
            mock_get.assert_called_once_with('http://example.com/type')
    
    def test_init_invalid_type(self):
        """Test that HTTPScanpathSaliencyMapModel raises error for invalid type"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            with self.assertRaises(ValueError):
                HTTPScanpathSaliencyMapModel('http://example.com')
    
    def test_init_invalid_version(self):
        """Test that HTTPScanpathSaliencyMapModel raises error for invalid version"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathSaliencyMapModel', 'version': 'v2.0.0'}
            mock_get.return_value = mock_response
            
            with self.assertRaises(ValueError):
                HTTPScanpathSaliencyMapModel('http://example.com')
    
    def test_conditional_saliency_map(self):
        """Test conditional_saliency_map method"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathSaliencyMapModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathSaliencyMapModel('http://example.com')
        
        # Mock the POST request for saliency map
        with patch('pysaliency.http_models.requests.post') as mock_post:
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            expected_saliency_map = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_post_response.text = json.dumps({'saliency_map': expected_saliency_map})
            mock_post.return_value = mock_post_response
            
            # Create test stimulus
            stimulus = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
            x_hist = np.array([1, 2, 3])
            y_hist = np.array([4, 5, 6])
            t_hist = np.array([0.1, 0.2, 0.3])
            
            result = model.conditional_saliency_map(stimulus, x_hist, y_hist, t_hist)
            
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected_saliency_map)
            
            # Verify the POST request was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertEqual(call_args[0][0], 'http://example.com/conditional_saliency_map')
            self.assertIn('data', call_args[1])
            self.assertIn('files', call_args[1])
    
    def test_conditional_saliency_map_with_attributes(self):
        """Test conditional_saliency_map method with attributes"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathSaliencyMapModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathSaliencyMapModel('http://example.com')
        
        # Mock the POST request for saliency map
        with patch('pysaliency.http_models.requests.post') as mock_post:
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            expected_saliency_map = [[0.1, 0.2], [0.3, 0.4]]
            mock_post_response.text = json.dumps({'saliency_map': expected_saliency_map})
            mock_post.return_value = mock_post_response
            
            # Create test stimulus
            stimulus = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
            x_hist = np.array([1, 2])
            y_hist = np.array([4, 5])
            t_hist = np.array([0.1, 0.2])
            attributes = {'subject': 1, 'task': 'search'}
            
            result = model.conditional_saliency_map(stimulus, x_hist, y_hist, t_hist, attributes=attributes)
            
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected_saliency_map)
    
    def test_conditional_saliency_map_server_error(self):
        """Test conditional_saliency_map method handles server errors"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathSaliencyMapModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathSaliencyMapModel('http://example.com')
        
        # Mock the POST request to return error
        with patch('pysaliency.http_models.requests.post') as mock_post:
            mock_post_response = MagicMock()
            mock_post_response.status_code = 500
            mock_post.return_value = mock_post_response
            
            # Create test stimulus
            stimulus = np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)
            x_hist = np.array([1, 2])
            y_hist = np.array([4, 5])
            t_hist = np.array([0.1, 0.2])
            
            with self.assertRaises(ValueError):
                model.conditional_saliency_map(stimulus, x_hist, y_hist, t_hist)
    
    def test_inheritance(self):
        """Test that HTTPScanpathSaliencyMapModel inherits from ScanpathSaliencyMapModel"""
        with patch('pysaliency.http_models.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'type': 'ScanpathSaliencyMapModel', 'version': 'v1.0.0'}
            mock_get.return_value = mock_response
            
            model = HTTPScanpathSaliencyMapModel('http://example.com')
            self.assertIsInstance(model, pysaliency.ScanpathSaliencyMapModel)
            self.assertTrue(hasattr(model, 'conditional_saliency_map'))


if __name__ == '__main__':
    unittest.main()