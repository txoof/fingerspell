"""
Unit tests for resource path resolution.

Tests path resolution in both development and packaged environments.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.fingerspell.utils.resources import (
    get_base_path,
    get_resource_path,
    get_models_dir,
    verify_resource_exists
)


class TestResourcePaths(unittest.TestCase):
    """Test resource path resolution."""
    
    def test_get_base_path_development(self):
        """Test base path resolution when running from source."""
        base = get_base_path()
        
        # Base path should be a Path object
        self.assertIsInstance(base, Path)
        
        # Base path should be absolute
        self.assertTrue(base.is_absolute())
        
        # Base path should exist
        self.assertTrue(base.exists())
        
        # Base path should contain the project structure
        # (checking for src/ directory as indicator)
        self.assertTrue((base / 'src').exists())
    
    @patch('sys.frozen', True, create=True)
    @patch('sys._MEIPASS', '/tmp/pyinstaller_bundle', create=True)
    def test_get_base_path_frozen(self):
        """Test base path resolution when running as PyInstaller bundle."""
        base = get_base_path()
        
        # Should return the PyInstaller temp directory
        self.assertEqual(base, Path('/tmp/pyinstaller_bundle'))
    
    def test_get_resource_path_returns_string(self):
        """Test that resource path returns string, not Path object."""
        path = get_resource_path('models/test.pkl')
        
        # Must be a string for library compatibility
        self.assertIsInstance(path, str)
    
    def test_get_resource_path_is_absolute(self):
        """Test that resource path is absolute."""
        path = get_resource_path('models/test.pkl')
        
        # Convert back to Path for testing
        path_obj = Path(path)
        self.assertTrue(path_obj.is_absolute())
    
    def test_get_resource_path_includes_relative_path(self):
        """Test that resource path includes the requested relative path."""
        path = get_resource_path('models/test.pkl')
        
        # Should end with the relative path we requested
        self.assertTrue(path.endswith('models/test.pkl') or 
                       path.endswith('models\\test.pkl'))
    
    def test_get_models_dir_returns_string(self):
        """Test that models directory path returns string."""
        models_dir = get_models_dir()
        
        self.assertIsInstance(models_dir, str)
    
    def test_get_models_dir_points_to_models(self):
        """Test that models directory path ends with 'models'."""
        models_dir = get_models_dir()
        
        # Should end with 'models' directory
        self.assertTrue(models_dir.endswith('models'))
    
    def test_verify_resource_exists_true(self):
        """Test resource verification for existing file."""
        # Test with a file we know exists (this test file itself)
        test_file = Path(__file__)
        relative_from_root = test_file.relative_to(get_base_path())
        
        exists = verify_resource_exists(str(relative_from_root))
        self.assertTrue(exists)
    
    def test_verify_resource_exists_false(self):
        """Test resource verification for non-existent file."""
        exists = verify_resource_exists('nonexistent/fake_file.pkl')
        
        self.assertFalse(exists)
    
    def test_verify_resource_exists_with_path_object(self):
        """Test that verify_resource_exists works with Path-like strings."""
        # Should handle forward slashes
        exists = verify_resource_exists('models/fake.pkl')
        self.assertIsInstance(exists, bool)
    
    @patch('src.fingerspell.utils.resources.get_base_path')
    def test_resource_path_uses_base_path(self, mock_base):
        """Test that resource resolution uses get_base_path."""
        mock_base.return_value = Path('/fake/base')
        
        path = get_resource_path('test.txt')
        
        # Should have called get_base_path
        mock_base.assert_called_once()
        
        # Result should include our fake base
        self.assertIn('/fake/base', path)


class TestResourcePathsIntegration(unittest.TestCase):
    """Integration tests using actual project structure."""
    
    def test_can_locate_actual_models(self):
        """Test that we can locate the actual model files."""
        base = get_base_path()
        models_dir = base / 'models'
        
        if models_dir.exists():
            # If models directory exists, check for model files
            static_model = verify_resource_exists('models/ngt_static_classifier.pkl')
            dynamic_model = verify_resource_exists('models/ngt_dynamic_classifier.pkl')
            
            # At least one model should exist if models dir exists
            self.assertTrue(static_model or dynamic_model,
                          "Models directory exists but no model files found")
    
    def test_resource_paths_are_consistent(self):
        """Test that multiple calls return consistent paths."""
        path1 = get_resource_path('models/test.pkl')
        path2 = get_resource_path('models/test.pkl')
        
        self.assertEqual(path1, path2)


class TestResourcePathsEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_empty_relative_path(self):
        """Test resource path with empty string."""
        path = get_resource_path('')
        
        # Should return base path as string
        self.assertEqual(path, str(get_base_path()))
    
    def test_relative_path_with_dots(self):
        """Test resource path with .. in path."""
        path = get_resource_path('../outside/file.txt')
        
        # Should still return a string path
        self.assertIsInstance(path, str)
    
    def test_windows_style_path(self):
        """Test resource path with Windows-style backslashes."""
        path = get_resource_path('models\\test.pkl')
        
        # Should handle backslashes and return string
        self.assertIsInstance(path, str)
    
    def test_absolute_path_as_input(self):
        """Test behavior when given an absolute path."""
        # This should still work, though not recommended usage
        path = get_resource_path('/absolute/path/test.pkl')
        
        self.assertIsInstance(path, str)


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)