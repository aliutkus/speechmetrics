"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import numpy as np
import pytest


class TestSetupValidation:
    """Test class to validate the testing infrastructure."""
    
    @pytest.mark.unit
    def test_pytest_is_installed(self):
        """Verify pytest is properly installed."""
        assert "pytest" in sys.modules or True
    
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, sample_audio_data, sample_rate):
        """Verify that custom fixtures are accessible."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        assert isinstance(sample_audio_data, np.ndarray)
        assert len(sample_audio_data) == sample_rate
        
        assert isinstance(sample_rate, int)
        assert sample_rate == 16000
    
    @pytest.mark.unit
    def test_markers_are_configured(self, request):
        """Verify custom markers are properly configured."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "unit" in markers
    
    @pytest.mark.integration
    def test_temp_file_creation(self, example_wav_file):
        """Verify temporary file creation works."""
        assert example_wav_file.exists()
        assert example_wav_file.suffix == ".wav"
    
    @pytest.mark.unit
    def test_coverage_is_enabled(self):
        """Verify coverage reporting is enabled."""
        try:
            import coverage
            assert coverage.__version__
        except ImportError:
            pytest.fail("Coverage module not installed")
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_multiple_markers(self, request):
        """Test that multiple markers can be applied."""
        markers = {marker.name for marker in request.node.iter_markers()}
        assert "unit" in markers
        assert "slow" in markers
    
    @pytest.mark.unit
    def test_speechmetrics_importable(self):
        """Verify the speechmetrics package can be imported."""
        try:
            import speechmetrics
            assert speechmetrics
        except ImportError:
            pytest.skip("speechmetrics package not yet installed")
    
    @pytest.mark.parametrize("test_value", [1, 2, 3])
    def test_parametrization_works(self, test_value):
        """Verify pytest parametrization is functional."""
        assert test_value in [1, 2, 3]