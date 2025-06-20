"""Shared pytest fixtures and configuration for speechmetrics tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    return audio


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for test audio."""
    return 16000


@pytest.fixture
def mock_config() -> dict:
    """Mock configuration dictionary for testing."""
    return {
        "model_path": "/path/to/model",
        "batch_size": 32,
        "num_workers": 4,
        "device": "cpu",
    }


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def example_wav_file(temp_dir: Path, sample_audio_data: np.ndarray) -> Path:
    """Create a temporary WAV file for testing."""
    import scipy.io.wavfile as wavfile
    
    wav_path = temp_dir / "test_audio.wav"
    wavfile.write(str(wav_path), 16000, (sample_audio_data * 32767).astype(np.int16))
    
    return wav_path


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_tensorflow(monkeypatch):
    """Mock TensorFlow imports for faster testing."""
    import sys
    from unittest.mock import MagicMock
    
    mock_tf = MagicMock()
    monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)
    
    return mock_tf


@pytest.fixture
def noisy_audio_pair(sample_audio_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean and noisy audio pair for testing."""
    clean = sample_audio_data
    noise = np.random.normal(0, 0.1, size=clean.shape)
    noisy = clean + noise
    
    return clean, noisy


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring a pre-trained model"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU support"
    )