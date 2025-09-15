"""
Tests for configuration system.
"""
import os
import tempfile
import yaml
from pathlib import Path
from src.config import ConfigManager, Config


def test_default_config():
    """Test default configuration creation."""
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    assert isinstance(config, Config)
    assert config.app.data_dir == "./storage"
    assert config.app.project_cache_size == 3
    assert config.chunk.target_tokens == 500
    assert config.chunk.max_tokens == 900
    assert config.retrieve.top_k == 5


def test_config_from_yaml():
    """Test configuration loading from YAML file."""
    test_config = {
        "app": {
            "data_dir": "./test_storage",
            "project_cache_size": 5
        },
        "chunk": {
            "target_tokens": 300,
            "max_tokens": 600
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigManager(temp_path)
        config = config_manager.load_config()
        
        assert config.app.data_dir == "./test_storage"
        assert config.app.project_cache_size == 5
        assert config.chunk.target_tokens == 300
        assert config.chunk.max_tokens == 600
        # Default values should still be present
        assert config.retrieve.top_k == 5
    finally:
        os.unlink(temp_path)


def test_config_validation():
    """Test configuration validation."""
    test_config = {
        "chunk": {
            "target_tokens": 1000,
            "max_tokens": 500  # Invalid: max < target
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigManager(temp_path)
        try:
            config_manager.load_config()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max_tokens must be >= target_tokens" in str(e)
    finally:
        os.unlink(temp_path)


def test_env_vars():
    """Test environment variable handling."""
    # Set test environment variable
    os.environ["OPENAI_API_KEY"] = "test_key_123"
    
    try:
        config_manager = ConfigManager()
        api_key = config_manager.get_env_var("OPENAI_API_KEY")
        assert api_key == "test_key_123"
    finally:
        # Clean up
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":
    test_default_config()
    test_config_from_yaml()
    test_config_validation()
    test_env_vars()
    print("All configuration tests passed!")