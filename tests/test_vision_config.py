"""
Tests for vision configuration integration.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.config import Config, VisionConfig, ConfigManager
from src.services.vision_factory import (
    create_vision_assistant_from_config,
    get_vision_config_from_system_config
)


class TestVisionConfig:
    """Test vision configuration."""
    
    def test_default_vision_config(self):
        """Test default vision configuration values."""
        config = VisionConfig()
        
        assert config.enabled is False
        assert config.max_images == 3
        assert config.resolution_scale == 2.0
    
    def test_vision_config_validation(self):
        """Test vision configuration validation."""
        config_manager = ConfigManager()
        
        # Test valid config
        config = Config()
        config.vision.max_images = 3
        config.vision.resolution_scale = 2.0
        
        # Should not raise
        config_manager._validate_config(config)
        
        # Test invalid max_images
        config.vision.max_images = 0
        with pytest.raises(ValueError, match="max_images must be between 1 and 5"):
            config_manager._validate_config(config)
        
        config.vision.max_images = 6
        with pytest.raises(ValueError, match="max_images must be between 1 and 5"):
            config_manager._validate_config(config)
        
        # Test invalid resolution_scale
        config.vision.max_images = 3
        config.vision.resolution_scale = 0
        with pytest.raises(ValueError, match="resolution_scale must be positive"):
            config_manager._validate_config(config)
    
    def test_vision_config_from_yaml(self):
        """Test loading vision config from YAML."""
        yaml_data = {
            "vision": {
                "enabled": True,
                "max_images": 5,
                "resolution_scale": 1.5
            }
        }
        
        config_manager = ConfigManager()
        config = config_manager._create_config_from_dict(yaml_data)
        
        assert config.vision.enabled is True
        assert config.vision.max_images == 5
        assert config.vision.resolution_scale == 1.5
    
    def test_vision_config_partial_yaml(self):
        """Test loading partial vision config from YAML."""
        yaml_data = {
            "vision": {
                "enabled": True
                # max_images and resolution_scale should use defaults
            }
        }
        
        config_manager = ConfigManager()
        config = config_manager._create_config_from_dict(yaml_data)
        
        assert config.vision.enabled is True
        assert config.vision.max_images == 3  # default
        assert config.vision.resolution_scale == 2.0  # default


class TestVisionFactory:
    """Test vision assistant factory functions."""
    
    def test_get_vision_config_from_system_config(self):
        """Test extracting vision config from system config."""
        config = Config()
        config.vision.enabled = True
        config.vision.max_images = 4
        config.vision.resolution_scale = 1.8
        
        vision_config = get_vision_config_from_system_config(config)
        
        assert vision_config["enabled"] is True
        assert vision_config["max_images"] == 4
        assert vision_config["resolution_scale"] == 1.8
    
    @patch('src.services.vision_factory.create_vision_assistant')
    @patch('src.services.vision_factory.get_env_var')
    def test_create_vision_assistant_from_config(self, mock_get_env, mock_create):
        """Test creating vision assistant from config."""
        mock_get_env.return_value = "test_api_key"
        
        config = Config()
        config.vision.enabled = True
        config.vision.max_images = 2
        config.vision.resolution_scale = 1.5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            
            create_vision_assistant_from_config(config, project_dir)
            
            # Verify the factory was called with correct parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            
            assert call_args[1]["cache_dir"] == project_dir / "pages"
            assert call_args[1]["api_key"] == "test_api_key"
            
            vision_config = call_args[1]["config"]
            assert vision_config["enabled"] is True
            assert vision_config["max_images"] == 2
            assert vision_config["resolution_scale"] == 1.5
    
    @patch('src.services.vision_factory.create_vision_assistant')
    @patch('src.services.vision_factory.get_env_var')
    def test_create_vision_assistant_no_project_dir(self, mock_get_env, mock_create):
        """Test creating vision assistant without project directory."""
        mock_get_env.return_value = "test_api_key"
        
        config = Config()
        
        create_vision_assistant_from_config(config)
        
        # Verify cache_dir is None when no project directory provided
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["cache_dir"] is None


class TestConfigIntegration:
    """Test full configuration integration."""
    
    def test_config_file_with_vision(self):
        """Test loading complete config file with vision settings."""
        config_data = """
app:
  data_dir: ./storage
  project_cache_size: 3
  max_upload_mb: 100000

llm:
  chat_model: gpt-4o
  embed_model: text-embedding-3-large
  vision_assist: false

vision:
  enabled: true
  max_images: 4
  resolution_scale: 1.8

embeddings:
  provider: openai
  local_model: all-MiniLM-L12-v2
  batch_size: 64

extract:
  pipeline_priority:
    - docling
    - unstructured_hi_res
    - native_pdf
    - ocr_ppstructure
  languages:
    - en
  ocr:
    engine: paddleocr
    ppstructure_model: TableMaster
    min_conf: 0.5

chunk:
  target_tokens: 500
  max_tokens: 900
  preserve:
    tables: true
    lists: true
  drawing:
    cluster_text: true
    max_regions: 8

retrieve:
  top_k: 5
  hybrid: true
  reranker: none
  sliding_window: false
  window_size: 1
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()
            
            # Verify vision config was loaded correctly
            assert config.vision.enabled is True
            assert config.vision.max_images == 4
            assert config.vision.resolution_scale == 1.8
            
            # Verify other configs still work
            assert config.app.data_dir == "./storage"
            assert config.llm.chat_model == "gpt-4o"
            assert config.retrieve.top_k == 5
            
        finally:
            Path(config_path).unlink()
    
    def test_default_config_creation_with_vision(self):
        """Test that default config creation includes vision settings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        # Delete the file so create_default_config_file can create it
        Path(config_path).unlink()
        
        try:
            config_manager = ConfigManager(config_path)
            config_manager.create_default_config_file()
            
            # Load the created config
            config = config_manager.load_config()
            
            # Verify vision config is present with defaults
            assert config.vision.enabled is False
            assert config.vision.max_images == 3
            assert config.vision.resolution_scale == 2.0
            
        finally:
            if Path(config_path).exists():
                Path(config_path).unlink()