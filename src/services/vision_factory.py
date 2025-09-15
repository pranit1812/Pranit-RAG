"""
Factory for creating vision assistant instances from configuration.
"""
from pathlib import Path
from typing import Optional

from config import Config, get_env_var
from models.types import VisionConfig as VisionConfigType
from services.vision import VisionAssistant, create_vision_assistant


def create_vision_assistant_from_config(
    config: Config,
    project_storage_dir: Optional[Path] = None
) -> VisionAssistant:
    """
    Create a vision assistant from system configuration.
    
    Args:
        config: System configuration
        project_storage_dir: Project storage directory for image caching
        
    Returns:
        Configured VisionAssistant instance
    """
    # Convert config to VisionConfig type
    vision_config: VisionConfigType = {
        "enabled": config.vision.enabled,
        "max_images": config.vision.max_images,
        "resolution_scale": config.vision.resolution_scale
    }
    
    # Set up cache directory
    cache_dir = None
    if project_storage_dir:
        cache_dir = project_storage_dir / "pages"
    
    # Get API key from environment
    api_key = get_env_var("OPENAI_API_KEY")
    
    return create_vision_assistant(
        cache_dir=cache_dir,
        api_key=api_key,
        config=vision_config
    )


def get_vision_config_from_system_config(config: Config) -> VisionConfigType:
    """
    Extract vision configuration as VisionConfig type.
    
    Args:
        config: System configuration
        
    Returns:
        VisionConfig dictionary
    """
    return {
        "enabled": config.vision.enabled,
        "max_images": config.vision.max_images,
        "resolution_scale": config.vision.resolution_scale
    }