"""
Configuration management for the Construction RAG System.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try to load .env manually
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value


@dataclass
class AppConfig:
    """Application configuration."""
    data_dir: str = "./storage"
    project_cache_size: int = 3
    max_upload_mb: int = 100000


@dataclass
class LLMConfig:
    """LLM service configuration."""
    chat_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-large"
    vision_assist: bool = False


@dataclass
class EmbeddingsConfig:
    """Embeddings configuration."""
    provider: str = "openai"  # openai | local
    local_model: str = "all-MiniLM-L12-v2"
    batch_size: int = 64


@dataclass
class OCRConfig:
    """OCR configuration."""
    engine: str = "paddleocr"
    ppstructure_model: str = "TableMaster"
    min_conf: float = 0.5


@dataclass
class ExtractConfig:
    """Extraction configuration."""
    pipeline_priority: List[str] = field(default_factory=lambda: [
        "native_pdf",        # Works best, fastest - put first!
        "docling",           # Good but slow
        "unstructured_hi_res", 
        "ocr_ppstructure"
    ])
    languages: List[str] = field(default_factory=lambda: ["en"])
    ocr: OCRConfig = field(default_factory=OCRConfig)
    timeout_seconds: int = 300  # 5 minutes for complex documents
    max_memory_mb: int = 8192   # 8GB for large documents


@dataclass
class PreserveConfig:
    """Preserve configuration for chunking."""
    tables: bool = True
    lists: bool = True


@dataclass
class DrawingConfig:
    """Drawing-specific chunking configuration."""
    cluster_text: bool = True
    max_regions: int = 8


@dataclass
class ChunkConfig:
    """Chunking configuration."""
    target_tokens: int = 500
    max_tokens: int = 900
    preserve: PreserveConfig = field(default_factory=PreserveConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)


@dataclass
class RetrieveConfig:
    """Retrieval configuration."""
    top_k: int = 5
    hybrid: bool = True
    reranker: str = "none"  # none | cross_encoder
    sliding_window: bool = False
    window_size: int = 1


@dataclass
class VisionConfig:
    """Vision assistance configuration."""
    enabled: bool = False
    max_images: int = 3
    resolution_scale: float = 2.0


@dataclass
class Config:
    """Main configuration class."""
    app: AppConfig = field(default_factory=AppConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


class ConfigManager:
    """Configuration manager with YAML loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Optional[Config] = None
        self._env_vars = self._load_env_vars()
    
    def _load_env_vars(self) -> Dict[str, str]:
        """Load environment variables for API keys."""
        return {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        }
    
    def load_config(self) -> Config:
        """Load configuration from YAML file with validation."""
        if self._config is not None:
            return self._config
        
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f) or {}
                self._config = self._create_config_from_dict(yaml_data)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}. Using default configuration.")
                self._config = Config()
        else:
            logger.info(f"Config file {config_file} not found. Using default configuration.")
            self._config = Config()
        
        # Validate configuration
        self._validate_config(self._config)
        return self._config
    
    def _create_config_from_dict(self, data: Dict[str, Any]) -> Config:
        """Create Config object from dictionary data."""
        config = Config()
        
        # App config
        if "app" in data:
            app_data = data["app"]
            config.app = AppConfig(
                data_dir=app_data.get("data_dir", config.app.data_dir),
                project_cache_size=app_data.get("project_cache_size", config.app.project_cache_size),
                max_upload_mb=app_data.get("max_upload_mb", config.app.max_upload_mb)
            )
        
        # LLM config
        if "llm" in data:
            llm_data = data["llm"]
            config.llm = LLMConfig(
                chat_model=llm_data.get("chat_model", config.llm.chat_model),
                embed_model=llm_data.get("embed_model", config.llm.embed_model),
                vision_assist=llm_data.get("vision_assist", config.llm.vision_assist)
            )
        
        # Embeddings config
        if "embeddings" in data:
            emb_data = data["embeddings"]
            config.embeddings = EmbeddingsConfig(
                provider=emb_data.get("provider", config.embeddings.provider),
                local_model=emb_data.get("local_model", config.embeddings.local_model),
                batch_size=emb_data.get("batch_size", config.embeddings.batch_size)
            )
        
        # Extract config
        if "extract" in data:
            ext_data = data["extract"]
            ocr_data = ext_data.get("ocr", {})
            config.extract = ExtractConfig(
                pipeline_priority=ext_data.get("pipeline_priority", config.extract.pipeline_priority),
                languages=ext_data.get("languages", config.extract.languages),
                ocr=OCRConfig(
                    engine=ocr_data.get("engine", config.extract.ocr.engine),
                    ppstructure_model=ocr_data.get("ppstructure_model", config.extract.ocr.ppstructure_model),
                    min_conf=ocr_data.get("min_conf", config.extract.ocr.min_conf)
                ),
                timeout_seconds=ext_data.get("timeout_seconds", config.extract.timeout_seconds),
                max_memory_mb=ext_data.get("max_memory_mb", config.extract.max_memory_mb)
            )
        
        # Chunk config
        if "chunk" in data:
            chunk_data = data["chunk"]
            preserve_data = chunk_data.get("preserve", {})
            drawing_data = chunk_data.get("drawing", {})
            config.chunk = ChunkConfig(
                target_tokens=chunk_data.get("target_tokens", config.chunk.target_tokens),
                max_tokens=chunk_data.get("max_tokens", config.chunk.max_tokens),
                preserve=PreserveConfig(
                    tables=preserve_data.get("tables", config.chunk.preserve.tables),
                    lists=preserve_data.get("lists", config.chunk.preserve.lists)
                ),
                drawing=DrawingConfig(
                    cluster_text=drawing_data.get("cluster_text", config.chunk.drawing.cluster_text),
                    max_regions=drawing_data.get("max_regions", config.chunk.drawing.max_regions)
                )
            )
        
        # Retrieve config
        if "retrieve" in data:
            ret_data = data["retrieve"]
            config.retrieve = RetrieveConfig(
                top_k=ret_data.get("top_k", config.retrieve.top_k),
                hybrid=ret_data.get("hybrid", config.retrieve.hybrid),
                reranker=ret_data.get("reranker", config.retrieve.reranker),
                sliding_window=ret_data.get("sliding_window", config.retrieve.sliding_window),
                window_size=ret_data.get("window_size", config.retrieve.window_size)
            )
        
        # Vision config
        if "vision" in data:
            vis_data = data["vision"]
            config.vision = VisionConfig(
                enabled=vis_data.get("enabled", config.vision.enabled),
                max_images=vis_data.get("max_images", config.vision.max_images),
                resolution_scale=vis_data.get("resolution_scale", config.vision.resolution_scale)
            )
        
        return config
    
    def _validate_config(self, config: Config) -> None:
        """Validate configuration values."""
        # Validate app config
        if config.app.project_cache_size < 1:
            raise ValueError("project_cache_size must be at least 1")
        if config.app.max_upload_mb < 1:
            raise ValueError("max_upload_mb must be at least 1")
        
        # Validate embeddings provider
        if config.embeddings.provider not in ["openai", "local"]:
            raise ValueError("embeddings.provider must be 'openai' or 'local'")
        
        # Validate extraction pipeline
        valid_extractors = ["docling", "unstructured_hi_res", "native_pdf", "ocr_ppstructure"]
        for extractor in config.extract.pipeline_priority:
            if extractor not in valid_extractors:
                raise ValueError(f"Invalid extractor '{extractor}' in pipeline_priority")
        if config.extract.timeout_seconds < 15:
            raise ValueError("extract.timeout_seconds must be at least 15 seconds")
        if config.extract.max_memory_mb < 512:
            raise ValueError("extract.max_memory_mb must be at least 512 MB")
        
        # Validate chunk config
        if config.chunk.target_tokens < 1:
            raise ValueError("chunk.target_tokens must be at least 1")
        if config.chunk.max_tokens < config.chunk.target_tokens:
            raise ValueError("chunk.max_tokens must be >= target_tokens")
        
        # Validate retrieve config
        if config.retrieve.top_k < 1:
            raise ValueError("retrieve.top_k must be at least 1")
        if config.retrieve.reranker not in ["none", "cross_encoder"]:
            raise ValueError("retrieve.reranker must be 'none' or 'cross_encoder'")
        
        # Validate vision config
        if config.vision.max_images < 1 or config.vision.max_images > 5:
            raise ValueError("vision.max_images must be between 1 and 5")
        if config.vision.resolution_scale <= 0:
            raise ValueError("vision.resolution_scale must be positive")
        
        # Validate API keys if using OpenAI
        if config.embeddings.provider == "openai" and not self._env_vars.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment variables")
    
    def get_env_var(self, key: str) -> str:
        """Get environment variable value."""
        return self._env_vars.get(key, "")
    
    def create_default_config_file(self) -> None:
        """Create a default config.yaml file."""
        default_config = {
            "app": {
                "data_dir": "./storage",
                "project_cache_size": 3,
                "max_upload_mb": 100000
            },
            "llm": {
                "chat_model": "gpt-4o",
                "embed_model": "text-embedding-3-large",
                "vision_assist": False
            },
            "embeddings": {
                "provider": "openai",
                "local_model": "all-MiniLM-L12-v2",
                "batch_size": 64
            },
            "extract": {
                "pipeline_priority": [
                    "native_pdf",        # Works best, fastest - put first!
                    "docling",           # Good but slow
                    "unstructured_hi_res", 
                    "ocr_ppstructure"
                ],
                "languages": ["en"],
                "timeout_seconds": 180,
                "max_memory_mb": 4096,
                "ocr": {
                    "engine": "paddleocr",
                    "ppstructure_model": "TableMaster",
                    "min_conf": 0.5
                }
            },
            "chunk": {
                "target_tokens": 500,
                "max_tokens": 900,
                "preserve": {
                    "tables": True,
                    "lists": True
                },
                "drawing": {
                    "cluster_text": True,
                    "max_regions": 8
                }
            },
            "retrieve": {
                "top_k": 5,
                "hybrid": True,
                "reranker": "none",
                "sliding_window": False,
                "window_size": 1
            },
            "vision": {
                "enabled": False,
                "max_images": 3,
                "resolution_scale": 2.0
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created default configuration file: {self.config_path}")


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.load_config()


def get_env_var(key: str) -> str:
    """Get environment variable value."""
    return config_manager.get_env_var(key)