"""
Centralized logging configuration for the Construction RAG System.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LoggingConfig:
    """Centralized logging configuration manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize logging configuration.
        
        Args:
            config: Optional configuration dictionary with logging settings
        """
        self.config = config or {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration with file and console handlers."""
        # Get logging configuration
        log_level = self.config.get('log_level', 'INFO').upper()
        log_dir = Path(self.config.get('log_dir', 'logs'))
        max_file_size = self.config.get('max_file_size_mb', 10) * 1024 * 1024
        backup_count = self.config.get('backup_count', 5)
        console_logging = self.config.get('console_logging', True)
        
        # Create logs directory
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        log_file = log_dir / 'construction_rag.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler (optional)
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Error file handler for errors and critical messages
        error_file = log_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log handler
        perf_file = log_dir / 'performance.log'
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for the given name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    @staticmethod
    def get_performance_logger() -> logging.Logger:
        """Get the performance logger instance.
        
        Returns:
            Performance logger instance
        """
        return logging.getLogger('performance')


def setup_logging(config: Optional[Dict[str, Any]] = None) -> LoggingConfig:
    """Set up logging configuration for the application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LoggingConfig instance
    """
    return LoggingConfig(config)


def log_performance(operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
    """Log performance metrics for operations.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        details: Optional additional details
    """
    perf_logger = LoggingConfig.get_performance_logger()
    details_str = f" - {details}" if details else ""
    perf_logger.info(f"{operation}: {duration:.2f}s{details_str}")


def log_memory_usage(operation: str, memory_mb: float):
    """Log memory usage for operations.
    
    Args:
        operation: Name of the operation
        memory_mb: Memory usage in MB
    """
    perf_logger = LoggingConfig.get_performance_logger()
    perf_logger.info(f"Memory usage - {operation}: {memory_mb:.1f}MB")