"""
Comprehensive error handling utilities for the Construction RAG System.
"""
import logging
import traceback
import functools
import time
import psutil
import os
from typing import Any, Callable, Optional, Dict, Type, Union, List
from pathlib import Path
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class ConstructionRAGError(Exception):
    """Base exception for Construction RAG System."""
    pass


class ExtractionError(ConstructionRAGError):
    """Exception raised during document extraction."""
    pass


class ChunkingError(ConstructionRAGError):
    """Exception raised during document chunking."""
    pass


class EmbeddingError(ConstructionRAGError):
    """Exception raised during embedding generation."""
    pass


class RetrievalError(ConstructionRAGError):
    """Exception raised during retrieval operations."""
    pass


class VisionError(ConstructionRAGError):
    """Exception raised during vision processing."""
    pass


class ProjectError(ConstructionRAGError):
    """Exception raised during project operations."""
    pass


class APIError(ConstructionRAGError):
    """Exception raised during API calls."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def handle_extraction_errors(
    fallback_providers: Optional[List[str]] = None,
    continue_on_error: bool = True
):
    """Decorator for handling extraction errors with provider fallback.
    
    Args:
        fallback_providers: List of fallback provider names
        continue_on_error: Whether to continue processing other files on error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Extraction failed in {func.__name__}: {e}")
                logger.debug(f"Extraction error traceback: {traceback.format_exc()}")
                
                if not continue_on_error:
                    raise ExtractionError(f"Extraction failed: {e}") from e
                
                # Return empty result to continue processing
                return []
        
        return wrapper
    return decorator


def safe_api_call(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    timeout: Optional[float] = None
):
    """Decorator for safe API calls with retry logic and timeout.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier for delay
        timeout: Optional timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if timeout:
                        # Simple timeout implementation
                        import signal
                        import platform
                        
                        # Only use signal timeout on Unix systems
                        if platform.system() != 'Windows':
                            def timeout_handler(signum, frame):
                                raise TimeoutError(f"API call timed out after {timeout}s")
                            
                            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(int(timeout))
                            
                            try:
                                result = func(*args, **kwargs)
                            finally:
                                signal.alarm(0)
                                signal.signal(signal.SIGALRM, old_handler)
                            
                            return result
                        else:
                            # On Windows, just run without timeout
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except (TimeoutError, ConnectionError, APIError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"API call {func.__name__} failed after {max_retries} retries: {e}")
                        raise APIError(f"API call failed: {e}") from e
                    
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(f"API call {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                
                except Exception as e:
                    logger.error(f"Unexpected error in API call {func.__name__}: {e}")
                    raise APIError(f"Unexpected API error: {e}") from e
            
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def memory_monitor(operation_name: str, max_memory_mb: Optional[float] = None):
    """Context manager for monitoring memory usage during operations.
    
    Args:
        operation_name: Name of the operation being monitored
        max_memory_mb: Optional maximum memory limit in MB
    """
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.debug(f"Starting {operation_name} - Initial memory: {start_memory:.1f}MB")
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - start_memory
        
        logger.info(f"Completed {operation_name} - Memory used: {memory_used:.1f}MB, Final: {end_memory:.1f}MB")
        
        if max_memory_mb and end_memory > max_memory_mb:
            logger.warning(f"Memory usage ({end_memory:.1f}MB) exceeded limit ({max_memory_mb}MB) for {operation_name}")


@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    start_time = time.time()
    logger.debug(f"Starting {operation_name}")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} in {duration:.2f}s")
        
        # Log to performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.info(f"{operation_name}: {duration:.2f}s")


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    operation: str,
    file_path: Optional[Union[str, Path]] = None
):
    """Log error with detailed context information.
    
    Args:
        error: The exception that occurred
        context: Dictionary of context information
        operation: Name of the operation that failed
        file_path: Optional file path related to the error
    """
    error_msg = f"Error in {operation}: {error}"
    
    if file_path:
        error_msg += f" (File: {file_path})"
    
    logger.error(error_msg)
    logger.debug(f"Error context: {context}")
    logger.debug(f"Error traceback: {traceback.format_exc()}")


def create_user_friendly_error(error: Exception, operation: str) -> str:
    """Create user-friendly error message from exception.
    
    Args:
        error: The exception that occurred
        operation: Name of the operation that failed
        
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    
    # Map technical errors to user-friendly messages
    error_messages = {
        'FileNotFoundError': f"File not found during {operation}. Please check the file path and try again.",
        'PermissionError': f"Permission denied during {operation}. Please check file permissions.",
        'MemoryError': f"Out of memory during {operation}. Try processing smaller files or restart the application.",
        'TimeoutError': f"Operation timed out during {operation}. Please try again or check your connection.",
        'ConnectionError': f"Connection failed during {operation}. Please check your internet connection.",
        'APIError': f"API service error during {operation}. Please try again later.",
        'ExtractionError': f"Document extraction failed during {operation}. The file may be corrupted or in an unsupported format.",
        'ChunkingError': f"Document processing failed during {operation}. Please try with a different file.",
        'EmbeddingError': f"Text embedding failed during {operation}. Please check your API configuration.",
        'RetrievalError': f"Search failed during {operation}. Please try a different query.",
        'VisionError': f"Image analysis failed during {operation}. Please try again or disable vision assist.",
        'ProjectError': f"Project operation failed during {operation}. Please check project settings."
    }
    
    user_message = error_messages.get(error_type, f"An error occurred during {operation}: {str(error)}")
    
    # Add troubleshooting suggestions
    troubleshooting = {
        'FileNotFoundError': "• Verify the file exists and the path is correct\n• Check file permissions",
        'MemoryError': "• Try processing smaller files\n• Restart the application\n• Close other applications to free memory",
        'ConnectionError': "• Check your internet connection\n• Verify API keys are correct\n• Try again in a few minutes",
        'APIError': "• Check your API key configuration\n• Verify you have sufficient API credits\n• Try again later"
    }
    
    if error_type in troubleshooting:
        user_message += f"\n\nTroubleshooting suggestions:\n{troubleshooting[error_type]}"
    
    return user_message


def cleanup_temp_files(temp_paths: List[Union[str, Path]]):
    """Clean up temporary files with error handling.
    
    Args:
        temp_paths: List of temporary file/directory paths to clean up
    """
    for temp_path in temp_paths:
        try:
            path_obj = Path(temp_path)
            if path_obj.exists():
                if path_obj.is_file():
                    path_obj.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                elif path_obj.is_dir():
                    import shutil
                    shutil.rmtree(path_obj)
                    logger.debug(f"Cleaned up temp directory: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp path {temp_path}: {e}")


def validate_file_access(file_path: Union[str, Path]) -> bool:
    """Validate that a file can be accessed for reading.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        True if file can be accessed, False otherwise
    """
    try:
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        if not path_obj.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        
        # Try to open the file
        with open(path_obj, 'rb') as f:
            f.read(1)  # Read one byte to test access
        
        return True
        
    except Exception as e:
        logger.error(f"Cannot access file {file_path}: {e}")
        return False


def handle_graceful_shutdown(cleanup_functions: List[Callable] = None):
    """Handle graceful shutdown with cleanup.
    
    Args:
        cleanup_functions: List of cleanup functions to call
    """
    logger.info("Initiating graceful shutdown...")
    
    if cleanup_functions:
        for cleanup_func in cleanup_functions:
            try:
                cleanup_func()
                logger.debug(f"Cleanup function {cleanup_func.__name__} completed")
            except Exception as e:
                logger.error(f"Error in cleanup function {cleanup_func.__name__}: {e}")
    
    logger.info("Graceful shutdown completed")