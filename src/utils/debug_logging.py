"""
Debug logging utilities for troubleshooting extraction and chunking issues.
"""
import logging
import json
import traceback
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import inspect

try:
    from models.types import PageParse, Block, Chunk, ChunkMetadata
except ImportError:
    # Handle case when running as standalone script
    from models.types import PageParse, Block, Chunk, ChunkMetadata


logger = logging.getLogger(__name__)
debug_logger = logging.getLogger('debug')


class DebugLogger:
    """Debug logger for detailed troubleshooting information."""
    
    def __init__(self, enabled: bool = False, output_dir: Optional[Path] = None):
        """
        Initialize debug logger.
        
        Args:
            enabled: Whether debug logging is enabled
            output_dir: Optional directory for debug output files
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir) if output_dir else Path("debug_logs")
        
        if self.enabled:
            self.output_dir.mkdir(exist_ok=True)
            debug_logger.info(f"Debug logging enabled, output dir: {self.output_dir}")
    
    def log_extraction_input(self, file_path: Path, extractor_name: str, metadata: Optional[Dict] = None):
        """Log extraction input details."""
        if not self.enabled:
            return
        
        try:
            file_stats = file_path.stat()
            input_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'extraction_input',
                'file_path': str(file_path),
                'file_size': file_stats.st_size,
                'file_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'extractor': extractor_name,
                'metadata': metadata or {}
            }
            
            debug_logger.debug(f"Extraction input: {json.dumps(input_info, indent=2)}")
            
            # Save to file for complex debugging
            debug_file = self.output_dir / f"extraction_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(debug_file, 'w') as f:
                json.dump(input_info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log extraction input: {e}")
    
    def log_extraction_output(self, file_path: Path, extractor_name: str, pages: List[PageParse], 
                             processing_time: float, success: bool, error: Optional[str] = None):
        """Log extraction output details."""
        if not self.enabled:
            return
        
        try:
            output_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'extraction_output',
                'file_path': str(file_path),
                'extractor': extractor_name,
                'processing_time': processing_time,
                'success': success,
                'error': error,
                'page_count': len(pages),
                'pages_summary': []
            }
            
            # Summarize each page
            for page in pages:
                page_summary = {
                    'page_no': page['page_no'],
                    'width': page['width'],
                    'height': page['height'],
                    'block_count': len(page['blocks']),
                    'block_types': {},
                    'total_text_length': 0,
                    'artifacts_removed': page.get('artifacts_removed', [])
                }
                
                # Count block types and text length
                for block in page['blocks']:
                    block_type = block['type']
                    page_summary['block_types'][block_type] = page_summary['block_types'].get(block_type, 0) + 1
                    page_summary['total_text_length'] += len(block.get('text', ''))
                
                output_info['pages_summary'].append(page_summary)
            
            debug_logger.debug(f"Extraction output: {json.dumps(output_info, indent=2)}")
            
            # Save detailed output for complex debugging
            if len(pages) > 0:
                debug_file = self.output_dir / f"extraction_output_{extractor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(debug_file, 'w') as f:
                    json.dump({
                        'summary': output_info,
                        'detailed_pages': [self._serialize_page(page) for page in pages]
                    }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log extraction output: {e}")
    
    def log_chunking_input(self, pages: List[PageParse], chunking_config: Dict[str, Any]):
        """Log chunking input details."""
        if not self.enabled:
            return
        
        try:
            input_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'chunking_input',
                'page_count': len(pages),
                'chunking_config': chunking_config,
                'pages_summary': []
            }
            
            # Summarize pages for chunking
            for page in pages:
                page_summary = {
                    'page_no': page['page_no'],
                    'block_count': len(page['blocks']),
                    'block_types': {},
                    'total_text_length': 0
                }
                
                for block in page['blocks']:
                    block_type = block['type']
                    page_summary['block_types'][block_type] = page_summary['block_types'].get(block_type, 0) + 1
                    page_summary['total_text_length'] += len(block.get('text', ''))
                
                input_info['pages_summary'].append(page_summary)
            
            debug_logger.debug(f"Chunking input: {json.dumps(input_info, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to log chunking input: {e}")
    
    def log_chunking_output(self, chunks: List[Chunk], processing_time: float, 
                           success: bool, error: Optional[str] = None):
        """Log chunking output details."""
        if not self.enabled:
            return
        
        try:
            output_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'chunking_output',
                'processing_time': processing_time,
                'success': success,
                'error': error,
                'chunk_count': len(chunks),
                'chunks_summary': []
            }
            
            # Summarize chunks
            total_tokens = 0
            content_types = {}
            
            for chunk in chunks:
                chunk_summary = {
                    'id': chunk['id'],
                    'token_count': chunk['token_count'],
                    'text_length': len(chunk['text']),
                    'content_type': chunk['metadata'].get('content_type'),
                    'page_start': chunk['metadata'].get('page_start'),
                    'page_end': chunk['metadata'].get('page_end'),
                    'has_html': bool(chunk.get('html')),
                    'low_conf': chunk['metadata'].get('low_conf', False)
                }
                
                total_tokens += chunk['token_count']
                content_type = chunk['metadata'].get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                output_info['chunks_summary'].append(chunk_summary)
            
            output_info['total_tokens'] = total_tokens
            output_info['content_types'] = content_types
            output_info['avg_tokens_per_chunk'] = total_tokens / len(chunks) if chunks else 0
            
            debug_logger.debug(f"Chunking output: {json.dumps(output_info, indent=2)}")
            
            # Save detailed chunking results
            if len(chunks) > 0:
                debug_file = self.output_dir / f"chunking_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(debug_file, 'w') as f:
                    json.dump({
                        'summary': output_info,
                        'detailed_chunks': [self._serialize_chunk(chunk) for chunk in chunks]
                    }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log chunking output: {e}")
    
    def log_retrieval_query(self, query: str, filters: Dict[str, Any], retrieval_config: Dict[str, Any]):
        """Log retrieval query details."""
        if not self.enabled:
            return
        
        try:
            query_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'retrieval_query',
                'query': query,
                'query_length': len(query),
                'filters': filters,
                'retrieval_config': retrieval_config
            }
            
            debug_logger.debug(f"Retrieval query: {json.dumps(query_info, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to log retrieval query: {e}")
    
    def log_retrieval_results(self, query: str, results: List[Dict[str, Any]], 
                             processing_time: float, method: str):
        """Log retrieval results details."""
        if not self.enabled:
            return
        
        try:
            results_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'retrieval_results',
                'query': query,
                'method': method,
                'processing_time': processing_time,
                'result_count': len(results),
                'results_summary': []
            }
            
            # Summarize results
            for i, result in enumerate(results):
                result_summary = {
                    'rank': i + 1,
                    'score': result.get('score', 0.0),
                    'chunk_id': result.get('chunk', {}).get('id'),
                    'content_type': result.get('chunk', {}).get('metadata', {}).get('content_type'),
                    'doc_name': result.get('chunk', {}).get('metadata', {}).get('doc_name'),
                    'page_start': result.get('chunk', {}).get('metadata', {}).get('page_start'),
                    'text_length': len(result.get('chunk', {}).get('text', ''))
                }
                results_info['results_summary'].append(result_summary)
            
            debug_logger.debug(f"Retrieval results: {json.dumps(results_info, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to log retrieval results: {e}")
    
    def log_error_context(self, error: Exception, operation: str, context: Dict[str, Any]):
        """Log detailed error context for debugging."""
        if not self.enabled:
            return
        
        try:
            # Get call stack
            stack = traceback.extract_tb(error.__traceback__)
            
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'error_context',
                'error_operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'call_stack': [
                    {
                        'filename': frame.filename,
                        'line_number': frame.lineno,
                        'function': frame.name,
                        'code': frame.line
                    }
                    for frame in stack
                ]
            }
            
            debug_logger.error(f"Error context: {json.dumps(error_info, indent=2)}")
            
            # Save detailed error context
            debug_file = self.output_dir / f"error_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(debug_file, 'w') as f:
                json.dump(error_info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log error context: {e}")
    
    def log_performance_issue(self, operation: str, duration: float, memory_used: float, 
                             threshold_duration: float = 30.0, threshold_memory: float = 1024.0):
        """Log performance issues when thresholds are exceeded."""
        if not self.enabled:
            return
        
        if duration > threshold_duration or memory_used > threshold_memory:
            try:
                perf_info = {
                    'timestamp': datetime.now().isoformat(),
                    'operation': 'performance_issue',
                    'slow_operation': operation,
                    'duration': duration,
                    'memory_used': memory_used,
                    'threshold_duration': threshold_duration,
                    'threshold_memory': threshold_memory,
                    'exceeded_duration': duration > threshold_duration,
                    'exceeded_memory': memory_used > threshold_memory
                }
                
                debug_logger.warning(f"Performance issue: {json.dumps(perf_info, indent=2)}")
                
            except Exception as e:
                logger.error(f"Failed to log performance issue: {e}")
    
    def _serialize_page(self, page: PageParse) -> Dict[str, Any]:
        """Serialize page for JSON output."""
        return {
            'page_no': page['page_no'],
            'width': page['width'],
            'height': page['height'],
            'blocks': [self._serialize_block(block) for block in page['blocks']],
            'artifacts_removed': page.get('artifacts_removed', [])
        }
    
    def _serialize_block(self, block: Block) -> Dict[str, Any]:
        """Serialize block for JSON output."""
        return {
            'type': block['type'],
            'text': block['text'][:500] + '...' if len(block['text']) > 500 else block['text'],  # Truncate long text
            'text_length': len(block['text']),
            'bbox': block['bbox'],
            'has_html': bool(block.get('html')),
            'span_count': len(block.get('spans', [])),
            'meta': block.get('meta', {})
        }
    
    def _serialize_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """Serialize chunk for JSON output."""
        return {
            'id': chunk['id'],
            'text': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text'],  # Truncate long text
            'text_length': len(chunk['text']),
            'token_count': chunk['token_count'],
            'text_hash': chunk['text_hash'],
            'has_html': bool(chunk.get('html')),
            'metadata': chunk['metadata']
        }


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get the global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(enabled=False)  # Disabled by default
    return _debug_logger


def enable_debug_logging(output_dir: Optional[Path] = None):
    """Enable debug logging."""
    global _debug_logger
    _debug_logger = DebugLogger(enabled=True, output_dir=output_dir)
    logger.info("Debug logging enabled")


def disable_debug_logging():
    """Disable debug logging."""
    global _debug_logger
    if _debug_logger:
        _debug_logger.enabled = False
    logger.info("Debug logging disabled")


def log_function_call(func_name: str, args: tuple, kwargs: dict, result: Any = None, error: Exception = None):
    """Log function call details for debugging."""
    debug_logger = get_debug_logger()
    if not debug_logger.enabled:
        return
    
    try:
        call_info = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'function_call',
            'function': func_name,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys()),
            'success': error is None,
            'error': str(error) if error else None
        }
        
        # Add result summary if available
        if result is not None:
            if isinstance(result, (list, tuple)):
                call_info['result_type'] = type(result).__name__
                call_info['result_length'] = len(result)
            elif isinstance(result, dict):
                call_info['result_type'] = 'dict'
                call_info['result_keys'] = list(result.keys())
            else:
                call_info['result_type'] = type(result).__name__
        
        debug_logger.debug_logger.debug(f"Function call: {json.dumps(call_info, indent=2)}")
        
    except Exception as e:
        logger.error(f"Failed to log function call: {e}")


def debug_function(func):
    """Decorator to automatically log function calls."""
    def wrapper(*args, **kwargs):
        debug_logger = get_debug_logger()
        if debug_logger.enabled:
            try:
                result = func(*args, **kwargs)
                log_function_call(func.__name__, args, kwargs, result=result)
                return result
            except Exception as e:
                log_function_call(func.__name__, args, kwargs, error=e)
                raise
        else:
            return func(*args, **kwargs)
    
    return wrapper