"""
Log management utilities for rotation, cleanup, and storage management.
"""
import logging
import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import glob


logger = logging.getLogger(__name__)


class LogManager:
    """Manager for log file rotation, compression, and cleanup."""
    
    def __init__(self, 
                 log_dir: Path,
                 max_log_age_days: int = 30,
                 max_total_size_mb: int = 1000,
                 compress_old_logs: bool = True):
        """
        Initialize log manager.
        
        Args:
            log_dir: Directory containing log files
            max_log_age_days: Maximum age of log files in days
            max_total_size_mb: Maximum total size of all logs in MB
            compress_old_logs: Whether to compress old log files
        """
        self.log_dir = Path(log_dir)
        self.max_log_age_days = max_log_age_days
        self.max_total_size_mb = max_total_size_mb
        self.compress_old_logs = compress_old_logs
        
        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)
    
    def cleanup_old_logs(self) -> Dict[str, Any]:
        """
        Clean up old log files based on age and size limits.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'files_removed': 0,
            'files_compressed': 0,
            'space_freed_mb': 0.0,
            'errors': []
        }
        
        try:
            # Get all log files
            log_files = self._get_log_files()
            
            # Remove files older than max age
            cutoff_date = datetime.now() - timedelta(days=self.max_log_age_days)
            
            for log_file in log_files:
                try:
                    file_path = self.log_dir / log_file['name']
                    file_date = datetime.fromtimestamp(log_file['modified'])
                    
                    if file_date < cutoff_date:
                        file_size = log_file['size'] / 1024 / 1024  # MB
                        file_path.unlink()
                        
                        stats['files_removed'] += 1
                        stats['space_freed_mb'] += file_size
                        
                        logger.info(f"Removed old log file: {log_file['name']} ({file_size:.1f}MB)")
                
                except Exception as e:
                    error_msg = f"Failed to remove log file {log_file['name']}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
            # Compress old logs if enabled
            if self.compress_old_logs:
                stats.update(self._compress_old_logs())
            
            # Check total size and remove oldest if necessary
            remaining_files = self._get_log_files()
            total_size_mb = sum(f['size'] for f in remaining_files) / 1024 / 1024
            
            if total_size_mb > self.max_total_size_mb:
                stats.update(self._reduce_total_size(remaining_files, total_size_mb))
            
            logger.info(f"Log cleanup completed: {stats}")
            
        except Exception as e:
            error_msg = f"Log cleanup failed: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
        
        return stats
    
    def _get_log_files(self) -> List[Dict[str, Any]]:
        """Get list of log files with metadata."""
        log_files = []
        
        # Look for various log file patterns
        patterns = ['*.log', '*.log.*', '*.json', 'debug_logs/*.json']
        
        for pattern in patterns:
            for file_path in self.log_dir.glob(pattern):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        log_files.append({
                            'name': file_path.name,
                            'path': file_path,
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'created': stat.st_ctime
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get stats for {file_path}: {e}")
        
        # Sort by modification time (oldest first)
        log_files.sort(key=lambda x: x['modified'])
        
        return log_files
    
    def _compress_old_logs(self) -> Dict[str, Any]:
        """Compress old log files to save space."""
        stats = {
            'files_compressed': 0,
            'space_saved_mb': 0.0,
            'errors': []
        }
        
        # Find uncompressed log files older than 1 day
        cutoff_date = datetime.now() - timedelta(days=1)
        
        for log_file in glob.glob(str(self.log_dir / "*.log*")):
            log_path = Path(log_file)
            
            # Skip already compressed files
            if log_path.suffix == '.gz':
                continue
            
            # Skip current log files
            if log_path.name in ['construction_rag.log', 'errors.log', 'performance.log']:
                continue
            
            try:
                file_date = datetime.fromtimestamp(log_path.stat().st_mtime)
                
                if file_date < cutoff_date:
                    original_size = log_path.stat().st_size
                    compressed_path = log_path.with_suffix(log_path.suffix + '.gz')
                    
                    # Compress the file
                    with open(log_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    log_path.unlink()
                    
                    compressed_size = compressed_path.stat().st_size
                    space_saved = (original_size - compressed_size) / 1024 / 1024  # MB
                    
                    stats['files_compressed'] += 1
                    stats['space_saved_mb'] += space_saved
                    
                    logger.info(f"Compressed log file: {log_path.name} "
                              f"(saved {space_saved:.1f}MB)")
            
            except Exception as e:
                error_msg = f"Failed to compress log file {log_path}: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
        
        return stats
    
    def _reduce_total_size(self, log_files: List[Dict[str, Any]], current_size_mb: float) -> Dict[str, Any]:
        """Remove oldest files to reduce total size."""
        stats = {
            'files_removed': 0,
            'space_freed_mb': 0.0,
            'errors': []
        }
        
        target_size_mb = self.max_total_size_mb * 0.8  # Reduce to 80% of limit
        size_to_free_mb = current_size_mb - target_size_mb
        
        if size_to_free_mb <= 0:
            return stats
        
        logger.info(f"Total log size ({current_size_mb:.1f}MB) exceeds limit "
                   f"({self.max_total_size_mb}MB), removing oldest files")
        
        freed_mb = 0.0
        
        for log_file in log_files:
            if freed_mb >= size_to_free_mb:
                break
            
            # Don't remove current active log files
            if log_file['name'] in ['construction_rag.log', 'errors.log', 'performance.log']:
                continue
            
            try:
                file_size_mb = log_file['size'] / 1024 / 1024
                log_file['path'].unlink()
                
                stats['files_removed'] += 1
                stats['space_freed_mb'] += file_size_mb
                freed_mb += file_size_mb
                
                logger.info(f"Removed log file for size limit: {log_file['name']} "
                          f"({file_size_mb:.1f}MB)")
            
            except Exception as e:
                error_msg = f"Failed to remove log file {log_file['name']}: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
        
        return stats
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get statistics about log files."""
        try:
            log_files = self._get_log_files()
            
            total_size_mb = sum(f['size'] for f in log_files) / 1024 / 1024
            compressed_files = sum(1 for f in log_files if f['name'].endswith('.gz'))
            
            # Group by file type
            file_types = {}
            for log_file in log_files:
                ext = Path(log_file['name']).suffix
                if ext == '.gz':
                    ext = Path(log_file['name']).stem.split('.')[-1] + '.gz'
                
                if ext not in file_types:
                    file_types[ext] = {'count': 0, 'size_mb': 0.0}
                
                file_types[ext]['count'] += 1
                file_types[ext]['size_mb'] += log_file['size'] / 1024 / 1024
            
            # Find oldest and newest files
            oldest_file = min(log_files, key=lambda x: x['modified']) if log_files else None
            newest_file = max(log_files, key=lambda x: x['modified']) if log_files else None
            
            return {
                'total_files': len(log_files),
                'total_size_mb': total_size_mb,
                'compressed_files': compressed_files,
                'file_types': file_types,
                'oldest_file': {
                    'name': oldest_file['name'],
                    'date': datetime.fromtimestamp(oldest_file['modified']).isoformat()
                } if oldest_file else None,
                'newest_file': {
                    'name': newest_file['name'],
                    'date': datetime.fromtimestamp(newest_file['modified']).isoformat()
                } if newest_file else None,
                'size_limit_mb': self.max_total_size_mb,
                'age_limit_days': self.max_log_age_days
            }
        
        except Exception as e:
            logger.error(f"Failed to get log statistics: {e}")
            return {'error': str(e)}
    
    def export_logs(self, output_path: Path, include_compressed: bool = True) -> bool:
        """
        Export all logs to a compressed archive.
        
        Args:
            output_path: Path for the output archive
            include_compressed: Whether to include already compressed files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import tarfile
            
            with tarfile.open(output_path, 'w:gz') as tar:
                log_files = self._get_log_files()
                
                for log_file in log_files:
                    if not include_compressed and log_file['name'].endswith('.gz'):
                        continue
                    
                    tar.add(log_file['path'], arcname=log_file['name'])
            
            logger.info(f"Exported logs to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            return False


def setup_log_rotation(log_dir: Path, config: Optional[Dict[str, Any]] = None) -> LogManager:
    """
    Set up automatic log rotation and cleanup.
    
    Args:
        log_dir: Directory containing log files
        config: Optional configuration for log management
        
    Returns:
        LogManager instance
    """
    config = config or {}
    
    manager = LogManager(
        log_dir=log_dir,
        max_log_age_days=config.get('max_log_age_days', 30),
        max_total_size_mb=config.get('max_total_size_mb', 1000),
        compress_old_logs=config.get('compress_old_logs', True)
    )
    
    # Perform initial cleanup
    try:
        stats = manager.cleanup_old_logs()
        logger.info(f"Initial log cleanup completed: {stats}")
    except Exception as e:
        logger.error(f"Initial log cleanup failed: {e}")
    
    return manager


def schedule_log_cleanup(manager: LogManager, interval_hours: int = 24):
    """
    Schedule periodic log cleanup.
    
    Args:
        manager: LogManager instance
        interval_hours: Cleanup interval in hours
    """
    import threading
    import time
    
    def cleanup_loop():
        while True:
            try:
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
                stats = manager.cleanup_old_logs()
                logger.info(f"Scheduled log cleanup completed: {stats}")
            except Exception as e:
                logger.error(f"Scheduled log cleanup failed: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    
    logger.info(f"Scheduled log cleanup every {interval_hours} hours")