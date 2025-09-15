"""
Performance monitoring and metrics collection for the Construction RAG System.
"""
import time
import psutil
import os
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import json


logger = logging.getLogger(__name__)
perf_logger = logging.getLogger('performance')


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: float = 0.0
    memory_end: Optional[float] = None
    memory_peak: float = 0.0
    memory_used: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """Mark operation as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
        
        # Get final memory usage
        process = psutil.Process(os.getpid())
        self.memory_end = process.memory_info().rss / 1024 / 1024  # MB
        self.memory_used = self.memory_end - self.memory_start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'operation': self.operation_name,
            'duration': self.duration,
            'memory_used': self.memory_used,
            'memory_peak': self.memory_peak,
            'success': self.success,
            'error': self.error_message,
            'metadata': self.metadata,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat()
        }


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_operations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'active_operations': self.active_operations
        }


class PerformanceMonitor:
    """Performance monitoring system for tracking operations and system metrics."""
    
    def __init__(self, 
                 max_history: int = 1000,
                 system_monitoring_interval: float = 30.0,
                 enable_system_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of operation metrics to keep in memory
            system_monitoring_interval: Interval for system metrics collection (seconds)
            enable_system_monitoring: Whether to enable background system monitoring
        """
        self.max_history = max_history
        self.system_monitoring_interval = system_monitoring_interval
        self.enable_system_monitoring = enable_system_monitoring
        
        # Operation tracking
        self.active_operations: Dict[str, OperationMetrics] = {}
        self.completed_operations: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'total_memory': 0.0,
            'success_count': 0,
            'error_count': 0,
            'avg_duration': 0.0,
            'avg_memory': 0.0,
            'success_rate': 0.0
        })
        
        # System metrics
        self.system_metrics_history: deque = deque(maxlen=max_history)
        self.system_monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start system monitoring if enabled
        if self.enable_system_monitoring:
            self.start_system_monitoring()
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Optional metadata for the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            memory_start=memory_start,
            memory_peak=memory_start,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_operations[operation_id] = metrics
        
        logger.debug(f"Started operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def update_operation_memory(self, operation_id: str):
        """Update peak memory usage for an operation."""
        with self.lock:
            if operation_id in self.active_operations:
                process = psutil.Process(os.getpid())
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                metrics = self.active_operations[operation_id]
                if current_memory > metrics.memory_peak:
                    metrics.memory_peak = current_memory
    
    def complete_operation(self, 
                          operation_id: str, 
                          success: bool = True, 
                          error_message: Optional[str] = None,
                          additional_metadata: Optional[Dict[str, Any]] = None):
        """
        Complete an operation and record metrics.
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            additional_metadata: Additional metadata to record
        """
        with self.lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation ID not found: {operation_id}")
                return
            
            metrics = self.active_operations.pop(operation_id)
            metrics.complete(success, error_message)
            
            # Add additional metadata
            if additional_metadata:
                metrics.metadata.update(additional_metadata)
            
            # Update operation statistics
            stats = self.operation_stats[metrics.operation_name]
            stats['count'] += 1
            stats['total_duration'] += metrics.duration or 0
            stats['total_memory'] += metrics.memory_used or 0
            
            if success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
            
            # Calculate averages
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            stats['avg_memory'] = stats['total_memory'] / stats['count']
            stats['success_rate'] = stats['success_count'] / stats['count']
            
            # Store completed operation
            self.completed_operations.append(metrics)
        
        # Log performance metrics
        perf_logger.info(f"Operation completed: {json.dumps(metrics.to_dict())}")
        
        if success:
            logger.debug(f"Completed operation: {metrics.operation_name} "
                        f"(Duration: {metrics.duration:.2f}s, Memory: {metrics.memory_used:.1f}MB)")
        else:
            logger.error(f"Failed operation: {metrics.operation_name} "
                        f"(Duration: {metrics.duration:.2f}s, Error: {error_message})")
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get operation statistics.
        
        Args:
            operation_name: Specific operation name, or None for all operations
            
        Returns:
            Dictionary of operation statistics
        """
        with self.lock:
            if operation_name:
                return dict(self.operation_stats.get(operation_name, {}))
            else:
                return {name: dict(stats) for name, stats in self.operation_stats.items()}
    
    def get_recent_operations(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent completed operations.
        
        Args:
            count: Number of recent operations to return
            
        Returns:
            List of operation dictionaries
        """
        with self.lock:
            recent = list(self.completed_operations)[-count:]
            return [op.to_dict() for op in reversed(recent)]
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations."""
        with self.lock:
            return [
                {
                    'id': op_id,
                    'operation': metrics.operation_name,
                    'duration': time.time() - metrics.start_time,
                    'memory_current': metrics.memory_peak,
                    'metadata': metrics.metadata
                }
                for op_id, metrics in self.active_operations.items()
            ]
    
    def start_system_monitoring(self):
        """Start background system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.system_monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True
        )
        self.system_monitoring_thread.start()
        logger.info("Started system monitoring")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self.monitoring_active = False
        if self.system_monitoring_thread:
            self.system_monitoring_thread.join(timeout=5)
        logger.info("Stopped system monitoring")
    
    def _system_monitoring_loop(self):
        """Background loop for collecting system metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / 1024 / 1024,
                    disk_usage_percent=disk.percent,
                    active_operations=len(self.active_operations)
                )
                
                with self.lock:
                    self.system_metrics_history.append(metrics)
                
                # Log system metrics periodically
                perf_logger.info(f"System metrics: {json.dumps(metrics.to_dict())}")
                
                # Check for resource warnings
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                if disk.percent > 90:
                    logger.warning(f"High disk usage: {disk.percent:.1f}%")
                
                time.sleep(self.system_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.system_monitoring_interval)
    
    def get_system_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent system metrics.
        
        Args:
            count: Number of recent metrics to return
            
        Returns:
            List of system metrics dictionaries
        """
        with self.lock:
            recent = list(self.system_metrics_history)[-count:]
            return [metrics.to_dict() for metrics in reversed(recent)]
    
    def export_metrics(self, file_path: Path):
        """
        Export all metrics to a JSON file.
        
        Args:
            file_path: Path to export file
        """
        with self.lock:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'operation_stats': dict(self.operation_stats),
                'recent_operations': [op.to_dict() for op in self.completed_operations],
                'active_operations': self.get_active_operations(),
                'system_metrics': [metrics.to_dict() for metrics in self.system_metrics_history]
            }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {file_path}")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_system_monitoring()


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation
        metadata: Optional metadata for the operation
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            operation_id = monitor.start_operation(operation_name, metadata)
            
            try:
                result = func(*args, **kwargs)
                monitor.complete_operation(operation_id, success=True)
                return result
            except Exception as e:
                monitor.complete_operation(operation_id, success=False, error_message=str(e))
                raise
        
        return wrapper
    return decorator


def log_operation_metrics(operation_name: str, duration: float, memory_used: float, success: bool = True):
    """
    Log operation metrics directly.
    
    Args:
        operation_name: Name of the operation
        duration: Duration in seconds
        memory_used: Memory used in MB
        success: Whether operation succeeded
    """
    perf_logger.info(f"Operation: {operation_name}, Duration: {duration:.2f}s, "
                    f"Memory: {memory_used:.1f}MB, Success: {success}")


def log_system_resource_usage():
    """Log current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(f"System resources - CPU: {cpu_percent:.1f}%, "
                   f"Memory: {memory.percent:.1f}% ({memory.used / 1024 / 1024:.0f}MB), "
                   f"Disk: {disk.percent:.1f}%")
    except Exception as e:
        logger.error(f"Failed to log system resource usage: {e}")