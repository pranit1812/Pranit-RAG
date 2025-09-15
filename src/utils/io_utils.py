"""
I/O utilities for file handling and path management.
"""
import os
import shutil
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import logging


class IOUtilsError(Exception):
    """Custom exception for I/O utilities errors."""
    pass


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
        
    Raises:
        IOUtilsError: If directory cannot be created
    """
    path_obj = Path(path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        raise IOUtilsError(f"Failed to create directory {path}: {e}")


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Characters to remove or replace
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename
    
    # Replace unsafe characters with underscores
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove control characters
    safe_name = ''.join(char for char in safe_name if ord(char) >= 32)
    
    # Trim whitespace and dots from ends
    safe_name = safe_name.strip(' .')
    
    # Ensure not empty
    if not safe_name:
        safe_name = "unnamed_file"
    
    # Truncate if too long
    if len(safe_name) > max_length:
        name_part, ext_part = os.path.splitext(safe_name)
        max_name_length = max_length - len(ext_part)
        safe_name = name_part[:max_name_length] + ext_part
    
    return safe_name


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        IOUtilsError: If file doesn't exist or cannot be accessed
    """
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        raise IOUtilsError(f"Failed to get file size for {file_path}: {e}")


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension (without dot).
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension in lowercase
    """
    return Path(file_path).suffix.lower().lstrip('.')


def is_supported_file_type(file_path: Union[str, Path], supported_extensions: List[str]) -> bool:
    """
    Check if file has a supported extension.
    
    Args:
        file_path: Path to the file
        supported_extensions: List of supported extensions (without dots)
        
    Returns:
        True if file type is supported, False otherwise
    """
    file_ext = get_file_extension(file_path)
    return file_ext in [ext.lower() for ext in supported_extensions]


def copy_file_safe(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Safely copy a file, creating destination directory if needed.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        Path object for destination file
        
    Raises:
        IOUtilsError: If copy operation fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise IOUtilsError(f"Source file does not exist: {src}")
    
    try:
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        return dst_path
        
    except Exception as e:
        raise IOUtilsError(f"Failed to copy {src} to {dst}: {e}")


def move_file_safe(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Safely move a file, creating destination directory if needed.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        Path object for destination file
        
    Raises:
        IOUtilsError: If move operation fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise IOUtilsError(f"Source file does not exist: {src}")
    
    try:
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        
        # Move file
        shutil.move(str(src_path), str(dst_path))
        return dst_path
        
    except Exception as e:
        raise IOUtilsError(f"Failed to move {src} to {dst}: {e}")


def delete_file_safe(file_path: Union[str, Path]) -> bool:
    """
    Safely delete a file if it exists.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        True if file was deleted, False if it didn't exist
        
    Raises:
        IOUtilsError: If deletion fails
    """
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        return False
    
    try:
        path_obj.unlink()
        return True
    except Exception as e:
        raise IOUtilsError(f"Failed to delete file {file_path}: {e}")


def delete_directory_safe(dir_path: Union[str, Path]) -> bool:
    """
    Safely delete a directory and all its contents.
    
    Args:
        dir_path: Path to the directory to delete
        
    Returns:
        True if directory was deleted, False if it didn't exist
        
    Raises:
        IOUtilsError: If deletion fails
    """
    path_obj = Path(dir_path)
    
    if not path_obj.exists():
        return False
    
    try:
        shutil.rmtree(path_obj)
        return True
    except Exception as e:
        raise IOUtilsError(f"Failed to delete directory {dir_path}: {e}")


def list_files_recursive(
    directory: Union[str, Path], 
    extensions: Optional[List[str]] = None,
    max_depth: Optional[int] = None
) -> List[Path]:
    """
    List all files in directory recursively with optional filtering.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (without dots)
        max_depth: Maximum recursion depth (None for unlimited)
        
    Returns:
        List of Path objects for matching files
        
    Raises:
        IOUtilsError: If directory cannot be accessed
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise IOUtilsError(f"Directory does not exist: {directory}")
    
    if not dir_path.is_dir():
        raise IOUtilsError(f"Path is not a directory: {directory}")
    
    files = []
    
    try:
        def _scan_directory(path: Path, current_depth: int = 0):
            if max_depth is not None and current_depth > max_depth:
                return
            
            for item in path.iterdir():
                if item.is_file():
                    if extensions is None or get_file_extension(item) in [ext.lower() for ext in extensions]:
                        files.append(item)
                elif item.is_dir():
                    _scan_directory(item, current_depth + 1)
        
        _scan_directory(dir_path)
        return sorted(files)
        
    except Exception as e:
        raise IOUtilsError(f"Failed to list files in {directory}: {e}")


def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        IOUtilsError: If file cannot be read or parsed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOUtilsError(f"Failed to read JSON file {file_path}: {e}")


def write_json_file(file_path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> None:
    """
    Write data to JSON file.
    
    Args:
        file_path: Path to JSON file
        data: Data to write
        indent: JSON indentation level
        
    Raises:
        IOUtilsError: If file cannot be written
    """
    path_obj = Path(file_path)
    
    try:
        # Ensure directory exists
        ensure_directory(path_obj.parent)
        
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            
    except Exception as e:
        raise IOUtilsError(f"Failed to write JSON file {file_path}: {e}")


def read_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read text file content.
    
    Args:
        file_path: Path to text file
        encoding: Text encoding
        
    Returns:
        File content as string
        
    Raises:
        IOUtilsError: If file cannot be read
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise IOUtilsError(f"Failed to read text file {file_path}: {e}")


def write_text_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """
    Write content to text file.
    
    Args:
        file_path: Path to text file
        content: Content to write
        encoding: Text encoding
        
    Raises:
        IOUtilsError: If file cannot be written
    """
    path_obj = Path(file_path)
    
    try:
        # Ensure directory exists
        ensure_directory(path_obj.parent)
        
        with open(path_obj, 'w', encoding=encoding) as f:
            f.write(content)
            
    except Exception as e:
        raise IOUtilsError(f"Failed to write text file {file_path}: {e}")


def create_temp_file(suffix: str = "", prefix: str = "temp_", dir: Optional[str] = None) -> str:
    """
    Create a temporary file and return its path.
    
    Args:
        suffix: File suffix/extension
        prefix: File prefix
        dir: Directory for temp file (None for system temp)
        
    Returns:
        Path to temporary file
        
    Raises:
        IOUtilsError: If temp file cannot be created
    """
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        os.close(fd)  # Close file descriptor, keep file
        return temp_path
    except Exception as e:
        raise IOUtilsError(f"Failed to create temporary file: {e}")


def create_temp_directory(prefix: str = "temp_", dir: Optional[str] = None) -> str:
    """
    Create a temporary directory and return its path.
    
    Args:
        prefix: Directory prefix
        dir: Parent directory for temp dir (None for system temp)
        
    Returns:
        Path to temporary directory
        
    Raises:
        IOUtilsError: If temp directory cannot be created
    """
    try:
        return tempfile.mkdtemp(prefix=prefix, dir=dir)
    except Exception as e:
        raise IOUtilsError(f"Failed to create temporary directory: {e}")


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Calculate total size of directory and all its contents.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
        
    Raises:
        IOUtilsError: If directory cannot be accessed
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise IOUtilsError(f"Directory does not exist: {directory}")
    
    total_size = 0
    
    try:
        for item in dir_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size
    except Exception as e:
        raise IOUtilsError(f"Failed to calculate directory size for {directory}: {e}")


def cleanup_temp_files(temp_paths: List[str]) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        temp_paths: List of temporary file/directory paths to clean up
    """
    for temp_path in temp_paths:
        try:
            path_obj = Path(temp_path)
            if path_obj.exists():
                if path_obj.is_file():
                    path_obj.unlink()
                elif path_obj.is_dir():
                    shutil.rmtree(path_obj)
        except Exception as e:
            logging.warning(f"Failed to cleanup temp path {temp_path}: {e}")