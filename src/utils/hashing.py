"""
Hashing utilities for text content and data integrity.
"""
import hashlib
from typing import Union


def generate_text_hash(text: str) -> str:
    """
    Generate SHA-256 hash for text content.
    
    Args:
        text: Input text to hash
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def generate_content_hash(content: Union[str, bytes]) -> str:
    """
    Generate SHA-256 hash for any content (text or binary).
    
    Args:
        content: Input content to hash (string or bytes)
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def generate_file_hash(file_path: str) -> str:
    """
    Generate SHA-256 hash for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal SHA-256 hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def verify_text_hash(text: str, expected_hash: str) -> bool:
    """
    Verify that text matches expected hash.
    
    Args:
        text: Text content to verify
        expected_hash: Expected SHA-256 hash
        
    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = generate_text_hash(text)
    return actual_hash == expected_hash