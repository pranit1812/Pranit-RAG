"""
Base extractor interface and abstract class for document parsing.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List
from src.models.types import PageParse


class BaseExtractor(ABC):
    """
    Abstract base class for document extractors.
    
    All extractors must implement the supports() and parse_page() methods
    to provide consistent interface for document processing pipeline.
    """
    
    @abstractmethod
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this extractor can process the file, False otherwise
        """
        pass
    
    @abstractmethod
    def parse_page(self, file_path: Union[str, Path], page_no: int) -> PageParse:
        """
        Parse a specific page from the document.
        
        Args:
            file_path: Path to the document file
            page_no: Page number to parse (0-indexed)
            
        Returns:
            PageParse object containing extracted content and metadata
            
        Raises:
            ExtractorError: If parsing fails
            ValueError: If page_no is invalid
        """
        pass
    
    def get_page_count(self, file_path: Union[str, Path]) -> int:
        """
        Get the total number of pages in the document.
        
        Default implementation returns 1. Override for multi-page documents.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of pages in the document
        """
        return 1
    
    def parse_all_pages(self, file_path: Union[str, Path]) -> List[PageParse]:
        """
        Parse all pages from the document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of PageParse objects for all pages
        """
        page_count = self.get_page_count(file_path)
        pages = []
        
        for page_no in range(page_count):
            try:
                page = self.parse_page(file_path, page_no)
                pages.append(page)
            except Exception as e:
                # Log error but continue with other pages
                import logging
                logging.error(f"Failed to parse page {page_no} from {file_path}: {e}")
                continue
        
        return pages
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this extractor.
        
        Returns:
            List of supported file extensions (without dots)
        """
        return []
    
    def get_extractor_name(self) -> str:
        """
        Get the name of this extractor for logging and debugging.
        
        Returns:
            Human-readable name of the extractor
        """
        return self.__class__.__name__


class ExtractorError(Exception):
    """Custom exception for extractor-related errors."""
    
    def __init__(self, message: str, extractor_name: str = "", file_path: str = "", page_no: int = -1):
        self.extractor_name = extractor_name
        self.file_path = file_path
        self.page_no = page_no
        
        error_context = []
        if extractor_name:
            error_context.append(f"extractor={extractor_name}")
        if file_path:
            error_context.append(f"file={file_path}")
        if page_no >= 0:
            error_context.append(f"page={page_no}")
        
        if error_context:
            full_message = f"{message} ({', '.join(error_context)})"
        else:
            full_message = message
            
        super().__init__(full_message)


class ExtractionResult:
    """
    Container for extraction results with metadata and error handling.
    """
    
    def __init__(self, file_path: Union[str, Path], extractor_name: str):
        self.file_path = str(file_path)
        self.extractor_name = extractor_name
        self.pages: List[PageParse] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.processing_time: float = 0.0
        self.success = False
    
    def add_page(self, page: PageParse) -> None:
        """Add a successfully parsed page."""
        self.pages.append(page)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def mark_success(self) -> None:
        """Mark extraction as successful."""
        self.success = True
    
    def has_errors(self) -> bool:
        """Check if extraction had errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if extraction had warnings."""
        return len(self.warnings) > 0
    
    def get_page_count(self) -> int:
        """Get number of successfully parsed pages."""
        return len(self.pages)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "extractor_name": self.extractor_name,
            "page_count": len(self.pages),
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time": self.processing_time,
            "success": self.success
        }