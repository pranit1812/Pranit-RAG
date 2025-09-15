"""
Data validation utilities for Construction RAG System types.
"""
from typing import List, Dict, Any, Optional
from .types import (
    Span, Block, PageParse, ChunkMetadata, Chunk, ProjectContext,
    validate_bbox, validate_span, validate_block, validate_page_parse,
    validate_chunk_metadata, validate_chunk, validate_project_context,
    validate_division_code, validate_section_code, extract_division_from_section,
    get_division_title, MASTERFORMAT_DIVISIONS
)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field_name: str = "", object_type: str = ""):
        self.field_name = field_name
        self.object_type = object_type
        
        error_context = []
        if object_type:
            error_context.append(f"type={object_type}")
        if field_name:
            error_context.append(f"field={field_name}")
        
        if error_context:
            full_message = f"{message} ({', '.join(error_context)})"
        else:
            full_message = message
            
        super().__init__(full_message)


class DataValidator:
    """
    Comprehensive data validator for Construction RAG System types.
    """
    
    @staticmethod
    def validate_and_raise(obj: Any, validator_func, object_type: str) -> None:
        """
        Validate object and raise ValidationError if invalid.
        
        Args:
            obj: Object to validate
            validator_func: Validation function to use
            object_type: Type name for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        if not validator_func(obj):
            raise ValidationError(f"Invalid {object_type} structure", object_type=object_type)
    
    @staticmethod
    def validate_span_strict(span: Span) -> None:
        """Validate Span with detailed error reporting."""
        DataValidator.validate_and_raise(span, validate_span, "Span")
    
    @staticmethod
    def validate_block_strict(block: Block) -> None:
        """Validate Block with detailed error reporting."""
        DataValidator.validate_and_raise(block, validate_block, "Block")
    
    @staticmethod
    def validate_page_parse_strict(page: PageParse) -> None:
        """Validate PageParse with detailed error reporting."""
        DataValidator.validate_and_raise(page, validate_page_parse, "PageParse")
    
    @staticmethod
    def validate_chunk_metadata_strict(metadata: ChunkMetadata) -> None:
        """Validate ChunkMetadata with detailed error reporting."""
        DataValidator.validate_and_raise(metadata, validate_chunk_metadata, "ChunkMetadata")
    
    @staticmethod
    def validate_chunk_strict(chunk: Chunk) -> None:
        """Validate Chunk with detailed error reporting."""
        DataValidator.validate_and_raise(chunk, validate_chunk, "Chunk")
    
    @staticmethod
    def validate_project_context_strict(context: ProjectContext) -> None:
        """Validate ProjectContext with detailed error reporting."""
        DataValidator.validate_and_raise(context, validate_project_context, "ProjectContext")
    
    @staticmethod
    def validate_batch_chunks(chunks: List[Chunk]) -> List[str]:
        """
        Validate a batch of chunks and return list of errors.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        
        for i, chunk in enumerate(chunks):
            try:
                DataValidator.validate_chunk_strict(chunk)
            except ValidationError as e:
                errors.append(f"Chunk {i}: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_batch_pages(pages: List[PageParse]) -> List[str]:
        """
        Validate a batch of pages and return list of errors.
        
        Args:
            pages: List of pages to validate
            
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        
        for i, page in enumerate(pages):
            try:
                DataValidator.validate_page_parse_strict(page)
            except ValidationError as e:
                errors.append(f"Page {i}: {str(e)}")
        
        return errors


class MasterFormatValidator:
    """
    Validator for MasterFormat codes and construction-specific data.
    """
    
    @staticmethod
    def get_all_division_codes() -> List[str]:
        """Get list of all valid division codes."""
        return list(MASTERFORMAT_DIVISIONS.keys())
    
    @staticmethod
    def get_all_division_titles() -> List[str]:
        """Get list of all division titles."""
        return list(MASTERFORMAT_DIVISIONS.values())
    
    @staticmethod
    def find_division_by_title(title: str, partial_match: bool = False) -> Optional[str]:
        """
        Find division code by title.
        
        Args:
            title: Division title to search for
            partial_match: Allow partial string matching
            
        Returns:
            Division code or None if not found
        """
        title_lower = title.lower()
        
        for code, div_title in MASTERFORMAT_DIVISIONS.items():
            if partial_match:
                if title_lower in div_title.lower():
                    return code
            else:
                if title_lower == div_title.lower():
                    return code
        
        return None
    
    @staticmethod
    def validate_discipline_code(discipline: str) -> bool:
        """
        Validate construction discipline code.
        
        Args:
            discipline: Discipline code to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_disciplines = ["A", "S", "M", "E", "P", "FP", "EL"]
        return discipline in valid_disciplines
    
    @staticmethod
    def get_discipline_name(discipline: str) -> Optional[str]:
        """
        Get full name for discipline code.
        
        Args:
            discipline: Discipline code
            
        Returns:
            Full discipline name or None if invalid
        """
        discipline_names = {
            "A": "Architectural",
            "S": "Structural", 
            "M": "Mechanical",
            "E": "Electrical",
            "P": "Plumbing",
            "FP": "Fire Protection",
            "EL": "Elevator"
        }
        return discipline_names.get(discipline)
    
    @staticmethod
    def extract_section_info(section_code: str) -> Optional[Dict[str, str]]:
        """
        Extract division and section information from section code.
        
        Args:
            section_code: Section code (e.g., "09 91 23")
            
        Returns:
            Dictionary with division_code, division_title, section_code
        """
        if not validate_section_code(section_code):
            return None
        
        division_code = extract_division_from_section(section_code)
        if not division_code:
            return None
        
        division_title = get_division_title(division_code)
        
        return {
            "division_code": division_code,
            "division_title": division_title,
            "section_code": section_code
        }


# Convenience functions for common validation tasks

def ensure_valid_chunk(chunk: Chunk) -> Chunk:
    """
    Ensure chunk is valid, raising exception if not.
    
    Args:
        chunk: Chunk to validate
        
    Returns:
        The same chunk if valid
        
    Raises:
        ValidationError: If chunk is invalid
    """
    DataValidator.validate_chunk_strict(chunk)
    return chunk


def ensure_valid_page(page: PageParse) -> PageParse:
    """
    Ensure page is valid, raising exception if not.
    
    Args:
        page: PageParse to validate
        
    Returns:
        The same page if valid
        
    Raises:
        ValidationError: If page is invalid
    """
    DataValidator.validate_page_parse_strict(page)
    return page


def filter_valid_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Filter list to only include valid chunks.
    
    Args:
        chunks: List of chunks to filter
        
    Returns:
        List containing only valid chunks
    """
    valid_chunks = []
    
    for chunk in chunks:
        if validate_chunk(chunk):
            valid_chunks.append(chunk)
    
    return valid_chunks


def filter_valid_pages(pages: List[PageParse]) -> List[PageParse]:
    """
    Filter list to only include valid pages.
    
    Args:
        pages: List of pages to filter
        
    Returns:
        List containing only valid pages
    """
    valid_pages = []
    
    for page in pages:
        if validate_page_parse(page):
            valid_pages.append(page)
    
    return valid_pages