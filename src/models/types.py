"""
Core type definitions for the Construction RAG System.
"""
from typing import Dict, List, Optional, Any, Literal, TypedDict
import hashlib
import re


class Span(TypedDict):
    """Text span with bounding box and confidence information."""
    text: str
    bbox: List[float]  # [x0, y0, x1, y1]
    rot: float
    conf: float


class Block(TypedDict):
    """Document block containing text, structure, and metadata."""
    type: Literal["paragraph", "heading", "table", "list", "figure", "caption", "artifact", "titleblock", "drawing"]
    text: str
    html: Optional[str]  # For tables
    bbox: List[float]    # [x0, y0, x1, y1]
    spans: List[Span]
    meta: Dict[str, Any]


class PageParse(TypedDict):
    """Parsed page with blocks and metadata."""
    page_no: int
    width: int
    height: int
    blocks: List[Block]
    artifacts_removed: List[str]


class ChunkMetadata(TypedDict):
    """Comprehensive metadata for document chunks."""
    project_id: str
    doc_id: str
    doc_name: str
    file_type: Literal["pdf", "docx", "xlsx", "image"]
    page_start: int
    page_end: int
    content_type: Literal["SpecSection", "Drawing", "ITB", "Table", "List"]
    division_code: Optional[str]
    division_title: Optional[str]
    section_code: Optional[str]
    section_title: Optional[str]
    discipline: Optional[Literal["A", "S", "M", "E", "P", "FP", "EL"]]
    sheet_number: Optional[str]
    sheet_title: Optional[str]
    bbox_regions: List[List[float]]
    low_conf: bool


class Chunk(TypedDict):
    """Document chunk with content and metadata."""
    id: str
    text: str
    html: Optional[str]
    metadata: ChunkMetadata
    token_count: int
    text_hash: str


class ProjectContext(TypedDict):
    """Project context for query enhancement."""
    project_name: str
    description: str
    project_type: str  # e.g., "Commercial Office Building", "Residential Complex"
    location: Optional[str]
    key_systems: List[str]  # e.g., ["HVAC", "Electrical", "Plumbing"]
    disciplines_involved: List[str]
    summary: str  # Auto-generated or user-provided project overview


class ChunkPolicy(TypedDict):
    """Configuration for chunking behavior."""
    target_tokens: int
    max_tokens: int
    preserve_tables: bool
    preserve_lists: bool
    drawing_cluster_text: bool
    drawing_max_regions: int


class Hit(TypedDict):
    """Search result hit with score and chunk."""
    id: str
    score: float
    chunk: Chunk


class SourceInfo(TypedDict):
    """Source information for citations."""
    doc_name: str
    page_number: int
    sheet_number: Optional[str]


class ContextPacket(TypedDict):
    """Context packet for LLM assembly."""
    chunks: List[Chunk]
    total_tokens: int
    sources: Dict[str, SourceInfo]
    project_context: ProjectContext


class VisionConfig(TypedDict):
    """Configuration for vision assistance."""
    enabled: bool
    max_images: int  # 1-5 chunks
    resolution_scale: float  # 2x for high quality


def generate_text_hash(text: str) -> str:
    """Generate SHA-256 hash for text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# Validation Functions

def validate_bbox(bbox: List[float]) -> bool:
    """
    Validate bounding box format and values.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        
    Returns:
        True if bbox is valid, False otherwise
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    
    try:
        x0, y0, x1, y1 = bbox
        # Check that coordinates are numbers
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            return False
        # Check that x1 >= x0 and y1 >= y0
        return x1 >= x0 and y1 >= y0
    except (ValueError, TypeError):
        return False


def validate_span(span: Span) -> bool:
    """
    Validate Span structure and data integrity.
    
    Args:
        span: Span object to validate
        
    Returns:
        True if span is valid, False otherwise
    """
    try:
        # Check required fields
        if not isinstance(span.get('text'), str):
            return False
        
        # Validate bbox
        bbox = span.get('bbox')
        if not validate_bbox(bbox):
            return False
        
        # Check rotation and confidence
        rot = span.get('rot', 0.0)
        conf = span.get('conf', 1.0)
        
        if not isinstance(rot, (int, float)):
            return False
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return False
        
        return True
    except (AttributeError, TypeError):
        return False


def validate_block(block: Block) -> bool:
    """
    Validate Block structure and data integrity.
    
    Args:
        block: Block object to validate
        
    Returns:
        True if block is valid, False otherwise
    """
    try:
        # Check block type
        valid_types = ["paragraph", "heading", "table", "list", "figure", "caption", "artifact", "titleblock", "drawing"]
        if block.get('type') not in valid_types:
            return False
        
        # Check text field
        if not isinstance(block.get('text'), str):
            return False
        
        # Validate bbox
        bbox = block.get('bbox')
        if not validate_bbox(bbox):
            return False
        
        # Validate spans
        spans = block.get('spans', [])
        if not isinstance(spans, list):
            return False
        
        for span in spans:
            if not validate_span(span):
                return False
        
        # Check optional fields
        html = block.get('html')
        if html is not None and not isinstance(html, str):
            return False
        
        meta = block.get('meta', {})
        if not isinstance(meta, dict):
            return False
        
        return True
    except (AttributeError, TypeError):
        return False


def validate_page_parse(page: PageParse) -> bool:
    """
    Validate PageParse structure and data integrity.
    
    Args:
        page: PageParse object to validate
        
    Returns:
        True if page is valid, False otherwise
    """
    try:
        # Check page number
        page_no = page.get('page_no')
        if not isinstance(page_no, int) or page_no < 0:
            return False
        
        # Check dimensions
        width = page.get('width')
        height = page.get('height')
        if not isinstance(width, (int, float)) or width <= 0:
            return False
        if not isinstance(height, (int, float)) or height <= 0:
            return False
        
        # Validate blocks
        blocks = page.get('blocks', [])
        if not isinstance(blocks, list):
            return False
        
        for block in blocks:
            if not validate_block(block):
                return False
        
        # Check artifacts_removed
        artifacts = page.get('artifacts_removed', [])
        if not isinstance(artifacts, list):
            return False
        
        for artifact in artifacts:
            if not isinstance(artifact, str):
                return False
        
        return True
    except (AttributeError, TypeError):
        return False


def validate_chunk_metadata(metadata: ChunkMetadata) -> bool:
    """
    Validate ChunkMetadata structure and data integrity.
    
    Args:
        metadata: ChunkMetadata object to validate
        
    Returns:
        True if metadata is valid, False otherwise
    """
    try:
        # Check required string fields
        required_strings = ['project_id', 'doc_id', 'doc_name']
        for field in required_strings:
            if not isinstance(metadata.get(field), str) or not metadata.get(field):
                return False
        
        # Check file_type
        valid_file_types = ["pdf", "docx", "xlsx", "image"]
        if metadata.get('file_type') not in valid_file_types:
            return False
        
        # Check page numbers
        page_start = metadata.get('page_start')
        page_end = metadata.get('page_end')
        if not isinstance(page_start, int) or page_start < 0:
            return False
        if not isinstance(page_end, int) or page_end < page_start:
            return False
        
        # Check content_type
        valid_content_types = ["SpecSection", "Drawing", "ITB", "Table", "List"]
        if metadata.get('content_type') not in valid_content_types:
            return False
        
        # Check optional fields
        optional_strings = ['division_code', 'division_title', 'section_code', 'section_title', 'sheet_number', 'sheet_title']
        for field in optional_strings:
            value = metadata.get(field)
            if value is not None and not isinstance(value, str):
                return False
        
        # Check discipline
        discipline = metadata.get('discipline')
        if discipline is not None:
            valid_disciplines = ["A", "S", "M", "E", "P", "FP", "EL"]
            if discipline not in valid_disciplines:
                return False
        
        # Check bbox_regions
        bbox_regions = metadata.get('bbox_regions', [])
        if not isinstance(bbox_regions, list):
            return False
        
        for bbox in bbox_regions:
            if not validate_bbox(bbox):
                return False
        
        # Check low_conf flag
        low_conf = metadata.get('low_conf', False)
        if not isinstance(low_conf, bool):
            return False
        
        return True
    except (AttributeError, TypeError):
        return False


def validate_chunk(chunk: Chunk) -> bool:
    """
    Validate Chunk structure and data integrity.
    
    Args:
        chunk: Chunk object to validate
        
    Returns:
        True if chunk is valid, False otherwise
    """
    try:
        # Check required fields
        chunk_id = chunk.get('id')
        if not isinstance(chunk_id, str) or not chunk_id:
            return False
        
        text = chunk.get('text')
        if not isinstance(text, str):
            return False
        
        # Check optional html field
        html = chunk.get('html')
        if html is not None and not isinstance(html, str):
            return False
        
        # Validate metadata
        metadata = chunk.get('metadata')
        if not validate_chunk_metadata(metadata):
            return False
        
        # Check token_count
        token_count = chunk.get('token_count')
        if not isinstance(token_count, int) or token_count < 0:
            return False
        
        # Check text_hash
        text_hash = chunk.get('text_hash')
        if not isinstance(text_hash, str) or not text_hash:
            return False
        
        # Verify text_hash matches text content
        expected_hash = generate_text_hash(text)
        if text_hash != expected_hash:
            return False
        
        return True
    except (AttributeError, TypeError):
        return False


def validate_project_context(context: ProjectContext) -> bool:
    """
    Validate ProjectContext structure and data integrity.
    
    Args:
        context: ProjectContext object to validate
        
    Returns:
        True if context is valid, False otherwise
    """
    try:
        # Check required string fields
        required_strings = ['project_name', 'description', 'project_type', 'summary']
        for field in required_strings:
            if not isinstance(context.get(field), str) or not context.get(field):
                return False
        
        # Check optional location
        location = context.get('location')
        if location is not None and not isinstance(location, str):
            return False
        
        # Check key_systems
        key_systems = context.get('key_systems', [])
        if not isinstance(key_systems, list):
            return False
        
        for system in key_systems:
            if not isinstance(system, str):
                return False
        
        # Check disciplines_involved
        disciplines = context.get('disciplines_involved', [])
        if not isinstance(disciplines, list):
            return False
        
        for discipline in disciplines:
            if not isinstance(discipline, str):
                return False
        
        return True
    except (AttributeError, TypeError):
        return False


def validate_division_code(code: str) -> bool:
    """
    Validate MasterFormat division code.
    
    Args:
        code: Division code to validate
        
    Returns:
        True if code is valid, False otherwise
    """
    if not isinstance(code, str):
        return False
    
    # Import here to avoid circular imports
    from services.filtering import MASTERFORMAT_DIVISIONS
    
    # Check if code exists in MasterFormat divisions
    return code in MASTERFORMAT_DIVISIONS


def validate_section_code(code: str) -> bool:
    """
    Validate MasterFormat section code format (e.g., "09 91 23").
    
    Args:
        code: Section code to validate
        
    Returns:
        True if code format is valid, False otherwise
    """
    if not isinstance(code, str):
        return False
    
    # Pattern for section codes: XX XX XX (two digits, space, two digits, space, two digits)
    pattern = r'^\d{2}\s\d{2}\s\d{2}$'
    return bool(re.match(pattern, code))


def extract_division_from_section(section_code: str) -> Optional[str]:
    """
    Extract division code from section code.
    
    Args:
        section_code: Section code (e.g., "09 91 23")
        
    Returns:
        Division code (e.g., "09") or None if invalid
    """
    if not validate_section_code(section_code):
        return None
    
    return section_code[:2]


def get_division_title(division_code: str) -> Optional[str]:
    """
    Get division title from division code.
    
    Args:
        division_code: Division code (e.g., "09")
        
    Returns:
        Division title or None if code is invalid
    """
    # Import here to avoid circular imports
    from services.filtering import MASTERFORMAT_DIVISIONS
    
    return MASTERFORMAT_DIVISIONS.get(division_code)