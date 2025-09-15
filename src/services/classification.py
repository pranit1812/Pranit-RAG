"""
Content type classification and metadata extraction for construction documents.
"""
import re
from typing import Dict, List, Optional, Tuple, Literal, Any
from pathlib import Path

from models.types import Block, PageParse, ChunkMetadata
from services.filtering import MASTERFORMAT_DIVISIONS


class ContentClassifier:
    """
    Classifies document content into construction-specific types.
    """
    
    def classify_blocks(self, blocks: List[Block], filename: str = "") -> str:
        """
        Classify blocks and return content type.
        
        Args:
            blocks: List of blocks to classify
            filename: Document filename for context
            
        Returns:
            Content type string
        """
        return self.classify_content_type(blocks, filename)
    
    # ITB and instruction document filename patterns
    ITB_PATTERNS = [
        r'itb',
        r'invitation.*bid',
        r'instruction.*bidder',
        r'bid.*instruction',
        r'general.*condition',
        r'supplementary.*condition',
        r'special.*condition',
        r'addend',
        r'amendment'
    ]
    
    # Provider block type mappings to content types
    BLOCK_TYPE_MAPPING = {
        'table': 'Table',
        'list': 'List', 
        'drawing': 'Drawing',
        'titleblock': 'Drawing',
        'figure': 'Drawing',
        'paragraph': None,  # Requires further analysis
        'heading': None,    # Requires further analysis
        'caption': None,    # Context dependent
        'artifact': None    # Usually filtered out
    }
    
    def __init__(self):
        """Initialize the content classifier."""
        # Compile regex patterns for efficiency
        self.itb_regex = re.compile(
            '|'.join(self.ITB_PATTERNS), 
            re.IGNORECASE
        )
        
        # Spec section indicators
        self.spec_indicators = [
            r'section\s+\d{2}\s+\d{2}\s+\d{2}',
            r'part\s+[123]',
            r'general\s+requirements',
            r'products\s+and\s+materials',
            r'execution',
            r'submittals',
            r'quality\s+assurance',
            r'delivery.*storage.*handling',
            r'project\s+conditions',
            r'warranty'
        ]
        self.spec_regex = re.compile(
            '|'.join(self.spec_indicators),
            re.IGNORECASE
        )
    
    def classify_content_type(
        self, 
        blocks: List[Block], 
        filename: str = "",
        page_no: int = 1
    ) -> Tuple[Literal["SpecSection", "Drawing", "ITB", "Table", "List"], float]:
        """
        Classify content type based on blocks and filename.
        
        Args:
            blocks: List of blocks from the page
            filename: Original filename for context
            page_no: Page number for context
            
        Returns:
            Tuple of (content_type, confidence_score)
        """
        # Check filename-based classification first
        filename_type, filename_conf = self._classify_by_filename(filename)
        
        # Analyze block types
        block_type_scores = self._analyze_block_types(blocks)
        
        # Analyze text content
        text_scores = self._analyze_text_content(blocks)
        
        # Combine scores with weights
        combined_scores = {}
        for content_type in ["SpecSection", "Drawing", "ITB", "Table", "List"]:
            filename_score = filename_conf if filename_type == content_type else 0.0
            combined_scores[content_type] = (
                0.4 * filename_score +
                0.4 * block_type_scores.get(content_type, 0.0) +
                0.2 * text_scores.get(content_type, 0.0)
            )
        
        # Find highest scoring type
        best_type = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        best_score = combined_scores[best_type]
        
        # Apply minimum confidence threshold
        if best_score < 0.3:
            # Default to SpecSection for text-heavy content
            if any(block['type'] in ['paragraph', 'heading'] for block in blocks):
                return "SpecSection", 0.3
            else:
                return "Drawing", 0.3
        
        return best_type, min(best_score, 1.0)
    
    def _classify_by_filename(self, filename: str) -> Tuple[Optional[str], float]:
        """
        Classify content type based on filename patterns.
        
        Args:
            filename: Original filename
            
        Returns:
            Tuple of (content_type, confidence_score)
        """
        if not filename:
            return None, 0.0
        
        filename_lower = filename.lower()
        
        # ITB documents (highest priority)
        if self.itb_regex.search(filename_lower):
            return "ITB", 0.9
        
        # Specification documents (check before drawings to avoid conflicts)
        spec_patterns = [
            r'spec', r'section.*\d{2}.*\d{2}.*\d{2}',
            r'division.*\d{2}', r'masterformat'
        ]
        
        for pattern in spec_patterns:
            if re.search(pattern, filename_lower):
                return "SpecSection", 0.8
        
        # Drawing files (common extensions and patterns)
        drawing_patterns = [
            r'\.dwg$', r'\.dxf$', 
            r'^[asmepf]\d+.*\.pdf$', r'^[asmepf]\d+.*floor.*plan',
            r'sheet.*\d+', r'.*plan.*\d+',
            r'elevation', r'detail'
        ]
        
        for pattern in drawing_patterns:
            if re.search(pattern, filename_lower):
                return "Drawing", 0.7
        
        return None, 0.0
    
    def _analyze_block_types(self, blocks: List[Block]) -> Dict[str, float]:
        """
        Analyze block types to determine content classification.
        
        Args:
            blocks: List of blocks to analyze
            
        Returns:
            Dictionary of content_type -> confidence scores
        """
        if not blocks:
            return {}
        
        type_counts = {}
        total_blocks = len(blocks)
        
        for block in blocks:
            block_type = block.get('type', 'unknown')
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        scores = {}
        
        # Table classification
        table_ratio = type_counts.get('table', 0) / total_blocks
        scores['Table'] = min(table_ratio * 2.0, 1.0)  # Boost table score
        
        # List classification  
        list_ratio = type_counts.get('list', 0) / total_blocks
        scores['List'] = min(list_ratio * 2.0, 1.0)  # Boost list score
        
        # Drawing classification
        drawing_types = ['drawing', 'titleblock', 'figure']
        drawing_count = sum(type_counts.get(dt, 0) for dt in drawing_types)
        drawing_ratio = drawing_count / total_blocks
        scores['Drawing'] = min(drawing_ratio * 1.5, 1.0)
        
        # SpecSection classification (text-heavy content)
        text_types = ['paragraph', 'heading']
        text_count = sum(type_counts.get(tt, 0) for tt in text_types)
        text_ratio = text_count / total_blocks
        scores['SpecSection'] = min(text_ratio * 0.8, 1.0)
        
        # ITB gets lower priority from block types alone
        scores['ITB'] = scores.get('SpecSection', 0.0) * 0.5
        
        return scores
    
    def _analyze_text_content(self, blocks: List[Block]) -> Dict[str, float]:
        """
        Analyze text content for classification clues.
        
        Args:
            blocks: List of blocks to analyze
            
        Returns:
            Dictionary of content_type -> confidence scores
        """
        if not blocks:
            return {}
        
        # Combine all text content
        all_text = ' '.join(block.get('text', '') for block in blocks).lower()
        
        scores = {}
        
        # SpecSection indicators
        spec_matches = len(self.spec_regex.findall(all_text))
        scores['SpecSection'] = min(spec_matches * 0.2, 1.0)
        
        # ITB indicators
        itb_indicators = [
            'invitation to bid', 'bid', 'bidder', 'proposal', 'contract', 
            'general conditions', 'supplementary conditions', 'addendum', 'amendment'
        ]
        itb_count = sum(1 for indicator in itb_indicators if indicator in all_text)
        # Boost ITB score if "invitation to bid" appears
        if 'invitation to bid' in all_text:
            itb_count += 3
        scores['ITB'] = min(itb_count * 0.2, 1.0)
        
        # Drawing indicators
        drawing_indicators = [
            'sheet', 'drawing', 'plan', 'elevation', 'section', 'detail',
            'scale', 'north', 'legend', 'title block'
        ]
        drawing_count = sum(1 for indicator in drawing_indicators if indicator in all_text)
        scores['Drawing'] = min(drawing_count * 0.1, 1.0)
        
        # Table indicators (less reliable from text alone)
        table_indicators = ['table', 'column', 'row', 'cell']
        table_count = sum(1 for indicator in table_indicators if indicator in all_text)
        scores['Table'] = min(table_count * 0.05, 0.5)
        
        # List indicators
        list_indicators = ['list', 'item', 'bullet', 'numbered']
        list_count = sum(1 for indicator in list_indicators if indicator in all_text)
        scores['List'] = min(list_count * 0.05, 0.5)
        
        return scores


class MasterFormatExtractor:
    """
    Extracts MasterFormat division and section codes from text content.
    """
    
    def __init__(self):
        """Initialize the MasterFormat extractor."""
        # Division code patterns (00-48)
        self.division_patterns = [
            r'division\s+(\d{2})',
            r'div\.\s*(\d{2})',
            r'section\s+(\d{2})\s+\d{2}\s+\d{2}',
            r'^(\d{2})\s+[A-Z]'  # Start of line with division code
        ]
        
        # Section code patterns (XX XX XX format)
        self.section_patterns = [
            r'section\s+(\d{2}\s+\d{2}\s+\d{2})',
            r'(\d{2}\s+\d{2}\s+\d{2})\s*[-–—]\s*[A-Z]',
            r'^(\d{2}\s+\d{2}\s+\d{2})',  # Start of line
            r'(\d{2}\s+\d{2}\s+\d{2})\s+[A-Z][a-z]'  # Followed by title
        ]
        
        # Compile patterns for efficiency
        self.division_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                              for pattern in self.division_patterns]
        self.section_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                             for pattern in self.section_patterns]
    
    def extract_division_info(self, text: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Extract division code and title from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (division_code, division_title, confidence)
        """
        if not text:
            return None, None, 0.0
        
        # Try to find division codes
        found_divisions = set()
        
        for regex in self.division_regex:
            matches = regex.findall(text)
            for match in matches:
                if match in MASTERFORMAT_DIVISIONS:
                    found_divisions.add(match)
        
        if not found_divisions:
            return None, None, 0.0
        
        # If multiple divisions found, pick the most common one
        # For now, just pick the first valid one
        division_code = next(iter(found_divisions))
        division_title = MASTERFORMAT_DIVISIONS[division_code]
        
        # Calculate confidence based on context
        confidence = self._calculate_division_confidence(text, division_code)
        
        return division_code, division_title, confidence
    
    def extract_section_info(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """
        Extract section code and derive division information.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (section_code, division_code, division_title, confidence)
        """
        if not text:
            return None, None, None, 0.0
        
        # Try to find section codes
        found_sections = set()
        
        for regex in self.section_regex:
            matches = regex.findall(text)
            for match in matches:
                # Validate section code format
                if self._validate_section_format(match):
                    found_sections.add(match)
        
        if not found_sections:
            return None, None, None, 0.0
        
        # Pick the first valid section (could be enhanced to pick best)
        section_code = next(iter(found_sections))
        
        # Extract division from section
        division_code = section_code[:2]
        division_title = MASTERFORMAT_DIVISIONS.get(division_code)
        
        if not division_title:
            return section_code, None, None, 0.5
        
        # Calculate confidence
        confidence = self._calculate_section_confidence(text, section_code)
        
        return section_code, division_code, division_title, confidence
    
    def _validate_section_format(self, section_code: str) -> bool:
        """
        Validate section code format (XX XX XX).
        
        Args:
            section_code: Section code to validate
            
        Returns:
            True if format is valid
        """
        pattern = r'^\d{2}\s\d{2}\s\d{2}$'
        return bool(re.match(pattern, section_code))
    
    def _calculate_division_confidence(self, text: str, division_code: str) -> float:
        """
        Calculate confidence score for division detection.
        
        Args:
            text: Source text
            division_code: Detected division code
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.6
        
        # Boost confidence if division appears multiple times
        division_count = len(re.findall(rf'\b{division_code}\b', text))
        if division_count > 1:
            base_confidence += 0.2
        
        # Boost if appears with "Division" keyword
        if re.search(rf'division\s+{division_code}', text, re.IGNORECASE):
            base_confidence += 0.2
        
        # Boost if division title appears in text
        division_title = MASTERFORMAT_DIVISIONS[division_code]
        title_words = division_title.lower().split()
        if len(title_words) > 1:
            # Check if significant portion of title appears
            title_matches = sum(1 for word in title_words if word in text.lower())
            if title_matches >= len(title_words) // 2:
                base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _calculate_section_confidence(self, text: str, section_code: str) -> float:
        """
        Calculate confidence score for section detection.
        
        Args:
            text: Source text
            section_code: Detected section code
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.7
        
        # Boost if appears with "Section" keyword
        if re.search(rf'section\s+{re.escape(section_code)}', text, re.IGNORECASE):
            base_confidence += 0.2
        
        # Boost if appears at start of line (likely a header)
        if re.search(rf'^{re.escape(section_code)}', text, re.MULTILINE):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class DrawingMetadataExtractor:
    """
    Extracts metadata specific to construction drawings.
    """
    
    def __init__(self):
        """Initialize the drawing metadata extractor."""
        # Sheet number patterns
        self.sheet_patterns = [
            r'sheet\s+([A-Z]{1,2}\d+(?:\.\d+)?)',
            r'drawing\s+([A-Z]{1,2}\d+(?:\.\d+)?)',
            r'([A-Z]{1,2}\d+(?:\.\d+)?)\s*[-–—]\s*[A-Z]',
            r'^([A-Z]{1,2}\d+(?:\.\d+)?)$'  # Standalone sheet number
        ]
        
        # Title patterns (usually after sheet number)
        self.title_patterns = [
            r'sheet\s+[A-Z]{1,2}\d+(?:\.\d+)?\s*[-–—]\s*(.+?)(?:\n|$)',
            r'([A-Z][A-Z\s]+(?:PLAN|ELEVATION|SECTION|DETAIL|SCHEDULE))',
            r'title:\s*(.+?)(?:\n|$)'
        ]
        
        # Scale patterns
        self.scale_patterns = [
            r'scale[:\s]+([0-9/\\"=\s\'-:]+)',
            r'([0-9/\\"=\s\'-]+)\s*=\s*[0-9/\\"\']+',
            r'(1/\d+["\']?\s*=\s*1["\']?)',
            r'(\d+["\']?\s*=\s*\d+["\']?)',
            r'(\d+:\d+)'  # For 1:100 format
        ]
        
        # Revision patterns
        self.revision_patterns = [
            r'rev(?:ision)?[:\s]+([A-Z0-9]+)',
            r'revision\s+([A-Z0-9]+)',
            r'^([A-Z0-9])\s+\d{2}/\d{2}/\d{2,4}'  # Rev with date
        ]
        
        # Discipline mapping from sheet prefixes
        self.discipline_mapping = {
            'A': 'A',   # Architectural
            'S': 'S',   # Structural  
            'M': 'M',   # Mechanical
            'E': 'E',   # Electrical
            'P': 'P',   # Plumbing
            'FP': 'FP', # Fire Protection
            'EL': 'EL'  # Elevator
        }
        
        # Compile regex patterns
        self.sheet_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                           for pattern in self.sheet_patterns]
        self.title_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                           for pattern in self.title_patterns]
        self.scale_regex = [re.compile(pattern, re.IGNORECASE)
                           for pattern in self.scale_patterns]
        self.revision_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                              for pattern in self.revision_patterns]
    
    def extract_drawing_metadata(
        self, 
        blocks: List[Block]
    ) -> Dict[str, Optional[str]]:
        """
        Extract drawing-specific metadata from blocks.
        
        Args:
            blocks: List of blocks from the drawing page
            
        Returns:
            Dictionary with sheet_number, sheet_title, discipline, scale, revision
        """
        # Focus on title block and drawing blocks
        relevant_blocks = [
            block for block in blocks 
            if block.get('type') in ['titleblock', 'drawing', 'paragraph', 'heading']
        ]
        
        if not relevant_blocks:
            return {
                'sheet_number': None,
                'sheet_title': None, 
                'discipline': None,
                'scale': None,
                'revision': None
            }
        
        # Combine text from relevant blocks
        combined_text = '\n'.join(block.get('text', '') for block in relevant_blocks)
        
        # Extract each component
        sheet_number = self._extract_sheet_number(combined_text)
        sheet_title = self._extract_sheet_title(combined_text, sheet_number)
        discipline = self._extract_discipline(sheet_number)
        scale = self._extract_scale(combined_text)
        revision = self._extract_revision(combined_text)
        
        return {
            'sheet_number': sheet_number,
            'sheet_title': sheet_title,
            'discipline': discipline, 
            'scale': scale,
            'revision': revision
        }
    
    def _extract_sheet_number(self, text: str) -> Optional[str]:
        """Extract sheet number from text."""
        if not text:
            return None
        
        for regex in self.sheet_regex:
            matches = regex.findall(text)
            for match in matches:
                # Validate sheet number format
                if self._validate_sheet_number(match):
                    return match.upper()
        
        return None
    
    def _extract_sheet_title(self, text: str, sheet_number: Optional[str]) -> Optional[str]:
        """Extract sheet title from text."""
        if not text:
            return None
        
        for regex in self.title_regex:
            matches = regex.findall(text)
            for match in matches:
                # Clean up the title
                title = match.strip()
                if len(title) > 5 and len(title) < 100:  # Reasonable title length
                    return title
        
        # If we have a sheet number, try to find title after it
        if sheet_number:
            pattern = rf'{re.escape(sheet_number)}\s*[-–—]\s*(.+?)(?:\n|$)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5:
                    return title
        
        return None
    
    def _extract_discipline(self, sheet_number: Optional[str]) -> Optional[str]:
        """Extract discipline from sheet number prefix."""
        if not sheet_number:
            return None
        
        # Check for two-letter prefixes first (FP, EL)
        for prefix in ['FP', 'EL']:
            if sheet_number.startswith(prefix):
                return prefix
        
        # Check single-letter prefixes
        first_char = sheet_number[0]
        if first_char in self.discipline_mapping:
            return self.discipline_mapping[first_char]
        
        return None
    
    def _extract_scale(self, text: str) -> Optional[str]:
        """Extract scale information from text."""
        if not text:
            return None
        
        for regex in self.scale_regex:
            matches = regex.findall(text)
            for match in matches:
                scale = match.strip()
                if self._validate_scale(scale):
                    return scale
        
        return None
    
    def _extract_revision(self, text: str) -> Optional[str]:
        """Extract revision information from text."""
        if not text:
            return None
        
        for regex in self.revision_regex:
            matches = regex.findall(text)
            for match in matches:
                revision = match.strip().upper()
                if len(revision) <= 5:  # Reasonable revision length
                    return revision
        
        return None
    
    def _validate_sheet_number(self, sheet_number: str) -> bool:
        """Validate sheet number format."""
        if not sheet_number or len(sheet_number) > 10:
            return False
        
        # Should start with letter(s) followed by numbers
        pattern = r'^[A-Z]{1,2}\d+(?:\.\d+)?$'
        return bool(re.match(pattern, sheet_number.upper()))
    
    def _validate_scale(self, scale: str) -> bool:
        """Validate scale format."""
        if not scale or len(scale) > 30:
            return False
        
        # Common scale formats
        scale_patterns = [
            r'^\d+/\d+["\']?\s*=\s*\d+["\']?$',  # 1/4" = 1'
            r'^\d+:\d+$',                         # 1:100
            r'^1/\d+$',                          # 1/48
            r'^\d+["\']?\s*=\s*\d+["\']?$',      # 1" = 20'
            r'^\d+/\d+["\']?\s*=\s*\d+\'$'       # 1/4" = 1'
        ]
        
        return any(re.match(pattern, scale.strip()) for pattern in scale_patterns)


class ClassificationService:
    """
    Main service for content classification and metadata extraction.
    """
    
    def __init__(self):
        """Initialize the classification service."""
        self.content_classifier = ContentClassifier()
        self.masterformat_extractor = MasterFormatExtractor()
        self.drawing_extractor = DrawingMetadataExtractor()
    
    def classify_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Classify a chunk and add metadata in place.
        
        Args:
            chunk: Chunk dictionary to classify and update
        """
        try:
            # Get text and metadata
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            filename = metadata.get("doc_name", "")
            
            # Classify content type
            content_type = self.classify_content_type(
                blocks=[],  # We don't have blocks at chunk level
                filename=filename
            )
            
            if content_type:
                chunk["metadata"]["content_type"] = content_type
            else:
                chunk["metadata"]["content_type"] = "Document"
                
        except Exception as e:
            logging.warning(f"Classification failed for chunk: {e}")
            chunk["metadata"]["content_type"] = "Document"
    
    def classify_and_extract_metadata(
        self,
        page: PageParse,
        filename: str = "",
        project_id: str = "",
        doc_id: str = "",
        doc_name: str = "",
        file_type: Literal["pdf", "docx", "xlsx", "image"] = "pdf"
    ) -> Tuple[Literal["SpecSection", "Drawing", "ITB", "Table", "List"], Dict[str, any], float]:
        """
        Classify content and extract all relevant metadata.
        
        Args:
            page: Parsed page data
            filename: Original filename
            project_id: Project identifier
            doc_id: Document identifier  
            doc_name: Document name
            file_type: File type
            
        Returns:
            Tuple of (content_type, metadata_dict, overall_confidence)
        """
        blocks = page.get('blocks', [])
        page_no = page.get('page_no', 1)
        
        # Classify content type
        content_type, content_confidence = self.content_classifier.classify_content_type(
            blocks, filename, page_no
        )
        
        # Initialize metadata
        metadata = {
            'project_id': project_id,
            'doc_id': doc_id,
            'doc_name': doc_name,
            'file_type': file_type,
            'page_start': page_no,
            'page_end': page_no,
            'content_type': content_type,
            'division_code': None,
            'division_title': None,
            'section_code': None,
            'section_title': None,
            'discipline': None,
            'sheet_number': None,
            'sheet_title': None,
            'bbox_regions': [block.get('bbox', [0, 0, 0, 0]) for block in blocks],
            'low_conf': content_confidence < 0.5
        }
        
        # Extract text for analysis
        combined_text = ' '.join(block.get('text', '') for block in blocks)
        
        # Extract MasterFormat information for spec sections and ITB
        if content_type in ['SpecSection', 'ITB']:
            # Try section code first (more specific)
            section_code, div_code, div_title, section_conf = (
                self.masterformat_extractor.extract_section_info(combined_text)
            )
            
            if section_code:
                metadata['section_code'] = section_code
                metadata['division_code'] = div_code
                metadata['division_title'] = div_title
            else:
                # Fall back to division-only extraction
                div_code, div_title, div_conf = (
                    self.masterformat_extractor.extract_division_info(combined_text)
                )
                if div_code:
                    metadata['division_code'] = div_code
                    metadata['division_title'] = div_title
        
        # Extract drawing metadata
        if content_type == 'Drawing':
            drawing_metadata = self.drawing_extractor.extract_drawing_metadata(blocks)
            metadata.update(drawing_metadata)
        
        # Calculate overall confidence
        overall_confidence = content_confidence
        
        return content_type, metadata, overall_confidence