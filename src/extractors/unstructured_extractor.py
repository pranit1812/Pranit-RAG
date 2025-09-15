"""
Unstructured extractor using hi-res strategy for semantic document partitioning.
"""
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import re

from src.extractors.base import BaseExtractor, ExtractorError
from src.models.types import PageParse, Block, Span
from src.utils.bbox import normalize_bbox

# Optional dependency with graceful fallback
try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.pptx import partition_pptx
    from unstructured.partition.image import partition_image
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("Unstructured not available. Unstructured extraction will be disabled.")


class UnstructuredExtractor(BaseExtractor):
    """
    Unstructured extractor using hi-res strategy for ML-based semantic partitioning.
    
    Maps Unstructured Elements to Block format with proper type classification
    and integrates OCR capabilities for image pages.
    """
    
    def __init__(self, 
                 strategy: str = "hi_res",
                 use_ocr: bool = True,
                 extract_images: bool = False,
                 infer_table_structure: bool = True,
                 chunking_strategy: str = None):
        """
        Initialize the Unstructured extractor.
        
        Args:
            strategy: Partitioning strategy ("hi_res", "fast", "ocr_only", "auto")
            use_ocr: Whether to use OCR for scanned documents
            extract_images: Whether to extract embedded images
            infer_table_structure: Whether to infer table structure
            chunking_strategy: Optional chunking strategy for Unstructured
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("Unstructured is required for UnstructuredExtractor")
        
        self.strategy = strategy
        self.use_ocr = use_ocr
        self.extract_images = extract_images
        self.infer_table_structure = infer_table_structure
        self.chunking_strategy = chunking_strategy
        
        # Validate strategy
        valid_strategies = ["hi_res", "fast", "ocr_only", "auto"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file format is supported by Unstructured, False otherwise
        """
        if not UNSTRUCTURED_AVAILABLE:
            return False
        
        path = Path(file_path)
        supported_extensions = [
            '.pdf', '.docx', '.pptx', '.xlsx', '.html', '.xml', '.txt', '.md',
            '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.heic'
        ]
        return path.suffix.lower() in supported_extensions
    
    def get_page_count(self, file_path: Union[str, Path]) -> int:
        """
        Get the total number of pages in the document.
        
        Note: Unstructured doesn't directly provide page count without processing,
        so we estimate or process the document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Estimated number of pages
        """
        path = Path(file_path)
        
        # For images, always 1 page
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.heic']
        if path.suffix.lower() in image_extensions:
            return 1
        
        # For other formats, try to get page count by processing
        try:
            elements = self._partition_document(file_path)
            
            # Count unique page numbers
            page_numbers = set()
            for element in elements:
                if hasattr(element, 'metadata') and element.metadata:
                    page_num = getattr(element.metadata, 'page_number', None)
                    if page_num is not None:
                        page_numbers.add(page_num)
            
            return len(page_numbers) if page_numbers else 1
            
        except Exception as e:
            if "poppler" in str(e).lower():
                logging.warning(f"Unstructured requires poppler for PDF processing. Install poppler-utils or use other extractors for PDFs.")
            else:
                logging.warning(f"Failed to get page count from Unstructured: {e}")
            return 1  # Fallback
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return [
            'pdf', 'docx', 'pptx', 'xlsx', 'html', 'xml', 'txt', 'md',
            'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'heic'
        ]
    
    def parse_page(self, file_path: Union[str, Path], page_no: int) -> PageParse:
        """
        Parse a specific page using Unstructured.
        
        Args:
            file_path: Path to the document file
            page_no: Page number to parse (0-indexed)
            
        Returns:
            PageParse object containing extracted content and metadata
            
        Raises:
            ExtractorError: If parsing fails
        """
        try:
            # Partition the entire document
            elements = self._partition_document(file_path)
            
            # Filter elements for the specific page
            page_elements = self._filter_elements_by_page(elements, page_no)
            
            if not page_elements:
                # If no elements found for this page, check if page exists
                max_page = self._get_max_page_number(elements)
                if page_no > max_page:
                    raise ExtractorError(f"Page {page_no} does not exist (max page: {max_page})", 
                                       self.get_extractor_name(), str(file_path), page_no)
                
                # Return empty page if no elements but page exists
                return PageParse(
                    page_no=page_no,
                    width=612,  # Default dimensions
                    height=792,
                    blocks=[],
                    artifacts_removed=[]
                )
            
            # Estimate page dimensions
            width, height = self._estimate_page_dimensions(page_elements)
            
            # Convert Unstructured elements to blocks
            blocks = self._convert_elements_to_blocks(page_elements, width, height)
            
            # Post-process blocks
            processed_blocks = self._post_process_blocks(blocks)
            
            return PageParse(
                page_no=page_no,
                width=int(width),
                height=int(height),
                blocks=processed_blocks,
                artifacts_removed=[]
            )
            
        except Exception as e:
            raise ExtractorError(f"Failed to parse page {page_no}: {e}", self.get_extractor_name(), str(file_path), page_no)
    
    def _partition_document(self, file_path: Union[str, Path]) -> List:
        """
        Partition document using Unstructured.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Unstructured elements
        """
        path = Path(file_path)
        
        # Common parameters
        common_params = {
            "strategy": self.strategy,
            "infer_table_structure": self.infer_table_structure,
            "include_page_breaks": True,
            "extract_images_in_pdf": self.extract_images,
        }
        
        # Add chunking strategy if specified
        if self.chunking_strategy:
            common_params["chunking_strategy"] = self.chunking_strategy
        
        try:
            # Use format-specific partitioners for better control
            if path.suffix.lower() == '.pdf':
                return partition_pdf(str(file_path), **common_params)
            elif path.suffix.lower() == '.docx':
                return partition_docx(str(file_path), **common_params)
            elif path.suffix.lower() == '.pptx':
                return partition_pptx(str(file_path), **common_params)
            elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.heic']:
                return partition_image(str(file_path), **common_params)
            else:
                # Use auto partitioner for other formats
                return partition(str(file_path), **common_params)
                
        except Exception as e:
            logging.error(f"Unstructured partitioning failed: {e}")
            raise
    
    def _filter_elements_by_page(self, elements: List, page_no: int) -> List:
        """
        Filter elements for a specific page.
        
        Args:
            elements: List of Unstructured elements
            page_no: Page number (0-indexed)
            
        Returns:
            List of elements for the specified page
        """
        page_elements = []
        target_page = page_no + 1  # Unstructured uses 1-indexed pages
        
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata:
                element_page = getattr(element.metadata, 'page_number', None)
                if element_page == target_page:
                    page_elements.append(element)
            elif page_no == 0:
                # If no page metadata, assume it's on the first page
                page_elements.append(element)
        
        return page_elements
    
    def _get_max_page_number(self, elements: List) -> int:
        """
        Get the maximum page number from elements.
        
        Args:
            elements: List of Unstructured elements
            
        Returns:
            Maximum page number (0-indexed)
        """
        max_page = 0
        
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata:
                page_num = getattr(element.metadata, 'page_number', None)
                if page_num is not None:
                    max_page = max(max_page, page_num - 1)  # Convert to 0-indexed
        
        return max_page
    
    def _estimate_page_dimensions(self, elements: List) -> tuple[float, float]:
        """
        Estimate page dimensions from element coordinates.
        
        Args:
            elements: List of Unstructured elements for the page
            
        Returns:
            Tuple of (width, height)
        """
        max_x, max_y = 612.0, 792.0  # Default letter size
        
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata:
                coordinates = getattr(element.metadata, 'coordinates', None)
                if coordinates and hasattr(coordinates, 'points'):
                    for point in coordinates.points:
                        if hasattr(point, 'x') and hasattr(point, 'y'):
                            max_x = max(max_x, point.x)
                            max_y = max(max_y, point.y)
        
        return max_x, max_y
    
    def _convert_elements_to_blocks(self, elements: List, width: float, height: float) -> List[Block]:
        """
        Convert Unstructured elements to Block objects.
        
        Args:
            elements: List of Unstructured elements
            width: Page width
            height: Page height
            
        Returns:
            List of Block objects
        """
        blocks = []
        
        for i, element in enumerate(elements):
            try:
                block = self._convert_element_to_block(element, i, width, height)
                if block:
                    blocks.append(block)
            except Exception as e:
                logging.warning(f"Failed to convert Unstructured element {i}: {e}")
                continue
        
        return blocks
    
    def _convert_element_to_block(self, element, element_id: int, width: float, height: float) -> Optional[Block]:
        """
        Convert a single Unstructured element to a Block.
        
        Args:
            element: Unstructured element
            element_id: Element identifier
            width: Page width
            height: Page height
            
        Returns:
            Block object or None if conversion fails
        """
        # Get element text
        text = str(element).strip() if element else ""
        if not text:
            return None
        
        # Get element type and map to our block types
        element_type = element.__class__.__name__
        block_type = self._map_unstructured_type(element_type)
        
        # Get bounding box
        bbox = self._get_element_bbox(element, width, height)
        
        # Create spans
        spans = self._create_spans_from_element(element, bbox)
        
        # Handle tables specially
        html_content = None
        if block_type == "table" and hasattr(element, 'metadata'):
            html_content = self._extract_table_html(element)
        
        # Extract metadata
        metadata = self._extract_element_metadata(element)
        
        # Create block
        block = Block(
            type=block_type,
            text=text,
            html=html_content,
            bbox=bbox,
            spans=spans,
            meta={
                "source": "unstructured",
                "element_type": element_type,
                "element_id": element_id,
                "strategy": self.strategy,
                **metadata
            }
        )
        
        return block
    
    def _map_unstructured_type(self, element_type: str) -> str:
        """
        Map Unstructured element type to our block type.
        
        Args:
            element_type: Unstructured element type name
            
        Returns:
            Block type string
        """
        type_mapping = {
            'Title': 'heading',
            'Header': 'heading',
            'NarrativeText': 'paragraph',
            'Text': 'paragraph',
            'UncategorizedText': 'paragraph',
            'Table': 'table',
            'ListItem': 'list',
            'BulletedText': 'list',
            'Figure': 'figure',
            'FigureCaption': 'caption',
            'Image': 'figure',
            'Footer': 'paragraph',
            'PageBreak': 'artifact',
            'Address': 'paragraph',
            'EmailAddress': 'paragraph'
        }
        
        return type_mapping.get(element_type, 'paragraph')
    
    def _get_element_bbox(self, element, width: float, height: float) -> List[float]:
        """
        Extract bounding box from Unstructured element.
        
        Args:
            element: Unstructured element
            width: Page width
            height: Page height
            
        Returns:
            Normalized bounding box [x0, y0, x1, y1]
        """
        try:
            if hasattr(element, 'metadata') and element.metadata:
                coordinates = getattr(element.metadata, 'coordinates', None)
                if coordinates and hasattr(coordinates, 'points'):
                    points = coordinates.points
                    if len(points) >= 2:
                        # Extract min/max coordinates
                        x_coords = [p.x for p in points if hasattr(p, 'x')]
                        y_coords = [p.y for p in points if hasattr(p, 'y')]
                        
                        if x_coords and y_coords:
                            x0, x1 = min(x_coords), max(x_coords)
                            y0, y1 = min(y_coords), max(y_coords)
                            
                            return normalize_bbox([x0, y0, x1, y1], width, height)
            
            # Fallback bbox
            return [0.0, 0.0, 1.0, 1.0]
            
        except Exception as e:
            logging.warning(f"Failed to extract bbox from element: {e}")
            return [0.0, 0.0, 1.0, 1.0]
    
    def _create_spans_from_element(self, element, bbox: List[float]) -> List[Span]:
        """
        Create text spans from Unstructured element.
        
        Args:
            element: Unstructured element
            bbox: Element bounding box
            
        Returns:
            List of Span objects
        """
        text = str(element).strip()
        if not text:
            return []
        
        # For now, create a single span per element
        # In the future, this could be enhanced to create multiple spans
        # based on text formatting or sub-elements
        span = Span(
            text=text,
            bbox=bbox,
            rot=0.0,
            conf=1.0  # Unstructured doesn't provide confidence scores
        )
        
        return [span]
    
    def _extract_table_html(self, element) -> Optional[str]:
        """
        Extract HTML representation of table element.
        
        Args:
            element: Unstructured table element
            
        Returns:
            HTML table string or None
        """
        try:
            if hasattr(element, 'metadata') and element.metadata:
                # Check if table HTML is available in metadata
                table_html = getattr(element.metadata, 'text_as_html', None)
                if table_html:
                    return table_html
                
                # Try to construct HTML from table structure
                if hasattr(element.metadata, 'table_data'):
                    return self._construct_table_html(element.metadata.table_data)
            
            return None
            
        except Exception as e:
            logging.warning(f"Failed to extract table HTML: {e}")
            return None
    
    def _construct_table_html(self, table_data) -> str:
        """
        Construct HTML table from structured table data.
        
        Args:
            table_data: Structured table data
            
        Returns:
            HTML table string
        """
        try:
            if not table_data:
                return ""
            
            html = "<table>\n"
            
            for i, row in enumerate(table_data):
                html += "  <tr>\n"
                tag = "th" if i == 0 else "td"
                
                for cell in row:
                    cell_text = str(cell).strip() if cell else ""
                    html += f"    <{tag}>{self._escape_html(cell_text)}</{tag}>\n"
                
                html += "  </tr>\n"
            
            html += "</table>"
            return html
            
        except Exception as e:
            logging.warning(f"Failed to construct table HTML: {e}")
            return ""
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    def _extract_element_metadata(self, element) -> Dict[str, Any]:
        """
        Extract metadata from Unstructured element.
        
        Args:
            element: Unstructured element
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        try:
            if hasattr(element, 'metadata') and element.metadata:
                elem_metadata = element.metadata
                
                # Extract common metadata fields
                fields_to_extract = [
                    'filename', 'file_directory', 'last_modified',
                    'page_number', 'languages', 'emphasized_text_contents',
                    'emphasized_text_tags', 'text_as_html', 'link_urls',
                    'link_texts', 'sent_from', 'sent_to', 'subject'
                ]
                
                for field in fields_to_extract:
                    value = getattr(elem_metadata, field, None)
                    if value is not None:
                        metadata[field] = value
        
        except Exception as e:
            logging.warning(f"Failed to extract element metadata: {e}")
        
        return metadata
    
    def _post_process_blocks(self, blocks: List[Block]) -> List[Block]:
        """
        Post-process blocks to improve classification and structure.
        
        Args:
            blocks: List of blocks to process
            
        Returns:
            List of processed blocks
        """
        processed_blocks = []
        
        for block in blocks:
            # Improve block type classification based on content
            if block["type"] == "paragraph":
                improved_type = self._classify_text_content(block["text"])
                if improved_type != "paragraph":
                    block["type"] = improved_type
            
            # Add text analysis metadata
            block["meta"]["text_length"] = len(block["text"])
            block["meta"]["word_count"] = len(block["text"].split())
            
            # Detect potential construction-specific content
            if self._is_construction_content(block["text"]):
                block["meta"]["construction_content"] = True
            
            processed_blocks.append(block)
        
        return processed_blocks
    
    def _classify_text_content(self, text: str) -> str:
        """
        Classify text content based on patterns.
        
        Args:
            text: Text content to classify
            
        Returns:
            Block type string
        """
        text_lower = text.lower().strip()
        
        # Title block patterns (construction drawings)
        title_patterns = [
            r'sheet\s+\d+', r'drawing\s+no', r'project\s+no',
            r'scale\s*:', r'date\s*:', r'drawn\s+by', r'checked\s+by'
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, text_lower):
                return "titleblock"
        
        # List patterns
        if re.match(r'^\s*[\d\w][\.\)]\s', text) or text.count('\n') > 2:
            return "list"
        
        return "paragraph"
    
    def _is_construction_content(self, text: str) -> bool:
        """
        Detect if text contains construction-specific content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if construction content is detected
        """
        construction_keywords = [
            'concrete', 'steel', 'hvac', 'electrical', 'plumbing',
            'structural', 'architectural', 'mechanical', 'specification',
            'drawing', 'plan', 'elevation', 'section', 'detail',
            'contractor', 'subcontractor', 'bid', 'estimate'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in construction_keywords)