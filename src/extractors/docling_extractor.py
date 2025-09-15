"""
Docling extractor that converts Docling DOM to PageParse format.
"""
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import re

from src.extractors.base import BaseExtractor, ExtractorError
from src.models.types import PageParse, Block, Span
from src.utils.bbox import normalize_bbox, merge_bboxes

# Optional dependency with graceful fallback
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available. Docling extraction will be disabled.")


class DoclingExtractor(BaseExtractor):
    """
    Docling extractor that converts Docling DOM elements to PageParse format.
    
    Handles PDFs, DOCX, and other formats supported by Docling with advanced
    layout parsing and structure preservation.
    """
    
    def __init__(self, 
                 use_ocr: bool = True,
                 extract_images: bool = False,
                 extract_tables: bool = True):
        """
        Initialize the Docling extractor.
        
        Args:
            use_ocr: Whether to use OCR for scanned documents
            extract_images: Whether to extract embedded images
            extract_tables: Whether to extract table structures
        """
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required for DoclingExtractor")
        
        self.use_ocr = use_ocr
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        
        # Initialize document converter with simple configuration
        try:
            # Use simple DocumentConverter without complex pipeline options
            self.converter = DocumentConverter()
            logging.info("Initialized Docling converter with default settings")
        except Exception as e:
            raise ImportError(f"Failed to initialize Docling converter: {e}")
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file format is supported by Docling, False otherwise
        """
        if not DOCLING_AVAILABLE:
            return False
        
        path = Path(file_path)
        supported_extensions = ['.pdf', '.docx', '.pptx', '.html', '.md']
        return path.suffix.lower() in supported_extensions
    
    def get_page_count(self, file_path: Union[str, Path]) -> int:
        """
        Get the total number of pages in the document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of pages in the document
        """
        try:
            # Convert document to get page count
            result = self.converter.convert(str(file_path))
            return len(result.document.pages)
        except Exception as e:
            logging.error(f"Failed to get page count from Docling: {e}")
            return 1  # Fallback
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['pdf', 'docx', 'pptx', 'html', 'md']
    
    def parse_page(self, file_path: Union[str, Path], page_no: int) -> PageParse:
        """
        Parse a specific page using Docling.
        
        Args:
            file_path: Path to the document file
            page_no: Page number to parse (0-indexed)
            
        Returns:
            PageParse object containing extracted content and metadata
            
        Raises:
            ExtractorError: If parsing fails
        """
        try:
            # Convert entire document
            result = self.converter.convert(str(file_path))
            document = result.document
            
            # Check if page exists
            if page_no >= len(document.pages):
                raise ExtractorError(f"Page {page_no} does not exist", self.get_extractor_name(), str(file_path), page_no)
            
            page = document.pages[page_no]
            
            # Get page dimensions
            width = float(page.size.width) if page.size else 612.0  # Default letter width
            height = float(page.size.height) if page.size else 792.0  # Default letter height
            
            # Convert Docling elements to blocks
            blocks = []
            
            # Get elements from the page and convert to blocks
            try:
                elements = []
                
                if hasattr(page, 'elements'):
                    elements = page.elements
                elif hasattr(page, 'items'):
                    elements = page.items
                elif hasattr(page, 'body'):
                    # Try Docling document body structure
                    if hasattr(page.body, 'elements'):
                        elements = page.body.elements
                
                if not elements:
                    # Create fallback from page text
                    page_text = str(page) if page else "No content extracted"
                    if page_text.strip():
                        blocks = [{
                            "type": "paragraph",
                            "text": page_text.strip(),
                            "html": None,
                            "bbox": [0, 0, width, height],
                            "spans": [{
                                "text": page_text.strip(),
                                "bbox": [0, 0, width, height],
                                "rot": 0,
                                "conf": 1.0
                            }],
                            "meta": {}
                        }]
                else:
                    # Convert elements to blocks
                    for element in elements:
                        try:
                            block = self._convert_element_to_block(element, document, width, height)
                            if block:
                                blocks.append(block)
                        except Exception as e:
                            logging.warning(f"Failed to convert element: {e}")
                            continue
                        
            except Exception as e:
                logging.error(f"Failed to extract elements from Docling page: {e}")
                # Create fallback block
                page_text = str(page) if page else "No content extracted"
                blocks = [{
                    "type": "paragraph",
                    "text": page_text,
                    "html": None,
                    "bbox": [0, 0, width, height],
                    "spans": [{"text": page_text, "bbox": [0, 0, width, height], "rot": 0, "conf": 0.5}],
                    "meta": {}
                }]
            
            # Post-process blocks
            processed_blocks = blocks  # Skip post-processing for now
            
            return PageParse(
                page_no=page_no,
                width=int(width),
                height=int(height),
                blocks=processed_blocks,
                artifacts_removed=[]
            )
            
        except Exception as e:
            raise ExtractorError(f"Failed to parse page {page_no}: {e}", self.get_extractor_name(), str(file_path), page_no)
    
    def _convert_docling_elements(self, page, document, width: float, height: float) -> List[Block]:
        """
        Convert Docling page elements to Block objects.
        
        Args:
            page: Docling page object
            document: Docling document object
            width: Page width
            height: Page height
            
        Returns:
            List of Block objects
        """
        blocks = []
        
        # Process each element in the page
        for element in page.elements:
            try:
                block = self._convert_element_to_block(element, document, width, height)
                if block:
                    blocks.append(block)
            except Exception as e:
                logging.warning(f"Failed to convert Docling element: {e}")
                continue
        
        return blocks
    
    def _convert_element_to_block(self, element, document, width: float, height: float) -> Optional[Block]:
        """
        Convert a single Docling element to a Block.
        
        Args:
            element: Docling element
            document: Docling document object
            width: Page width
            height: Page height
            
        Returns:
            Block object or None if conversion fails
        """
        # Get element text
        text = self._get_element_text(element, document)
        if not text or not text.strip():
            return None
        
        # Get element bbox
        bbox = self._get_element_bbox(element, width, height)
        
        # Determine block type
        block_type = self._get_docling_block_type(element)
        
        # Create spans
        spans = self._create_spans_from_element(element, document, width, height)
        
        # Handle tables specially
        html_content = None
        if block_type == "table" and hasattr(element, 'table'):
            html_content = self._convert_table_to_html(element, document)
        
        # Create block
        block = Block(
            type=block_type,
            text=text.strip(),
            html=html_content,
            bbox=bbox,
            spans=spans,
            meta={
                "source": "docling",
                "element_type": element.__class__.__name__,
                "element_id": getattr(element, 'id', None)
            }
        )
        
        return block
    
    def _get_element_text(self, element, document) -> str:
        """
        Extract text content from Docling element.
        
        Args:
            element: Docling element
            document: Docling document object
            
        Returns:
            Text content of the element
        """
        try:
            # Try to get text directly from element
            if hasattr(element, 'text') and element.text:
                return element.text
            
            # For table elements, extract cell text
            if hasattr(element, 'table') and element.table:
                return self._extract_table_text(element, document)
            
            # Try to get text from document export
            if hasattr(document, 'export_to_text'):
                # This is a simplified approach - in practice, you'd need to
                # extract text for the specific element
                pass
            
            return ""
            
        except Exception as e:
            logging.warning(f"Failed to extract text from Docling element: {e}")
            return ""
    
    def _extract_table_text(self, table_element, document) -> str:
        """
        Extract text from table element.
        
        Args:
            table_element: Docling table element
            document: Docling document object
            
        Returns:
            Plain text representation of table
        """
        try:
            if not hasattr(table_element, 'table') or not table_element.table:
                return ""
            
            table = table_element.table
            text_rows = []
            
            # Extract text from table cells
            for row in table.table_cells:
                row_texts = []
                for cell in row:
                    cell_text = getattr(cell, 'text', '') or ''
                    row_texts.append(cell_text.strip())
                text_rows.append(" | ".join(row_texts))
            
            return "\n".join(text_rows)
            
        except Exception as e:
            logging.warning(f"Failed to extract table text: {e}")
            return "Table content (extraction failed)"
    
    def _get_element_bbox(self, element, width: float, height: float) -> List[float]:
        """
        Get normalized bounding box for Docling element.
        
        Args:
            element: Docling element
            width: Page width
            height: Page height
            
        Returns:
            Normalized bounding box [x0, y0, x1, y1]
        """
        try:
            if hasattr(element, 'bbox') and element.bbox:
                bbox = element.bbox
                # Convert to list if needed
                if hasattr(bbox, 'l'):  # Docling BBox object
                    abs_bbox = [bbox.l, bbox.t, bbox.r, bbox.b]
                else:
                    abs_bbox = list(bbox)
                
                return normalize_bbox(abs_bbox, width, height)
            
            # Fallback bbox
            return [0.0, 0.0, 1.0, 1.0]
            
        except Exception as e:
            logging.warning(f"Failed to get element bbox: {e}")
            return [0.0, 0.0, 1.0, 1.0]
    
    def _get_docling_block_type(self, element) -> str:
        """
        Determine block type from Docling element.
        
        Args:
            element: Docling element
            
        Returns:
            Block type string
        """
        element_type = element.__class__.__name__.lower()
        
        # Map Docling element types to our block types
        type_mapping = {
            'title': 'heading',
            'sectionheader': 'heading',
            'paragraph': 'paragraph',
            'text': 'paragraph',
            'table': 'table',
            'list': 'list',
            'listitem': 'list',
            'figure': 'figure',
            'caption': 'caption',
            'footnote': 'paragraph',
            'header': 'heading',
            'footer': 'paragraph'
        }
        
        # Check for specific patterns in element type
        for pattern, block_type in type_mapping.items():
            if pattern in element_type:
                return block_type
        
        # Default to paragraph
        return 'paragraph'
    
    def _create_spans_from_element(self, element, document, width: float, height: float) -> List[Span]:
        """
        Create text spans from Docling element.
        
        Args:
            element: Docling element
            document: Docling document object
            width: Page width
            height: Page height
            
        Returns:
            List of Span objects
        """
        spans = []
        
        try:
            # Get element text and bbox
            text = self._get_element_text(element, document)
            bbox = self._get_element_bbox(element, width, height)
            
            if text and text.strip():
                span = Span(
                    text=text.strip(),
                    bbox=bbox,
                    rot=0.0,  # Docling handles rotation internally
                    conf=1.0  # Docling doesn't provide confidence scores
                )
                spans.append(span)
        
        except Exception as e:
            logging.warning(f"Failed to create spans from element: {e}")
        
        return spans
    
    def _convert_table_to_html(self, table_element, document) -> str:
        """
        Convert Docling table element to HTML.
        
        Args:
            table_element: Docling table element
            document: Docling document object
            
        Returns:
            HTML table string
        """
        try:
            if not hasattr(table_element, 'table') or not table_element.table:
                return ""
            
            table = table_element.table
            html = "<table>\n"
            
            # Process table rows
            for i, row in enumerate(table.table_cells):
                tag = "th" if i == 0 else "td"  # First row as header
                html += "  <tr>\n"
                
                for cell in row:
                    cell_text = getattr(cell, 'text', '') or ''
                    html += f"    <{tag}>{self._escape_html(cell_text.strip())}</{tag}>\n"
                
                html += "  </tr>\n"
            
            html += "</table>"
            return html
            
        except Exception as e:
            logging.warning(f"Failed to convert table to HTML: {e}")
            return ""
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
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
            # Improve block type classification
            if block["type"] == "paragraph":
                improved_type = self._classify_paragraph_block(block["text"])
                if improved_type != "paragraph":
                    block["type"] = improved_type
            
            # Add additional metadata
            block["meta"]["text_length"] = len(block["text"])
            block["meta"]["span_count"] = len(block["spans"])
            
            processed_blocks.append(block)
        
        return processed_blocks
    
    def _classify_paragraph_block(self, text: str) -> str:
        """
        Improve classification of paragraph blocks based on content.
        
        Args:
            text: Block text content
            
        Returns:
            Improved block type
        """
        text_lower = text.lower().strip()
        
        # Check for title block patterns
        title_patterns = [
            r'sheet\s+\d+', r'drawing\s+no', r'project\s+no',
            r'scale\s*:', r'date\s*:', r'drawn\s+by', r'checked\s+by'
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, text_lower):
                return "titleblock"
        
        # Check for headings (short text with section indicators)
        if len(text.strip()) < 100:
            heading_indicators = ['section', 'part', 'chapter', 'division', 'article']
            if any(indicator in text_lower for indicator in heading_indicators):
                return "heading"
        
        # Check for list patterns
        list_patterns = [
            r'^\s*[\d\w][\.\)]\s',  # 1. or a)
            r'^\s*[•·▪▫-]\s'        # Bullet points
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text):
                return "list"
        
        return "paragraph"