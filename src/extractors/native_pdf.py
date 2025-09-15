"""
Native PDF extractor using PyMuPDF, pdfplumber, and Camelot for comprehensive PDF parsing.
"""
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import re

from src.extractors.base import BaseExtractor, ExtractorError
from src.models.types import PageParse, Block, Span
from src.utils.bbox import normalize_bbox, merge_bboxes
from src.utils.pdf_utils import get_pdf_page_count, get_pdf_page_dimensions, validate_pdf_file

# Optional dependencies with graceful fallbacks
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Native PDF extraction will be disabled.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available. Table detection will be limited.")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available. Complex table extraction will be disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. Table processing will be limited.")


class NativePDFExtractor(BaseExtractor):
    """
    Native PDF extractor using multiple libraries for comprehensive extraction.
    
    Uses PyMuPDF for text extraction, pdfplumber for table detection,
    and Camelot as fallback for complex table structures.
    """
    
    def __init__(self, use_camelot: bool = True, min_confidence: float = 0.8):
        """
        Initialize the native PDF extractor.
        
        Args:
            use_camelot: Whether to use Camelot for complex table extraction
            min_confidence: Minimum confidence threshold for text spans
        """
        self.use_camelot = use_camelot and CAMELOT_AVAILABLE
        self.min_confidence = min_confidence
        
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for NativePDFExtractor")
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given PDF file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is a readable PDF, False otherwise
        """
        if not PYMUPDF_AVAILABLE:
            return False
        
        # Check file extension
        path = Path(file_path)
        if path.suffix.lower() != '.pdf':
            return False
        
        # Validate PDF file
        return validate_pdf_file(file_path)
    
    def get_page_count(self, file_path: Union[str, Path]) -> int:
        """
        Get the total number of pages in the PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of pages in the PDF
        """
        try:
            return get_pdf_page_count(file_path)
        except Exception as e:
            raise ExtractorError(f"Failed to get page count: {e}", self.get_extractor_name(), str(file_path))
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['pdf']
    
    def parse_page(self, file_path: Union[str, Path], page_no: int) -> PageParse:
        """
        Parse a specific page from the PDF using multiple extraction methods.
        
        Args:
            file_path: Path to the PDF file
            page_no: Page number to parse (0-indexed)
            
        Returns:
            PageParse object containing extracted content and metadata
            
        Raises:
            ExtractorError: If parsing fails
        """
        try:
            # Get page dimensions
            width, height = get_pdf_page_dimensions(file_path, page_no)
            
            # Extract text and structure using PyMuPDF
            pymupdf_blocks = self._extract_with_pymupdf(file_path, page_no, width, height)
            
            # Extract tables using pdfplumber
            table_blocks = []
            if PDFPLUMBER_AVAILABLE:
                table_blocks = self._extract_tables_with_pdfplumber(file_path, page_no, width, height)
            
            # Try Camelot for complex tables if pdfplumber didn't find good tables
            if self.use_camelot and len(table_blocks) == 0:
                camelot_tables = self._extract_tables_with_camelot(file_path, page_no, width, height)
                table_blocks.extend(camelot_tables)
            
            # Merge text and table blocks, removing overlaps
            all_blocks = self._merge_blocks(pymupdf_blocks, table_blocks, width, height)
            
            # Detect and classify special block types
            classified_blocks = self._classify_blocks(all_blocks)
            
            return PageParse(
                page_no=page_no,
                width=int(width),
                height=int(height),
                blocks=classified_blocks,
                artifacts_removed=[]
            )
            
        except Exception as e:
            raise ExtractorError(f"Failed to parse page {page_no}: {e}", self.get_extractor_name(), str(file_path), page_no)
    
    def _extract_with_pymupdf(self, file_path: Union[str, Path], page_no: int, width: float, height: float) -> List[Block]:
        """
        Extract text and basic structure using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            page_no: Page number (0-indexed)
            width: Page width
            height: Page height
            
        Returns:
            List of Block objects from PyMuPDF extraction
        """
        blocks = []
        
        try:
            doc = fitz.open(str(file_path))
            page = doc[page_no]
            
            # Get text blocks with formatting information
            text_dict = page.get_text("dict")
            
            for block_data in text_dict.get("blocks", []):
                if "lines" not in block_data:
                    continue  # Skip image blocks
                
                # Extract text and spans from block
                block_text = ""
                spans = []
                block_bbox = None
                
                for line in block_data["lines"]:
                    line_text = ""
                    line_spans = []
                    
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue
                        
                        # Create span with normalized bbox
                        span_bbox = span.get("bbox", [0, 0, 0, 0])
                        normalized_bbox = normalize_bbox(span_bbox, width, height)
                        
                        span_obj = Span(
                            text=span_text,
                            bbox=normalized_bbox,
                            rot=span.get("flags", 0) & 2**1,  # Rotation flag
                            conf=1.0  # PyMuPDF doesn't provide confidence
                        )
                        
                        line_spans.append(span_obj)
                        line_text += span_text + " "
                    
                    if line_text.strip():
                        block_text += line_text.strip() + "\n"
                        spans.extend(line_spans)
                
                if block_text.strip() and spans:
                    # Calculate block bbox from spans
                    span_bboxes = [span["bbox"] for span in spans]
                    if span_bboxes:
                        # Convert back to absolute coordinates for merging
                        abs_bboxes = []
                        for bbox in span_bboxes:
                            abs_bbox = [
                                bbox[0] * width, bbox[1] * height,
                                bbox[2] * width, bbox[3] * height
                            ]
                            abs_bboxes.append(abs_bbox)
                        
                        merged_bbox = merge_bboxes(abs_bboxes)
                        if merged_bbox:
                            block_bbox = normalize_bbox(merged_bbox, width, height)
                    
                    if not block_bbox:
                        block_bbox = [0.0, 0.0, 1.0, 1.0]  # Fallback
                    
                    # Determine block type based on formatting
                    block_type = self._determine_block_type(block_text, spans)
                    
                    block = Block(
                        type=block_type,
                        text=block_text.strip(),
                        html=None,
                        bbox=block_bbox,
                        spans=spans,
                        meta={"source": "pymupdf", "block_id": len(blocks)}
                    )
                    
                    blocks.append(block)
            
            doc.close()
            
        except Exception as e:
            logging.error(f"PyMuPDF extraction failed for page {page_no}: {e}")
        
        return blocks
    
    def _extract_tables_with_pdfplumber(self, file_path: Union[str, Path], page_no: int, width: float, height: float) -> List[Block]:
        """
        Extract tables using pdfplumber for better table structure detection.
        
        Args:
            file_path: Path to the PDF file
            page_no: Page number (0-indexed)
            width: Page width
            height: Page height
            
        Returns:
            List of table Block objects
        """
        table_blocks = []
        
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                if page_no >= len(pdf.pages):
                    return table_blocks
                
                page = pdf.pages[page_no]
                tables = page.find_tables()
                
                for i, table in enumerate(tables):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        if not table_data or len(table_data) < 2:  # Need at least header + 1 row
                            continue
                        
                        # Convert to HTML
                        html_table = self._table_data_to_html(table_data)
                        
                        # Create text representation
                        text_table = self._table_data_to_text(table_data)
                        
                        # Get table bbox
                        table_bbox = table.bbox
                        if table_bbox:
                            normalized_bbox = normalize_bbox(list(table_bbox), width, height)
                        else:
                            normalized_bbox = [0.0, 0.0, 1.0, 1.0]
                        
                        # Create spans for table cells (simplified)
                        spans = [Span(
                            text=text_table,
                            bbox=normalized_bbox,
                            rot=0.0,
                            conf=0.9
                        )]
                        
                        table_block = Block(
                            type="table",
                            text=text_table,
                            html=html_table,
                            bbox=normalized_bbox,
                            spans=spans,
                            meta={
                                "source": "pdfplumber",
                                "table_id": i,
                                "rows": len(table_data),
                                "cols": len(table_data[0]) if table_data else 0
                            }
                        )
                        
                        table_blocks.append(table_block)
                        
                    except Exception as e:
                        logging.warning(f"Failed to extract table {i} from page {page_no}: {e}")
                        continue
        
        except Exception as e:
            logging.error(f"pdfplumber table extraction failed for page {page_no}: {e}")
        
        return table_blocks
    
    def _extract_tables_with_camelot(self, file_path: Union[str, Path], page_no: int, width: float, height: float) -> List[Block]:
        """
        Extract complex tables using Camelot as fallback.
        
        Args:
            file_path: Path to the PDF file
            page_no: Page number (0-indexed)
            width: Page width
            height: Page height
            
        Returns:
            List of table Block objects from Camelot
        """
        table_blocks = []
        
        if not CAMELOT_AVAILABLE or not PANDAS_AVAILABLE:
            return table_blocks
        
        try:
            # Camelot uses 1-indexed pages
            camelot_page = str(page_no + 1)
            
            # Try lattice method first (for tables with lines)
            try:
                tables = camelot.read_pdf(str(file_path), pages=camelot_page, flavor='lattice')
                if len(tables) > 0:
                    table_blocks.extend(self._process_camelot_tables(tables, width, height, "lattice"))
            except Exception as e:
                logging.debug(f"Camelot lattice method failed: {e}")
            
            # Try stream method if lattice didn't work well
            if len(table_blocks) == 0:
                try:
                    tables = camelot.read_pdf(str(file_path), pages=camelot_page, flavor='stream')
                    if len(tables) > 0:
                        table_blocks.extend(self._process_camelot_tables(tables, width, height, "stream"))
                except Exception as e:
                    logging.debug(f"Camelot stream method failed: {e}")
        
        except Exception as e:
            logging.error(f"Camelot table extraction failed for page {page_no}: {e}")
        
        return table_blocks
    
    def _process_camelot_tables(self, tables, width: float, height: float, method: str) -> List[Block]:
        """
        Process Camelot table results into Block objects.
        
        Args:
            tables: Camelot TableList object
            width: Page width
            height: Page height
            method: Camelot method used ("lattice" or "stream")
            
        Returns:
            List of table Block objects
        """
        table_blocks = []
        
        for i, table in enumerate(tables):
            try:
                # Check table quality
                if hasattr(table, 'accuracy') and table.accuracy < 0.5:
                    continue  # Skip low-quality tables
                
                # Get table data as DataFrame
                df = table.df
                if df.empty or len(df) < 2:
                    continue
                
                # Convert DataFrame to list of lists
                table_data = [df.columns.tolist()] + df.values.tolist()
                
                # Convert to HTML and text
                html_table = self._table_data_to_html(table_data)
                text_table = self._table_data_to_text(table_data)
                
                # Estimate bbox (Camelot doesn't always provide exact coordinates)
                # Use table parsing area if available
                if hasattr(table, '_bbox') and table._bbox:
                    bbox = table._bbox
                    normalized_bbox = normalize_bbox(bbox, width, height)
                else:
                    # Fallback to estimated position
                    normalized_bbox = [0.1, 0.1, 0.9, 0.9]
                
                # Create spans
                spans = [Span(
                    text=text_table,
                    bbox=normalized_bbox,
                    rot=0.0,
                    conf=getattr(table, 'accuracy', 0.8)
                )]
                
                table_block = Block(
                    type="table",
                    text=text_table,
                    html=html_table,
                    bbox=normalized_bbox,
                    spans=spans,
                    meta={
                        "source": f"camelot_{method}",
                        "table_id": i,
                        "rows": len(table_data),
                        "cols": len(table_data[0]) if table_data else 0,
                        "accuracy": getattr(table, 'accuracy', None)
                    }
                )
                
                table_blocks.append(table_block)
                
            except Exception as e:
                logging.warning(f"Failed to process Camelot table {i}: {e}")
                continue
        
        return table_blocks
    
    def _table_data_to_html(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to HTML format.
        
        Args:
            table_data: List of rows, each row is a list of cell values
            
        Returns:
            HTML table string
        """
        if not table_data:
            return ""
        
        html = "<table>\n"
        
        # Add header row
        if len(table_data) > 0:
            html += "  <thead>\n    <tr>\n"
            for cell in table_data[0]:
                cell_text = str(cell).strip() if cell else ""
                html += f"      <th>{self._escape_html(cell_text)}</th>\n"
            html += "    </tr>\n  </thead>\n"
        
        # Add body rows
        if len(table_data) > 1:
            html += "  <tbody>\n"
            for row in table_data[1:]:
                html += "    <tr>\n"
                for cell in row:
                    cell_text = str(cell).strip() if cell else ""
                    html += f"      <td>{self._escape_html(cell_text)}</td>\n"
                html += "    </tr>\n"
            html += "  </tbody>\n"
        
        html += "</table>"
        return html
    
    def _table_data_to_text(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to plain text format.
        
        Args:
            table_data: List of rows, each row is a list of cell values
            
        Returns:
            Plain text table representation
        """
        if not table_data:
            return ""
        
        # Calculate column widths
        col_widths = []
        for row in table_data:
            for i, cell in enumerate(row):
                cell_text = str(cell).strip() if cell else ""
                if i >= len(col_widths):
                    col_widths.append(len(cell_text))
                else:
                    col_widths[i] = max(col_widths[i], len(cell_text))
        
        # Format table
        text_lines = []
        for i, row in enumerate(table_data):
            formatted_cells = []
            for j, cell in enumerate(row):
                cell_text = str(cell).strip() if cell else ""
                width = col_widths[j] if j < len(col_widths) else 10
                formatted_cells.append(cell_text.ljust(width))
            
            text_lines.append(" | ".join(formatted_cells))
            
            # Add separator after header
            if i == 0 and len(table_data) > 1:
                separator = " | ".join(["-" * width for width in col_widths])
                text_lines.append(separator)
        
        return "\n".join(text_lines)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    def _merge_blocks(self, text_blocks: List[Block], table_blocks: List[Block], width: float, height: float) -> List[Block]:
        """
        Merge text and table blocks, removing overlaps.
        
        Args:
            text_blocks: Blocks from text extraction
            table_blocks: Blocks from table extraction
            width: Page width
            height: Page height
            
        Returns:
            Merged list of blocks with overlaps removed
        """
        from src.utils.bbox import bbox_iou, denormalize_bbox
        
        # Convert normalized bboxes to absolute for IoU calculation
        def get_abs_bbox(block):
            norm_bbox = block["bbox"]
            return denormalize_bbox(norm_bbox, width, height)
        
        # Remove text blocks that significantly overlap with tables
        filtered_text_blocks = []
        for text_block in text_blocks:
            text_bbox = get_abs_bbox(text_block)
            
            overlaps_table = False
            for table_block in table_blocks:
                table_bbox = get_abs_bbox(table_block)
                iou = bbox_iou(text_bbox, table_bbox)
                
                # If text block significantly overlaps with table, remove it
                if iou > 0.5:
                    overlaps_table = True
                    break
            
            if not overlaps_table:
                filtered_text_blocks.append(text_block)
        
        # Combine filtered text blocks with table blocks
        all_blocks = filtered_text_blocks + table_blocks
        
        # Sort blocks by reading order (top to bottom, left to right)
        all_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        
        return all_blocks
    
    def _determine_block_type(self, text: str, spans: List[Span]) -> str:
        """
        Determine block type based on text content and formatting.
        
        Args:
            text: Block text content
            spans: List of text spans in the block
            
        Returns:
            Block type string
        """
        text_lower = text.lower().strip()
        
        # Check for title block patterns (common in construction drawings)
        title_block_patterns = [
            r'sheet\s+\d+',
            r'drawing\s+no',
            r'project\s+no',
            r'scale\s*:',
            r'date\s*:',
            r'drawn\s+by',
            r'checked\s+by',
            r'revision'
        ]
        
        for pattern in title_block_patterns:
            if re.search(pattern, text_lower):
                return "titleblock"
        
        # Check for headings (short text, potentially larger font)
        if len(text.strip()) < 100 and len(spans) > 0:
            # Simple heuristic: if text is short and might be a heading
            if any(keyword in text_lower for keyword in ['section', 'part', 'chapter', 'division']):
                return "heading"
        
        # Check for list patterns
        list_patterns = [
            r'^\s*[\d\w]\.\s',  # 1. or a.
            r'^\s*[\d\w]\)\s',  # 1) or a)
            r'^\s*[•·▪▫]\s',    # Bullet points
            r'^\s*[-*]\s'       # Dash or asterisk bullets
        ]
        
        lines = text.split('\n')
        list_line_count = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line.strip()):
                    list_line_count += 1
                    break
        
        if list_line_count >= 2 or (len(lines) >= 2 and list_line_count / len(lines) > 0.5):
            return "list"
        
        # Default to paragraph
        return "paragraph"
    
    def _classify_blocks(self, blocks: List[Block]) -> List[Block]:
        """
        Post-process blocks to improve classification.
        
        Args:
            blocks: List of blocks to classify
            
        Returns:
            List of blocks with improved classification
        """
        # Additional classification logic can be added here
        # For now, return blocks as-is since basic classification is done during extraction
        return blocks