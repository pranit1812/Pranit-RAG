"""
OCR extractor using PaddleOCR and PP-Structure for text detection, recognition, and table structure analysis.
"""
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import io

from src.extractors.base import BaseExtractor, ExtractorError
from src.models.types import PageParse, Block, Span
from src.utils.bbox import normalize_bbox, merge_bboxes
from src.utils.pdf_utils import render_pdf_page_to_image, get_pdf_page_count, get_pdf_page_dimensions

# Optional dependencies with graceful fallbacks
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. OCR extraction will be disabled.")

try:
    # Try different import methods for PP-Structure
    try:
        from paddleocr import PPStructure
        PPSTRUCTURE_AVAILABLE = True
    except ImportError:
        # Try alternative import path
        from paddleocr.ppstructure import PPStructure
        PPSTRUCTURE_AVAILABLE = True
except ImportError:
    PPSTRUCTURE_AVAILABLE = False
    logging.warning("PP-Structure not available. Table structure recognition will be disabled.")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Text clustering will be disabled.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Image processing will be limited.")


class OCRExtractor(BaseExtractor):
    """
    OCR extractor using PaddleOCR for text detection and recognition,
    with PP-Structure for table structure recognition and DBSCAN for text clustering.
    """
    
    def __init__(
        self, 
        languages: List[str] = None,
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        table_model: str = "TableMaster",
        cluster_eps: float = 20.0,
        cluster_min_samples: int = 2
    ):
        """
        Initialize the OCR extractor.
        
        Args:
            languages: List of languages for OCR (default: ['en'])
            use_gpu: Whether to use GPU acceleration
            min_confidence: Minimum confidence threshold for text detection
            table_model: Table structure model ("TableMaster" or "SLANet")
            cluster_eps: DBSCAN epsilon parameter for text clustering
            cluster_min_samples: DBSCAN minimum samples parameter
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is required for OCRExtractor")
        
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.table_model = table_model
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        
        # Initialize PaddleOCR
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.languages[0] if self.languages else 'en',
                use_gpu=self.use_gpu,
                show_log=False
            )
        except Exception as e:
            raise ImportError(f"Failed to initialize PaddleOCR: {e}")
        
        # Initialize PP-Structure if available
        self.pp_structure = None
        if PPSTRUCTURE_AVAILABLE:
            try:
                self.pp_structure = PPStructure(
                    table_model=self.table_model,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            except Exception as e:
                logging.warning(f"Failed to initialize PP-Structure: {e}")
    
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is supported (PDF or image), False otherwise
        """
        if not PADDLEOCR_AVAILABLE:
            return False
        
        path = Path(file_path)
        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        return path.suffix.lower() in supported_extensions
    
    def get_page_count(self, file_path: Union[str, Path]) -> int:
        """
        Get the total number of pages in the document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of pages (1 for images, actual count for PDFs)
        """
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            try:
                return get_pdf_page_count(file_path)
            except Exception:
                return 1
        else:
            return 1  # Single page for images
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']
    
    def parse_page(self, file_path: Union[str, Path], page_no: int) -> PageParse:
        """
        Parse a specific page using OCR and structure recognition.
        
        Args:
            file_path: Path to the document file
            page_no: Page number to parse (0-indexed)
            
        Returns:
            PageParse object containing extracted content and metadata
            
        Raises:
            ExtractorError: If parsing fails
        """
        try:
            # Get image data
            image_data, width, height = self._get_page_image(file_path, page_no)
            
            # Extract text using PaddleOCR
            ocr_blocks = self._extract_text_with_ocr(image_data, width, height)
            
            # Extract tables using PP-Structure if available
            table_blocks = []
            if self.pp_structure:
                table_blocks = self._extract_tables_with_ppstructure(image_data, width, height)
            
            # Merge OCR and table results
            all_blocks = self._merge_ocr_and_tables(ocr_blocks, table_blocks, width, height)
            
            # Apply text clustering for drawing regions if enabled
            if SKLEARN_AVAILABLE and self._is_likely_drawing(all_blocks):
                all_blocks = self._cluster_drawing_text(all_blocks, width, height)
            
            # Classify blocks and add confidence flags
            classified_blocks = self._classify_and_flag_blocks(all_blocks)
            
            return PageParse(
                page_no=page_no,
                width=int(width),
                height=int(height),
                blocks=classified_blocks,
                artifacts_removed=[]
            )
            
        except Exception as e:
            raise ExtractorError(f"Failed to parse page {page_no}: {e}", self.get_extractor_name(), str(file_path), page_no)
    
    def _get_page_image(self, file_path: Union[str, Path], page_no: int) -> Tuple[np.ndarray, int, int]:
        """
        Get image data for the specified page.
        
        Args:
            file_path: Path to the document file
            page_no: Page number (0-indexed)
            
        Returns:
            Tuple of (image_array, width, height)
        """
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            # Render PDF page to image
            image_bytes = render_pdf_page_to_image(file_path, page_no, scale=2.0)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Load image file directly
            if page_no > 0:
                raise ValueError(f"Image files only have one page, requested page {page_no}")
            image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        width, height = image.size
        
        return image_array, width, height
    
    def _extract_text_with_ocr(self, image: np.ndarray, width: int, height: int) -> List[Block]:
        """
        Extract text using PaddleOCR.
        
        Args:
            image: Image array
            width: Image width
            height: Image height
            
        Returns:
            List of text blocks from OCR
        """
        blocks = []
        
        try:
            # Run OCR
            ocr_results = self.ocr.ocr(image, cls=True)
            
            if not ocr_results or not ocr_results[0]:
                return blocks
            
            # Process OCR results
            for i, line_result in enumerate(ocr_results[0]):
                if not line_result:
                    continue
                
                bbox_points, (text, confidence) = line_result
                
                # Skip low-confidence results
                if confidence < self.min_confidence:
                    continue
                
                # Convert bbox points to normalized coordinates
                bbox_points = np.array(bbox_points)
                x_coords = bbox_points[:, 0]
                y_coords = bbox_points[:, 1]
                
                x0, y0 = float(np.min(x_coords)), float(np.min(y_coords))
                x1, y1 = float(np.max(x_coords)), float(np.max(y_coords))
                
                normalized_bbox = normalize_bbox([x0, y0, x1, y1], width, height)
                
                # Create span
                span = Span(
                    text=text,
                    bbox=normalized_bbox,
                    rot=0.0,  # PaddleOCR handles rotation internally
                    conf=float(confidence)
                )
                
                # Create block
                block = Block(
                    type="paragraph",  # Will be refined later
                    text=text,
                    html=None,
                    bbox=normalized_bbox,
                    spans=[span],
                    meta={
                        "source": "paddleocr",
                        "confidence": confidence,
                        "line_id": i
                    }
                )
                
                blocks.append(block)
        
        except Exception as e:
            logging.error(f"PaddleOCR text extraction failed: {e}")
        
        return blocks
    
    def _extract_tables_with_ppstructure(self, image: np.ndarray, width: int, height: int) -> List[Block]:
        """
        Extract tables using PP-Structure.
        
        Args:
            image: Image array
            width: Image width
            height: Image height
            
        Returns:
            List of table blocks
        """
        table_blocks = []
        
        try:
            # Run PP-Structure
            structure_results = self.pp_structure(image)
            
            for i, result in enumerate(structure_results):
                if result['type'] != 'table':
                    continue
                
                # Get table bbox
                bbox = result.get('bbox', [0, 0, width, height])
                normalized_bbox = normalize_bbox(bbox, width, height)
                
                # Extract table structure if available
                table_html = ""
                table_text = ""
                
                if 'res' in result and result['res']:
                    # PP-Structure provides structured table data
                    table_data = result['res']
                    table_html = self._structure_to_html(table_data)
                    table_text = self._structure_to_text(table_data)
                else:
                    # Fallback: extract text from table region
                    table_text = f"Table {i+1} (structure not available)"
                
                # Create span for table
                span = Span(
                    text=table_text,
                    bbox=normalized_bbox,
                    rot=0.0,
                    conf=result.get('confidence', 0.8)
                )
                
                # Create table block
                table_block = Block(
                    type="table",
                    text=table_text,
                    html=table_html if table_html else None,
                    bbox=normalized_bbox,
                    spans=[span],
                    meta={
                        "source": "ppstructure",
                        "table_id": i,
                        "model": self.table_model,
                        "confidence": result.get('confidence', 0.8)
                    }
                )
                
                table_blocks.append(table_block)
        
        except Exception as e:
            logging.error(f"PP-Structure table extraction failed: {e}")
        
        return table_blocks
    
    def _structure_to_html(self, table_data: List[List[Dict]]) -> str:
        """
        Convert PP-Structure table data to HTML.
        
        Args:
            table_data: Structured table data from PP-Structure
            
        Returns:
            HTML table string
        """
        if not table_data:
            return ""
        
        html = "<table>\n"
        
        for row_idx, row in enumerate(table_data):
            html += "  <tr>\n"
            for cell in row:
                cell_text = cell.get('text', '').strip()
                tag = "th" if row_idx == 0 else "td"
                html += f"    <{tag}>{self._escape_html(cell_text)}</{tag}>\n"
            html += "  </tr>\n"
        
        html += "</table>"
        return html
    
    def _structure_to_text(self, table_data: List[List[Dict]]) -> str:
        """
        Convert PP-Structure table data to plain text.
        
        Args:
            table_data: Structured table data from PP-Structure
            
        Returns:
            Plain text table representation
        """
        if not table_data:
            return ""
        
        text_rows = []
        for row in table_data:
            row_texts = [cell.get('text', '').strip() for cell in row]
            text_rows.append(" | ".join(row_texts))
        
        return "\n".join(text_rows)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    def _merge_ocr_and_tables(self, ocr_blocks: List[Block], table_blocks: List[Block], width: int, height: int) -> List[Block]:
        """
        Merge OCR text blocks and table blocks, removing overlaps.
        
        Args:
            ocr_blocks: Text blocks from OCR
            table_blocks: Table blocks from PP-Structure
            width: Image width
            height: Image height
            
        Returns:
            Merged list of blocks
        """
        from src.utils.bbox import bbox_iou, denormalize_bbox
        
        # Remove OCR blocks that significantly overlap with tables
        filtered_ocr_blocks = []
        for ocr_block in ocr_blocks:
            ocr_bbox = denormalize_bbox(ocr_block["bbox"], width, height)
            
            overlaps_table = False
            for table_block in table_blocks:
                table_bbox = denormalize_bbox(table_block["bbox"], width, height)
                iou = bbox_iou(ocr_bbox, table_bbox)
                
                if iou > 0.3:  # Lower threshold for OCR vs table overlap
                    overlaps_table = True
                    break
            
            if not overlaps_table:
                filtered_ocr_blocks.append(ocr_block)
        
        # Combine filtered OCR blocks with table blocks
        all_blocks = filtered_ocr_blocks + table_blocks
        
        # Sort by reading order
        all_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        
        return all_blocks
    
    def _is_likely_drawing(self, blocks: List[Block]) -> bool:
        """
        Determine if the page is likely a technical drawing based on text patterns.
        
        Args:
            blocks: List of blocks to analyze
            
        Returns:
            True if page appears to be a drawing, False otherwise
        """
        if not blocks:
            return False
        
        # Count blocks with drawing-related keywords
        drawing_keywords = [
            'scale', 'sheet', 'drawing', 'plan', 'elevation', 'section',
            'detail', 'note', 'dimension', 'grid', 'north', 'legend'
        ]
        
        drawing_block_count = 0
        total_text_length = 0
        
        for block in blocks:
            text_lower = block["text"].lower()
            total_text_length += len(block["text"])
            
            for keyword in drawing_keywords:
                if keyword in text_lower:
                    drawing_block_count += 1
                    break
        
        # Heuristics for drawing detection
        drawing_ratio = drawing_block_count / len(blocks) if blocks else 0
        avg_text_length = total_text_length / len(blocks) if blocks else 0
        
        # Drawings typically have:
        # - Higher ratio of drawing-related keywords
        # - Shorter average text per block (labels, dimensions)
        return drawing_ratio > 0.2 or avg_text_length < 20
    
    def _cluster_drawing_text(self, blocks: List[Block], width: int, height: int) -> List[Block]:
        """
        Apply DBSCAN clustering to group text regions in drawings.
        
        Args:
            blocks: List of blocks to cluster
            width: Image width
            height: Image height
            
        Returns:
            List of blocks with potential regional grouping
        """
        if not SKLEARN_AVAILABLE or len(blocks) < 3:
            return blocks
        
        try:
            from src.utils.bbox import bbox_center, denormalize_bbox
            
            # Extract center points of blocks
            centers = []
            for block in blocks:
                abs_bbox = denormalize_bbox(block["bbox"], width, height)
                center = bbox_center(abs_bbox)
                centers.append(center)
            
            if len(centers) < 2:
                return blocks
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples)
            cluster_labels = clustering.fit_predict(centers)
            
            # Group blocks by cluster
            clustered_blocks = []
            cluster_groups = {}
            
            for i, (block, label) in enumerate(zip(blocks, cluster_labels)):
                if label == -1:  # Noise point
                    clustered_blocks.append(block)
                else:
                    if label not in cluster_groups:
                        cluster_groups[label] = []
                    cluster_groups[label].append(block)
            
            # Merge blocks within each cluster
            for cluster_id, cluster_blocks in cluster_groups.items():
                if len(cluster_blocks) == 1:
                    clustered_blocks.extend(cluster_blocks)
                else:
                    # Merge blocks in cluster
                    merged_block = self._merge_cluster_blocks(cluster_blocks, cluster_id)
                    clustered_blocks.append(merged_block)
            
            return clustered_blocks
        
        except Exception as e:
            logging.warning(f"Text clustering failed: {e}")
            return blocks
    
    def _merge_cluster_blocks(self, cluster_blocks: List[Block], cluster_id: int) -> Block:
        """
        Merge multiple blocks from the same cluster into a single block.
        
        Args:
            cluster_blocks: List of blocks in the same cluster
            cluster_id: Cluster identifier
            
        Returns:
            Merged block
        """
        # Combine text
        texts = [block["text"] for block in cluster_blocks]
        combined_text = " ".join(texts)
        
        # Merge bboxes
        bboxes = [block["bbox"] for block in cluster_blocks]
        merged_bbox = merge_bboxes(bboxes)
        
        # Combine spans
        all_spans = []
        for block in cluster_blocks:
            all_spans.extend(block["spans"])
        
        # Calculate average confidence
        confidences = []
        for block in cluster_blocks:
            for span in block["spans"]:
                confidences.append(span["conf"])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Create merged block
        merged_block = Block(
            type="drawing",  # Mark as drawing region
            text=combined_text,
            html=None,
            bbox=merged_bbox or [0.0, 0.0, 1.0, 1.0],
            spans=all_spans,
            meta={
                "source": "ocr_clustered",
                "cluster_id": cluster_id,
                "merged_blocks": len(cluster_blocks),
                "avg_confidence": avg_confidence
            }
        )
        
        return merged_block
    
    def _classify_and_flag_blocks(self, blocks: List[Block]) -> List[Block]:
        """
        Classify blocks and add low-confidence flags.
        
        Args:
            blocks: List of blocks to classify
            
        Returns:
            List of classified blocks with confidence flags
        """
        classified_blocks = []
        
        for block in blocks:
            # Calculate average confidence for the block
            confidences = [span["conf"] for span in block["spans"]]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Flag low confidence blocks
            is_low_conf = avg_confidence < self.min_confidence
            
            # Update block metadata
            block["meta"]["avg_confidence"] = avg_confidence
            block["meta"]["low_confidence"] = is_low_conf
            
            # Improve block type classification
            if block["type"] == "paragraph":
                block["type"] = self._classify_text_block(block["text"])
            
            classified_blocks.append(block)
        
        return classified_blocks
    
    def _classify_text_block(self, text: str) -> str:
        """
        Classify text block type based on content patterns.
        
        Args:
            text: Block text content
            
        Returns:
            Block type string
        """
        import re
        
        text_lower = text.lower().strip()
        
        # Title block patterns
        title_patterns = [
            r'sheet\s+\d+', r'drawing\s+no', r'project\s+no',
            r'scale\s*:', r'date\s*:', r'drawn\s+by', r'checked\s+by'
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, text_lower):
                return "titleblock"
        
        # Heading patterns
        if len(text.strip()) < 50 and any(keyword in text_lower for keyword in 
                                         ['section', 'detail', 'plan', 'elevation', 'note']):
            return "heading"
        
        # List patterns
        if re.match(r'^\s*[\d\w][\.\)]\s', text) or text.count('\n') > 2:
            return "list"
        
        return "paragraph"