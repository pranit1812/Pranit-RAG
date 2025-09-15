"""
Extraction router with provider escalation and fallback logic.
"""
import logging
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from extractors.base import BaseExtractor, ExtractorError, ExtractionResult
from models.types import PageParse
from config import Config
from utils.error_handling import (
    handle_extraction_errors,
    retry_with_backoff,
    memory_monitor,
    performance_monitor,
    log_error_with_context,
    ExtractionError,
    cleanup_temp_files,
    validate_file_access
)

# Import extractors conditionally
try:
    from extractors.docling_extractor import DoclingExtractor
    DOCLING_AVAILABLE = True
except Exception:
    DOCLING_AVAILABLE = False

try:
    from extractors.unstructured_extractor import UnstructuredExtractor
    UNSTRUCTURED_AVAILABLE = True
except Exception:
    UNSTRUCTURED_AVAILABLE = False

try:
    from extractors.native_pdf import NativePDFExtractor
    NATIVE_PDF_AVAILABLE = True
except Exception:
    NATIVE_PDF_AVAILABLE = False

try:
    from extractors.office_extractors import DOCXExtractor, XLSXExtractor
    OFFICE_AVAILABLE = True
except Exception:
    OFFICE_AVAILABLE = False


@dataclass
class ExtractionAttempt:
    """Record of an extraction attempt."""
    extractor_name: str
    success: bool
    error_message: Optional[str]
    processing_time: float
    page_count: int
    confidence_score: float


class ExtractionRouter:
    """
    Extraction router that tries providers in priority order with fallback logic.
    
    Implements provider escalation when extraction fails or confidence is low,
    with comprehensive error handling and quality assessment.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the extraction router.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize available extractors
        self.extractors = self._initialize_extractors()
        
        # Get extraction pipeline priority from config
        self.pipeline_priority = config.extract.pipeline_priority
        
        # Quality thresholds  
        self.min_confidence_threshold = getattr(config.extract, 'min_confidence', 0.7)
        self.min_text_ratio = getattr(config.extract, 'min_text_ratio', 0.1)
        self.max_retry_attempts = getattr(config.extract, 'max_retry_attempts', 3)
    
    def _initialize_extractors(self) -> Dict[str, BaseExtractor]:
        """
        Initialize all available extractors.
        
        Returns:
            Dictionary mapping extractor names to instances
        """
        extractors = {}
        
        # Docling extractor
        if DOCLING_AVAILABLE:
            try:
                self.logger.info("Initializing Docling extractor...")
                extractors['docling'] = DoclingExtractor(
                    use_ocr=True,
                    extract_tables=True
                )
                self.logger.info("Docling extractor initialized")
            except Exception as e:
                self.logger.error(f"Docling extractor initialization failed: {e}")
        else:
            self.logger.warning("Docling extractor not available (import failed)")
        
        # Unstructured extractor
        if UNSTRUCTURED_AVAILABLE:
            try:
                self.logger.info("Initializing Unstructured extractor...")
                extractors['unstructured_hi_res'] = UnstructuredExtractor(
                    strategy="hi_res",
                    use_ocr=True,
                    infer_table_structure=True
                )
                self.logger.info("Unstructured hi-res extractor initialized")
            except Exception as e:
                self.logger.error(f"Unstructured extractor initialization failed: {e}")
        else:
            self.logger.warning("Unstructured extractor not available (import failed)")
        
        # Native PDF extractor
        if NATIVE_PDF_AVAILABLE:
            try:
                self.logger.info("Initializing Native PDF extractor...")
                extractors['native_pdf'] = NativePDFExtractor(
                    use_camelot=True,
                    min_confidence=0.8
                )
                self.logger.info("Native PDF extractor initialized")
            except Exception as e:
                self.logger.error(f"Native PDF extractor initialization failed: {e}")
        else:
            self.logger.warning("Native PDF extractor not available (import failed)")
        
        # OCR extractor disabled - import was hanging due to PaddleOCR
        self.logger.info("OCR extractor disabled - PaddleOCR import was hanging")
        
        # DOCX extractor - KEEP THIS ONE (simple and fast)
        try:
            self.logger.info("Initializing DOCX extractor...")
            extractors['docx'] = DOCXExtractor(
                extract_tables=True,
                preserve_formatting=True
            )
            self.logger.info("DOCX extractor initialized")
        except ImportError as e:
            self.logger.warning(f"DOCX extractor not available: {e}")
        except Exception as e:
            self.logger.error(f"DOCX extractor initialization failed: {e}")
        
        # XLSX extractor - KEEP THIS ONE (simple and fast)
        try:
            self.logger.info("Initializing XLSX extractor...")
            extractors['xlsx'] = XLSXExtractor(
                include_empty_cells=False,
                max_rows=10000,
                max_cols=100
            )
            self.logger.info("XLSX extractor initialized")
        except ImportError as e:
            self.logger.warning(f"XLSX extractor not available: {e}")
        except Exception as e:
            self.logger.error(f"XLSX extractor initialization failed: {e}")
        
        return extractors
    
    @handle_extraction_errors(continue_on_error=True)
    def extract_document(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract content from document using provider escalation with comprehensive error handling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ExtractionResult with pages and metadata
        """
        file_path = Path(file_path)
        
        # Validate file access first
        if not validate_file_access(file_path):
            result = ExtractionResult(file_path, "extraction_router")
            result.add_error(f"Cannot access file: {file_path}")
            return result
        
        with performance_monitor(f"extract_document_{file_path.name}"):
            with memory_monitor(f"extract_document_{file_path.name}", max_memory_mb=getattr(self.config.extract, 'max_memory_mb', 2048)):
                return self._extract_document_internal(file_path)
    
    def _extract_document_internal(self, file_path: Path) -> ExtractionResult:
        """Internal extraction method with error handling."""
        start_time = time.time()
        temp_files = []
        
        # Create result container
        result = ExtractionResult(file_path, "extraction_router")
        
        try:
            # Determine suitable extractors for this file
            suitable_extractors = self._get_suitable_extractors(file_path)
            
            if not suitable_extractors:
                error_msg = f"No suitable extractors found for file: {file_path}"
                self.logger.error(error_msg)
                result.add_error(error_msg)
                return result
            
            # Try extractors in priority order
            extraction_attempts = []
            
            for extractor_name in suitable_extractors:
                if extractor_name not in self.extractors:
                    self.logger.warning(f"Extractor {extractor_name} not available")
                    continue
                
                extractor = self.extractors[extractor_name]
                attempt_start = time.time()
                
                try:
                    self.logger.info(f"Attempting extraction with {extractor_name} for {file_path}")
                    
                    # Extract all pages with timeout protection
                    pages = self._extract_with_timeout(
                        extractor,
                        file_path,
                        timeout=getattr(self.config.extract, 'timeout_seconds', 60)
                    )
                    
                    # Assess extraction quality
                    quality_score = self._assess_extraction_quality(pages)
                    
                    attempt = ExtractionAttempt(
                        extractor_name=extractor_name,
                        success=True,
                        error_message=None,
                        processing_time=time.time() - attempt_start,
                        page_count=len(pages),
                        confidence_score=quality_score
                    )
                    
                    extraction_attempts.append(attempt)
                    self.logger.info(f"{extractor_name} completed in {attempt.processing_time:.2f}s, pages={attempt.page_count}, quality={attempt.confidence_score:.2f}")
                    
                    # Check if quality is acceptable
                    if quality_score >= self.min_confidence_threshold:
                        self.logger.info(f"Successful extraction with {extractor_name} "
                                       f"(quality: {quality_score:.2f}, pages: {len(pages)})")
                        
                        # Add pages to result
                        for page in pages:
                            result.add_page(page)
                        
                        result.mark_success()
                        break
                    else:
                        self.logger.warning(f"Low quality extraction with {extractor_name} "
                                          f"(quality: {quality_score:.2f}), trying next extractor")
                        result.add_warning(f"Low quality extraction with {extractor_name} "
                                         f"(quality: {quality_score:.2f})")
                
                except TimeoutError as e:
                    error_msg = f"Extraction timeout with {extractor_name}: {str(e)}"
                    self.logger.error(error_msg)
                    log_error_with_context(e, {"file_path": str(file_path), "extractor": extractor_name}, 
                                         "extraction_timeout")
                    
                    attempt = ExtractionAttempt(
                        extractor_name=extractor_name,
                        success=False,
                        error_message=str(e),
                        processing_time=time.time() - attempt_start,
                        page_count=0,
                        confidence_score=0.0
                    )
                    
                    extraction_attempts.append(attempt)
                    result.add_error(error_msg)
                    self.logger.warning(f"{extractor_name} timed out after {attempt.processing_time:.2f}s")
                    continue
                
                except MemoryError as e:
                    error_msg = f"Memory error with {extractor_name}: {str(e)}"
                    self.logger.error(error_msg)
                    log_error_with_context(e, {"file_path": str(file_path), "extractor": extractor_name}, 
                                         "extraction_memory_error")
                    
                    # Try to free memory
                    import gc
                    gc.collect()
                    
                    attempt = ExtractionAttempt(
                        extractor_name=extractor_name,
                        success=False,
                        error_message=str(e),
                        processing_time=time.time() - attempt_start,
                        page_count=0,
                        confidence_score=0.0
                    )
                    
                    extraction_attempts.append(attempt)
                    result.add_error(error_msg)
                    self.logger.warning(f"{extractor_name} failed with MemoryError after {attempt.processing_time:.2f}s")
                    continue
                
                except Exception as e:
                    error_msg = f"Extraction failed with {extractor_name}: {str(e)}"
                    self.logger.error(error_msg)
                    log_error_with_context(e, {"file_path": str(file_path), "extractor": extractor_name}, 
                                         "extraction_error")
                    
                    attempt = ExtractionAttempt(
                        extractor_name=extractor_name,
                        success=False,
                        error_message=str(e),
                        processing_time=time.time() - attempt_start,
                        page_count=0,
                        confidence_score=0.0
                    )
                    
                    extraction_attempts.append(attempt)
                    result.add_error(error_msg)
                    self.logger.warning(f"{extractor_name} failed after {attempt.processing_time:.2f}s")
                    continue
            
            # If no extractor succeeded with good quality, use the best attempt
            if not result.success and extraction_attempts:
                best_attempt = self._select_best_attempt(extraction_attempts)
                
                if best_attempt and best_attempt.success:
                    self.logger.info(f"Using best available extraction from {best_attempt.extractor_name} "
                                   f"(quality: {best_attempt.confidence_score:.2f})")
                    
                    # Re-extract with the best extractor
                    try:
                        extractor = self.extractors[best_attempt.extractor_name]
                        pages = self._extract_with_timeout(extractor, file_path, timeout=getattr(self.config.extract, 'timeout_seconds', 60))
                        
                        for page in pages:
                            result.add_page(page)
                        
                        result.mark_success()
                        result.add_warning(f"Used fallback extraction with lower quality "
                                         f"({best_attempt.confidence_score:.2f})")
                    
                    except Exception as e:
                        error_msg = f"Failed to re-extract with best attempt: {e}"
                        self.logger.error(error_msg)
                        result.add_error(error_msg)
            
            # Last resort: fast text-only fallback if still not successful
            if not result.success:
                self.logger.warning("All extractors failed. Using fast text-only fallback.")
                pages = self._fast_text_fallback(file_path)
                for page in pages:
                    result.add_page(page)
                if pages:
                    result.mark_success()
                    result.add_warning("Used fast text-only fallback extraction.")

            # Add extraction metadata
            result.processing_time = time.time() - start_time
            self._add_extraction_metadata(result, extraction_attempts)
            
            if not result.success:
                result.add_error("All extraction attempts failed")
            
            return result
        
        except Exception as e:
            error_msg = f"Extraction router failed: {e}"
            self.logger.error(error_msg)
            log_error_with_context(e, {"file_path": str(file_path)}, "extraction_router_error")
            result.add_error(error_msg)
            result.processing_time = time.time() - start_time
            return result
        
        finally:
            # Clean up any temporary files
            if temp_files:
                cleanup_temp_files(temp_files)
    
    def _extract_with_timeout(self, extractor: BaseExtractor, file_path: Path, timeout: int = 60) -> List[PageParse]:
        """Extract pages with timeout protection."""
        import signal
        import platform
        import threading
        import time
        
        # Use provided timeout, clamp to at least 30 seconds to avoid too-fast bailouts
        timeout = max(30, int(timeout))
        
        # Only use signal timeout on Unix systems
        if platform.system() != 'Windows':
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Extraction timed out after {timeout} seconds")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                pages = extractor.parse_all_pages(file_path)
                return pages
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows timeout implementation using threading
            result = [None]
            exception = [None]
            
            def extract_worker():
                try:
                    result[0] = extractor.parse_all_pages(file_path)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=extract_worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                # Timeout occurred
                raise TimeoutError(f"Extraction timed out after {timeout} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]

    def _fast_text_fallback(self, file_path: Path) -> List[PageParse]:
        """Very fast text-only fallback using PyMuPDF, without tables/images.
        Intended for last-resort extraction so the pipeline can proceed.
        """
        try:
            import fitz
        except Exception:
            return []
        pages: List[PageParse] = []
        try:
            doc = fitz.open(str(file_path))
            for i, page in enumerate(doc):
                width, height = page.rect.width, page.rect.height
                text = page.get_text("text") or ""
                if text.strip():
                    block = {
                        "type": "paragraph",
                        "text": text.strip(),
                        "html": None,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "spans": [{"text": text[:1000], "bbox": [0.0,0.0,1.0,1.0], "rot": 0.0, "conf": 0.8}],
                        "meta": {"source": "fast_text_fallback"}
                    }
                else:
                    block = {
                        "type": "paragraph",
                        "text": "",
                        "html": None,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "spans": [],
                        "meta": {"source": "fast_text_fallback"}
                    }
                pages.append({
                    "page_no": i + 1,
                    "width": int(width),
                    "height": int(height),
                    "blocks": [block] if block["text"] else [],
                    "artifacts_removed": []
                })
            doc.close()
        except Exception:
            return []
        return pages
    
    def _get_suitable_extractors(self, file_path: Path) -> List[str]:
        """
        Determine suitable extractors for the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of extractor names in priority order
        """
        file_extension = file_path.suffix.lower()
        suitable_extractors = []
        # Basic memory throttle based on file size to avoid heavy extractors on huge files
        try:
            file_size_mb = max(0, int(file_path.stat().st_size / (1024 * 1024)))
        except Exception:
            file_size_mb = 0
        
        # Special handling for specific file types
        if file_extension == '.docx':
            return ['docx']
        elif file_extension in ['.xlsx', '.xls']:
            return ['xlsx']
        
        # For PDFs and images, use the configured pipeline priority
        for extractor_name in self.pipeline_priority:
            if extractor_name in self.extractors:
                extractor = self.extractors[extractor_name]
                if extractor.supports(file_path):
                    suitable_extractors.append(extractor_name)
        # Memory-aware reordering/filtering
        # If file is very large, prefer fast native_pdf and skip hi-res/unstructured
        if file_size_mb >= 150:
            suitable_extractors = [name for name in suitable_extractors if name in ["native_pdf", "xlsx", "docx"]]
        # If moderately large, ensure native_pdf comes first
        elif file_size_mb >= 60 and "native_pdf" in suitable_extractors:
            suitable_extractors = ["native_pdf"] + [n for n in suitable_extractors if n != "native_pdf"]

        return suitable_extractors
    
    def _assess_extraction_quality(self, pages: List[PageParse]) -> float:
        """
        Assess the quality of extracted pages.
        
        Args:
            pages: List of extracted pages
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not pages:
            return 0.0
        
        total_score = 0.0
        
        for page in pages:
            page_score = self._assess_page_quality(page)
            total_score += page_score
        
        return total_score / len(pages)
    
    def _assess_page_quality(self, page: PageParse) -> float:
        """
        Assess the quality of a single page.
        
        Args:
            page: PageParse object to assess
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not page["blocks"]:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Text content ratio
        total_text_length = sum(len(block["text"]) for block in page["blocks"])
        page_area = page["width"] * page["height"]
        text_density = min(total_text_length / max(page_area / 1000, 1), 1.0)
        quality_factors.append(text_density)
        
        # Factor 2: Block structure diversity
        block_types = set(block["type"] for block in page["blocks"])
        type_diversity = min(len(block_types) / 5.0, 1.0)  # Max 5 types
        quality_factors.append(type_diversity)
        
        # Factor 3: Average span confidence
        all_confidences = []
        for block in page["blocks"]:
            for span in block["spans"]:
                all_confidences.append(span["conf"])
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
        quality_factors.append(avg_confidence)
        
        # Factor 4: Bbox validity
        valid_bboxes = sum(1 for block in page["blocks"] 
                          if self._is_valid_bbox(block["bbox"]))
        bbox_ratio = valid_bboxes / len(page["blocks"])
        quality_factors.append(bbox_ratio)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # Text density, diversity, confidence, bbox validity
        weighted_score = sum(factor * weight for factor, weight in zip(quality_factors, weights))
        
        return min(weighted_score, 1.0)
    
    def _is_valid_bbox(self, bbox: List[float]) -> bool:
        """
        Check if bounding box is valid.
        
        Args:
            bbox: Bounding box [x0, y0, x1, y1]
            
        Returns:
            True if bbox is valid
        """
        if len(bbox) != 4:
            return False
        
        x0, y0, x1, y1 = bbox
        
        # Check bounds
        if not (0 <= x0 <= 1 and 0 <= y0 <= 1 and 0 <= x1 <= 1 and 0 <= y1 <= 1):
            return False
        
        # Check ordering
        if x1 <= x0 or y1 <= y0:
            return False
        
        return True
    
    def _select_best_attempt(self, attempts: List[ExtractionAttempt]) -> Optional[ExtractionAttempt]:
        """
        Select the best extraction attempt from the list.
        
        Args:
            attempts: List of extraction attempts
            
        Returns:
            Best attempt or None if no successful attempts
        """
        successful_attempts = [attempt for attempt in attempts if attempt.success]
        
        if not successful_attempts:
            return None
        
        # Sort by confidence score, then by page count
        successful_attempts.sort(
            key=lambda x: (x.confidence_score, x.page_count),
            reverse=True
        )
        
        return successful_attempts[0]
    
    def _add_extraction_metadata(self, result: ExtractionResult, attempts: List[ExtractionAttempt]) -> None:
        """
        Add extraction metadata to the result.
        
        Args:
            result: ExtractionResult to update
            attempts: List of extraction attempts
        """
        # Add attempt summary
        attempt_summary = []
        for attempt in attempts:
            summary = {
                "extractor": attempt.extractor_name,
                "success": attempt.success,
                "confidence": attempt.confidence_score,
                "processing_time": attempt.processing_time,
                "page_count": attempt.page_count
            }
            
            if attempt.error_message:
                summary["error"] = attempt.error_message
            
            attempt_summary.append(summary)
        
        # Store in result metadata (if result has a way to store custom metadata)
        if hasattr(result, 'metadata'):
            result.metadata = {
                "extraction_attempts": attempt_summary,
                "total_attempts": len(attempts),
                "successful_attempts": len([a for a in attempts if a.success])
            }
    
    def get_available_extractors(self) -> List[str]:
        """
        Get list of available extractor names.
        
        Returns:
            List of available extractor names
        """
        return list(self.extractors.keys())
    
    def get_extractor_info(self, extractor_name: str) -> Dict[str, Any]:
        """
        Get information about a specific extractor.
        
        Args:
            extractor_name: Name of the extractor
            
        Returns:
            Dictionary with extractor information
        """
        if extractor_name not in self.extractors:
            return {"error": f"Extractor '{extractor_name}' not found"}
        
        extractor = self.extractors[extractor_name]
        
        return {
            "name": extractor_name,
            "class": extractor.__class__.__name__,
            "supported_extensions": extractor.get_supported_extensions(),
            "available": True
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate if a file can be processed by any available extractor.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        validation_result = {
            "file_path": str(file_path),
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "extension": file_path.suffix.lower(),
            "suitable_extractors": [],
            "can_process": False
        }
        
        if not file_path.exists():
            validation_result["error"] = "File does not exist"
            return validation_result
        
        # Check which extractors can handle this file
        suitable_extractors = self._get_suitable_extractors(file_path)
        validation_result["suitable_extractors"] = suitable_extractors
        validation_result["can_process"] = len(suitable_extractors) > 0
        
        if not suitable_extractors:
            validation_result["error"] = f"No suitable extractors for file type: {file_path.suffix}"
        
        return validation_result