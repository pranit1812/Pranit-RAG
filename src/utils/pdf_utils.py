"""
PDF utilities for page rendering and image extraction.
"""
import io
from pathlib import Path
from typing import Optional, Tuple, List, Union
import logging

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. PDF utilities will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing will be limited.")


class PDFUtilsError(Exception):
    """Custom exception for PDF utilities errors."""
    pass


def check_pdf_dependencies() -> bool:
    """
    Check if required PDF processing dependencies are available.
    
    Returns:
        True if dependencies are available, False otherwise
    """
    return PYMUPDF_AVAILABLE and PIL_AVAILABLE


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """
    Get the number of pages in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF
        
    Raises:
        PDFUtilsError: If PDF cannot be opened or dependencies missing
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFUtilsError("PyMuPDF is required for PDF operations")
    
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        raise PDFUtilsError(f"Failed to get page count from {pdf_path}: {e}")


def get_pdf_page_dimensions(pdf_path: Union[str, Path], page_no: int) -> Tuple[float, float]:
    """
    Get dimensions of a specific PDF page.
    
    Args:
        pdf_path: Path to the PDF file
        page_no: Page number (0-indexed)
        
    Returns:
        Tuple of (width, height) in points
        
    Raises:
        PDFUtilsError: If PDF cannot be opened or page doesn't exist
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFUtilsError("PyMuPDF is required for PDF operations")
    
    try:
        doc = fitz.open(str(pdf_path))
        if page_no >= len(doc):
            doc.close()
            raise PDFUtilsError(f"Page {page_no} does not exist in PDF")
        
        page = doc[page_no]
        rect = page.rect
        width, height = rect.width, rect.height
        doc.close()
        
        return width, height
    except Exception as e:
        raise PDFUtilsError(f"Failed to get page dimensions: {e}")


def render_pdf_page_to_image(
    pdf_path: Union[str, Path], 
    page_no: int, 
    scale: float = 2.0,
    format: str = "PNG"
) -> bytes:
    """
    Render a PDF page to high-resolution image bytes.
    
    Args:
        pdf_path: Path to the PDF file
        page_no: Page number (0-indexed)
        scale: Scaling factor for resolution (2.0 = 2x resolution)
        format: Image format ("PNG", "JPEG")
        
    Returns:
        Image bytes in specified format
        
    Raises:
        PDFUtilsError: If rendering fails or dependencies missing
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFUtilsError("PyMuPDF is required for PDF rendering")
    
    if not PIL_AVAILABLE:
        raise PDFUtilsError("PIL is required for image processing")
    
    try:
        doc = fitz.open(str(pdf_path))
        if page_no >= len(doc):
            doc.close()
            raise PDFUtilsError(f"Page {page_no} does not exist in PDF")
        
        page = doc[page_no]
        
        # Create transformation matrix for scaling
        mat = fitz.Matrix(scale, scale)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to requested format
        output = io.BytesIO()
        img.save(output, format=format)
        image_bytes = output.getvalue()
        
        # Cleanup
        pix = None
        doc.close()
        
        return image_bytes
        
    except Exception as e:
        raise PDFUtilsError(f"Failed to render PDF page {page_no}: {e}")


def save_pdf_page_as_image(
    pdf_path: Union[str, Path], 
    page_no: int, 
    output_path: Union[str, Path],
    scale: float = 2.0,
    format: str = "PNG"
) -> None:
    """
    Save a PDF page as an image file.
    
    Args:
        pdf_path: Path to the PDF file
        page_no: Page number (0-indexed)
        output_path: Path where image will be saved
        scale: Scaling factor for resolution (2.0 = 2x resolution)
        format: Image format ("PNG", "JPEG")
        
    Raises:
        PDFUtilsError: If rendering or saving fails
    """
    image_bytes = render_pdf_page_to_image(pdf_path, page_no, scale, format)
    
    try:
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
    except Exception as e:
        raise PDFUtilsError(f"Failed to save image to {output_path}: {e}")


def extract_pdf_images(pdf_path: Union[str, Path], page_no: Optional[int] = None) -> List[bytes]:
    """
    Extract embedded images from PDF pages.
    
    Args:
        pdf_path: Path to the PDF file
        page_no: Specific page number (0-indexed), or None for all pages
        
    Returns:
        List of image bytes
        
    Raises:
        PDFUtilsError: If extraction fails or dependencies missing
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFUtilsError("PyMuPDF is required for PDF operations")
    
    try:
        doc = fitz.open(str(pdf_path))
        images = []
        
        # Determine page range
        if page_no is not None:
            if page_no >= len(doc):
                doc.close()
                raise PDFUtilsError(f"Page {page_no} does not exist in PDF")
            page_range = [page_no]
        else:
            page_range = range(len(doc))
        
        # Extract images from specified pages
        for pno in page_range:
            page = doc[pno]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
                except Exception as e:
                    logging.warning(f"Failed to extract image {img_index} from page {pno}: {e}")
                    continue
        
        doc.close()
        return images
        
    except Exception as e:
        raise PDFUtilsError(f"Failed to extract images from PDF: {e}")


def get_pdf_metadata(pdf_path: Union[str, Path]) -> dict:
    """
    Extract metadata from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
        
    Raises:
        PDFUtilsError: If metadata extraction fails
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFUtilsError("PyMuPDF is required for PDF operations")
    
    try:
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "page_count": len(doc)
        }
        
    except Exception as e:
        raise PDFUtilsError(f"Failed to extract PDF metadata: {e}")


def is_pdf_encrypted(pdf_path: Union[str, Path]) -> bool:
    """
    Check if PDF file is encrypted/password protected.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if PDF is encrypted, False otherwise
        
    Raises:
        PDFUtilsError: If PDF cannot be opened
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFUtilsError("PyMuPDF is required for PDF operations")
    
    try:
        doc = fitz.open(str(pdf_path))
        is_encrypted = doc.needs_pass
        doc.close()
        return is_encrypted
    except Exception as e:
        raise PDFUtilsError(f"Failed to check PDF encryption status: {e}")


def validate_pdf_file(pdf_path: Union[str, Path]) -> bool:
    """
    Validate that a file is a readable PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if file is a valid PDF, False otherwise
    """
    if not PYMUPDF_AVAILABLE:
        return False
    
    try:
        doc = fitz.open(str(pdf_path))
        # Try to access first page to ensure PDF is readable
        if len(doc) > 0:
            _ = doc[0]
        doc.close()
        return True
    except Exception:
        return False