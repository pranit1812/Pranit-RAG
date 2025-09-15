"""
Vision assistance service for document image analysis.
"""
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import hashlib
import json
import base64

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing will be limited.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Vision assistance will be disabled.")

from models.types import Chunk, VisionConfig, ContextPacket
from utils.pdf_utils import render_pdf_page_to_image, PDFUtilsError
from utils.io_utils import ensure_directory


class VisionServiceError(Exception):
    """Custom exception for vision service errors."""
    pass


class ImageRenderer:
    """Service for rendering document pages to images."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize image renderer.
        
        Args:
            cache_dir: Directory for caching rendered images
        """
        self.cache_dir = cache_dir
        if self.cache_dir:
            ensure_directory(self.cache_dir)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_path(self, doc_path: str, page_no: int, scale: float) -> Optional[Path]:
        """
        Get cache file path for rendered image.
        
        Args:
            doc_path: Path to document
            page_no: Page number
            scale: Resolution scale
            
        Returns:
            Cache file path or None if caching disabled
        """
        if not self.cache_dir:
            return None
        
        # Create cache key from document path, page, and scale
        cache_key = f"{doc_path}_{page_no}_{scale}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        return self.cache_dir / f"{cache_hash}.png"
    
    def _is_cached(self, cache_path: Path, doc_path: str) -> bool:
        """
        Check if cached image is still valid.
        
        Args:
            cache_path: Path to cached image
            doc_path: Path to source document
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        try:
            doc_stat = Path(doc_path).stat()
            cache_stat = cache_path.stat()
            
            # Cache is valid if it's newer than the source document
            return cache_stat.st_mtime > doc_stat.st_mtime
        except (OSError, FileNotFoundError):
            return False
    
    def render_pdf_page(
        self, 
        doc_path: Union[str, Path], 
        page_no: int, 
        scale: float = 2.0
    ) -> bytes:
        """
        Render PDF page to high-resolution PNG image.
        
        Args:
            doc_path: Path to PDF document
            page_no: Page number (0-indexed)
            scale: Resolution scale factor (2.0 = 2x resolution)
            
        Returns:
            PNG image bytes
            
        Raises:
            VisionServiceError: If rendering fails
        """
        doc_path_str = str(doc_path)
        cache_path = self._get_cache_path(doc_path_str, page_no, scale)
        
        # Check cache first
        if cache_path and self._is_cached(cache_path, doc_path_str):
            try:
                with open(cache_path, 'rb') as f:
                    self.logger.debug(f"Using cached image for {doc_path_str} page {page_no}")
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read cached image: {e}")
        
        # Render new image
        try:
            self.logger.debug(f"Rendering {doc_path_str} page {page_no} at {scale}x scale")
            image_bytes = render_pdf_page_to_image(doc_path, page_no, scale, "PNG")
            
            # Cache the result
            if cache_path:
                try:
                    with open(cache_path, 'wb') as f:
                        f.write(image_bytes)
                    self.logger.debug(f"Cached image at {cache_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache image: {e}")
            
            return image_bytes
            
        except PDFUtilsError as e:
            raise VisionServiceError(f"Failed to render PDF page: {e}")
        except Exception as e:
            raise VisionServiceError(f"Unexpected error rendering page: {e}")
    
    def render_image_file(
        self, 
        image_path: Union[str, Path], 
        scale: float = 2.0,
        max_size: Tuple[int, int] = (2048, 2048)
    ) -> bytes:
        """
        Process image file for vision model consumption.
        
        Args:
            image_path: Path to image file
            scale: Scale factor for resizing
            max_size: Maximum dimensions (width, height)
            
        Returns:
            Optimized PNG image bytes
            
        Raises:
            VisionServiceError: If processing fails
        """
        if not PIL_AVAILABLE:
            raise VisionServiceError("PIL is required for image processing")
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply scaling
                if scale != 1.0:
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Ensure image doesn't exceed max size
                if img.width > max_size[0] or img.height > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to PNG bytes
                output = io.BytesIO()
                img.save(output, format='PNG', optimize=True)
                return output.getvalue()
                
        except Exception as e:
            raise VisionServiceError(f"Failed to process image file {image_path}: {e}")
    
    def get_page_image(
        self, 
        doc_path: Union[str, Path], 
        page_no: int, 
        file_type: str,
        scale: float = 2.0
    ) -> bytes:
        """
        Get page image for any supported document type.
        
        Args:
            doc_path: Path to document
            page_no: Page number (0-indexed)
            file_type: Document file type ("pdf", "docx", "xlsx", "image")
            scale: Resolution scale factor
            
        Returns:
            PNG image bytes
            
        Raises:
            VisionServiceError: If image extraction fails
        """
        if file_type == "pdf":
            return self.render_pdf_page(doc_path, page_no, scale)
        elif file_type == "image":
            return self.render_image_file(doc_path, scale)
        elif file_type in ["docx", "xlsx"]:
            # For Office documents, we would need to convert to PDF first
            # This is a simplified implementation - in practice, you might want
            # to use libraries like python-docx2pdf or similar
            raise VisionServiceError(f"Image extraction from {file_type} files not yet implemented")
        else:
            raise VisionServiceError(f"Unsupported file type for image extraction: {file_type}")


class VisionService:
    """Service for OpenAI Vision API integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize vision service.
        
        Args:
            api_key: OpenAI API key
            model: Vision model to use
        """
        if not OPENAI_AVAILABLE:
            raise VisionServiceError("OpenAI library is required for vision assistance")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def analyze_images_with_query(
        self, 
        query: str, 
        images: List[bytes], 
        text_context: str = "",
        max_tokens: int = 1000
    ) -> str:
        """
        Analyze images with OpenAI Vision API.
        
        Args:
            query: User query
            images: List of image bytes (PNG format)
            text_context: Additional text context
            max_tokens: Maximum tokens in response
            
        Returns:
            Vision analysis response
            
        Raises:
            VisionServiceError: If API call fails
        """
        if not images:
            raise VisionServiceError("No images provided for analysis")
        
        try:
            # Prepare image content for API
            image_content = []
            for i, image_bytes in enumerate(images):
                # Convert bytes to base64 for API
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high"
                    }
                })
            
            # Build system message
            system_message = (
                "You are an expert construction document analyst. "
                "Analyze the provided images in the context of the user's query. "
                "Focus on technical details, specifications, drawings, and any relevant construction information. "
                "Be specific and reference what you see in the images."
            )
            
            # Build user message
            user_message_parts = [{"type": "text", "text": f"Query: {query}"}]
            
            if text_context:
                user_message_parts.append({
                    "type": "text", 
                    "text": f"\nAdditional context from document text:\n{text_context}"
                })
            
            user_message_parts.append({
                "type": "text",
                "text": "\nPlease analyze the following images and provide insights relevant to the query:"
            })
            
            # Add images
            user_message_parts.extend(image_content)
            
            # Make API call
            self.logger.debug(f"Sending {len(images)} images to vision API")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message_parts}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            if not response.choices:
                raise VisionServiceError("No response from vision API")
            
            return response.choices[0].message.content or ""
            
        except openai.OpenAIError as e:
            raise VisionServiceError(f"OpenAI Vision API error: {e}")
        except Exception as e:
            raise VisionServiceError(f"Unexpected error in vision analysis: {e}")


class VisionAssistant:
    """High-level vision assistance coordinator."""
    
    def __init__(
        self, 
        image_renderer: ImageRenderer,
        vision_service: VisionService,
        config: VisionConfig
    ):
        """
        Initialize vision assistant.
        
        Args:
            image_renderer: Image rendering service
            vision_service: Vision API service
            config: Vision configuration
        """
        self.image_renderer = image_renderer
        self.vision_service = vision_service
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_images_for_chunks(
        self, 
        chunks: List[Chunk], 
        project_storage_dir: Path,
        max_images: Optional[int] = None
    ) -> List[Tuple[bytes, str]]:
        """
        Get images for the top chunks.
        
        Args:
            chunks: List of chunks to get images for
            project_storage_dir: Project storage directory
            max_images: Maximum number of images (overrides config)
            
        Returns:
            List of (image_bytes, source_info) tuples
            
        Raises:
            VisionServiceError: If image extraction fails
        """
        if not self.config["enabled"]:
            return []
        
        max_imgs = max_images or self.config["max_images"]
        max_imgs = min(max_imgs, len(chunks))
        
        images = []
        
        for i, chunk in enumerate(chunks[:max_imgs]):
            try:
                metadata = chunk["metadata"]
                doc_name = metadata["doc_name"]
                page_no = metadata["page_start"]  # Use first page of chunk
                file_type = metadata["file_type"]
                
                # Construct document path
                doc_path = project_storage_dir / "raw" / doc_name
                
                if not doc_path.exists():
                    self.logger.warning(f"Document not found: {doc_path}")
                    continue
                
                # Get image
                image_bytes = self.image_renderer.get_page_image(
                    doc_path, 
                    page_no, 
                    file_type,
                    self.config["resolution_scale"]
                )
                
                # Create source info
                source_info = f"{doc_name} (Page {page_no + 1})"
                if metadata.get("sheet_number"):
                    source_info += f" - Sheet {metadata['sheet_number']}"
                
                images.append((image_bytes, source_info))
                self.logger.debug(f"Added image for chunk {i+1}: {source_info}")
                
            except Exception as e:
                self.logger.warning(f"Failed to get image for chunk {i+1}: {e}")
                continue
        
        return images
    
    def enhance_answer_with_vision(
        self, 
        query: str, 
        context_packet: ContextPacket,
        project_storage_dir: Path,
        max_images: Optional[int] = None
    ) -> Optional[str]:
        """
        Enhance answer with vision analysis.
        
        Args:
            query: User query
            context_packet: Context packet with retrieved chunks
            project_storage_dir: Project storage directory
            max_images: Maximum number of images to analyze
            
        Returns:
            Vision analysis result or None if vision disabled/failed
        """
        if not self.config["enabled"]:
            return None
        
        try:
            # Get images for top chunks
            images_with_info = self.get_images_for_chunks(
                context_packet["chunks"], 
                project_storage_dir,
                max_images
            )
            
            if not images_with_info:
                self.logger.info("No images available for vision analysis")
                return None
            
            # Extract just the image bytes
            images = [img_bytes for img_bytes, _ in images_with_info]
            
            # Build text context from chunks
            text_context = "\n\n".join([
                f"Source: {chunk['metadata']['doc_name']} (Page {chunk['metadata']['page_start'] + 1})\n"
                f"Content: {chunk['text'][:500]}..."  # Truncate for API limits
                for chunk in context_packet["chunks"][:len(images)]
            ])
            
            # Get vision analysis
            vision_response = self.vision_service.analyze_images_with_query(
                query=query,
                images=images,
                text_context=text_context
            )
            
            # Log success
            source_list = [info for _, info in images_with_info]
            self.logger.info(f"Vision analysis completed for {len(images)} images: {source_list}")
            
            return vision_response
            
        except Exception as e:
            self.logger.error(f"Vision analysis failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if vision assistance is available.
        
        Returns:
            True if vision assistance can be used
        """
        return (
            self.config["enabled"] and 
            OPENAI_AVAILABLE and 
            PIL_AVAILABLE and
            self.vision_service is not None
        )


def create_vision_assistant(
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    config: Optional[VisionConfig] = None
) -> VisionAssistant:
    """
    Create a vision assistant with default configuration.
    
    Args:
        cache_dir: Directory for image caching
        api_key: OpenAI API key
        config: Vision configuration
        
    Returns:
        Configured VisionAssistant instance
        
    Raises:
        VisionServiceError: If required dependencies are missing
    """
    if config is None:
        config = {
            "enabled": False,
            "max_images": 3,
            "resolution_scale": 2.0
        }
    
    image_renderer = ImageRenderer(cache_dir)
    
    vision_service = None
    if config["enabled"]:
        try:
            vision_service = VisionService(api_key)
        except VisionServiceError as e:
            logging.warning(f"Vision service unavailable: {e}")
            config = dict(config)  # Make a copy
            config["enabled"] = False
    
    return VisionAssistant(image_renderer, vision_service, config)