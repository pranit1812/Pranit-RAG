"""
Tests for vision assistance service.
"""
import pytest
import tempfile
import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.services.vision import (
    ImageRenderer, VisionService, VisionAssistant, VisionServiceError,
    create_vision_assistant
)
from src.models.types import Chunk, ChunkMetadata, VisionConfig, ContextPacket


class TestImageRenderer:
    """Test image rendering functionality."""
    
    def test_init_with_cache_dir(self):
        """Test initialization with cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            renderer = ImageRenderer(cache_dir)
            
            assert renderer.cache_dir == cache_dir
            assert cache_dir.exists()
    
    def test_init_without_cache_dir(self):
        """Test initialization without cache directory."""
        renderer = ImageRenderer()
        assert renderer.cache_dir is None
    
    def test_get_cache_path(self):
        """Test cache path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            renderer = ImageRenderer(cache_dir)
            
            cache_path = renderer._get_cache_path("test.pdf", 0, 2.0)
            assert cache_path is not None
            assert cache_path.parent == cache_dir
            assert cache_path.suffix == ".png"
    
    def test_get_cache_path_no_cache_dir(self):
        """Test cache path when caching disabled."""
        renderer = ImageRenderer()
        cache_path = renderer._get_cache_path("test.pdf", 0, 2.0)
        assert cache_path is None
    
    @patch('src.services.vision.render_pdf_page_to_image')
    def test_render_pdf_page_success(self, mock_render):
        """Test successful PDF page rendering."""
        mock_render.return_value = b"fake_png_data"
        
        renderer = ImageRenderer()
        result = renderer.render_pdf_page("test.pdf", 0, 2.0)
        
        assert result == b"fake_png_data"
        mock_render.assert_called_once_with("test.pdf", 0, 2.0, "PNG")
    
    @patch('src.services.vision.render_pdf_page_to_image')
    def test_render_pdf_page_with_caching(self, mock_render):
        """Test PDF rendering with caching."""
        mock_render.return_value = b"fake_png_data"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            renderer = ImageRenderer(cache_dir)
            
            # First call should render and cache
            result1 = renderer.render_pdf_page("test.pdf", 0, 2.0)
            assert result1 == b"fake_png_data"
            assert mock_render.call_count == 1
            
            # Create a fake source file that's older than cache
            test_file = Path(temp_dir) / "test.pdf"
            test_file.touch()
            
            # Second call should use cache (but we can't easily test this without
            # creating actual files with proper timestamps)
    
    @patch('src.services.vision.render_pdf_page_to_image')
    def test_render_pdf_page_error(self, mock_render):
        """Test PDF rendering error handling."""
        from src.utils.pdf_utils import PDFUtilsError
        mock_render.side_effect = PDFUtilsError("Test error")
        
        renderer = ImageRenderer()
        
        with pytest.raises(VisionServiceError, match="Failed to render PDF page"):
            renderer.render_pdf_page("test.pdf", 0, 2.0)
    
    @patch('src.services.vision.PIL_AVAILABLE', True)
    @patch('src.services.vision.Image')
    def test_render_image_file_success(self, mock_image):
        """Test successful image file processing."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.mode = 'RGB'
        mock_img.width = 1000
        mock_img.height = 800
        mock_img.resize.return_value = mock_img
        
        mock_image.open.return_value.__enter__.return_value = mock_img
        
        # Mock save operation
        def mock_save(output, **kwargs):
            output.write(b"fake_png_data")
        mock_img.save = mock_save
        
        renderer = ImageRenderer()
        result = renderer.render_image_file("test.jpg", 2.0)
        
        assert result == b"fake_png_data"
        mock_img.resize.assert_called_once()
    
    @patch('src.services.vision.PIL_AVAILABLE', False)
    def test_render_image_file_no_pil(self):
        """Test image file processing without PIL."""
        renderer = ImageRenderer()
        
        with pytest.raises(VisionServiceError, match="PIL is required"):
            renderer.render_image_file("test.jpg")
    
    def test_get_page_image_pdf(self):
        """Test getting page image for PDF."""
        renderer = ImageRenderer()
        
        with patch.object(renderer, 'render_pdf_page') as mock_render:
            mock_render.return_value = b"pdf_image"
            
            result = renderer.get_page_image("test.pdf", 0, "pdf", 2.0)
            assert result == b"pdf_image"
            mock_render.assert_called_once_with("test.pdf", 0, 2.0)
    
    def test_get_page_image_image(self):
        """Test getting page image for image file."""
        renderer = ImageRenderer()
        
        with patch.object(renderer, 'render_image_file') as mock_render:
            mock_render.return_value = b"image_data"
            
            result = renderer.get_page_image("test.jpg", 0, "image", 2.0)
            assert result == b"image_data"
            mock_render.assert_called_once_with("test.jpg", 2.0)
    
    def test_get_page_image_unsupported(self):
        """Test getting page image for unsupported file type."""
        renderer = ImageRenderer()
        
        with pytest.raises(VisionServiceError, match="not yet implemented"):
            renderer.get_page_image("test.docx", 0, "docx", 2.0)
        
        with pytest.raises(VisionServiceError, match="Unsupported file type"):
            renderer.get_page_image("test.txt", 0, "txt", 2.0)


class TestVisionService:
    """Test OpenAI Vision API integration."""
    
    @patch('src.services.vision.OPENAI_AVAILABLE', True)
    @patch('src.services.vision.openai.OpenAI')
    def test_init_success(self, mock_openai):
        """Test successful initialization."""
        service = VisionService("test_key", "gpt-4o")
        assert service.model == "gpt-4o"
        mock_openai.assert_called_once_with(api_key="test_key")
    
    @patch('src.services.vision.OPENAI_AVAILABLE', False)
    def test_init_no_openai(self):
        """Test initialization without OpenAI library."""
        with pytest.raises(VisionServiceError, match="OpenAI library is required"):
            VisionService("test_key")
    
    @patch('src.services.vision.OPENAI_AVAILABLE', True)
    @patch('src.services.vision.openai.OpenAI')
    @patch('base64.b64encode')
    def test_analyze_images_success(self, mock_b64, mock_openai):
        """Test successful image analysis."""
        # Mock base64 encoding
        mock_b64.return_value.decode.return_value = "fake_base64"
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Analysis result"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        service = VisionService("test_key")
        result = service.analyze_images_with_query(
            "What do you see?", 
            [b"image1", b"image2"],
            "Context text"
        )
        
        assert result == "Analysis result"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.services.vision.OPENAI_AVAILABLE', True)
    @patch('src.services.vision.openai.OpenAI')
    def test_analyze_images_no_images(self, mock_openai):
        """Test analysis with no images."""
        service = VisionService("test_key")
        
        with pytest.raises(VisionServiceError, match="No images provided"):
            service.analyze_images_with_query("Query", [])
    
    @patch('src.services.vision.OPENAI_AVAILABLE', True)
    @patch('src.services.vision.openai.OpenAI')
    def test_analyze_images_api_error(self, mock_openai):
        """Test API error handling."""
        import openai
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.OpenAIError("API Error")
        mock_openai.return_value = mock_client
        
        service = VisionService("test_key")
        
        with pytest.raises(VisionServiceError, match="OpenAI Vision API error"):
            service.analyze_images_with_query("Query", [b"image"])


class TestVisionAssistant:
    """Test vision assistant coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_renderer = Mock(spec=ImageRenderer)
        self.mock_service = Mock(spec=VisionService)
        self.config = {
            "enabled": True,
            "max_images": 3,
            "resolution_scale": 2.0
        }
        self.assistant = VisionAssistant(
            self.mock_renderer, 
            self.mock_service, 
            self.config
        )
    
    def create_test_chunk(self, doc_name: str, page_start: int) -> Chunk:
        """Create a test chunk."""
        metadata: ChunkMetadata = {
            "project_id": "test_project",
            "doc_id": "test_doc",
            "doc_name": doc_name,
            "file_type": "pdf",
            "page_start": page_start,
            "page_end": page_start,
            "content_type": "SpecSection",
            "division_code": None,
            "division_title": None,
            "section_code": None,
            "section_title": None,
            "discipline": None,
            "sheet_number": None,
            "sheet_title": None,
            "bbox_regions": [],
            "low_conf": False
        }
        
        return {
            "id": f"chunk_{page_start}",
            "text": f"Test content for page {page_start}",
            "html": None,
            "metadata": metadata,
            "token_count": 100,
            "text_hash": "test_hash"
        }
    
    def test_get_images_for_chunks_success(self):
        """Test successful image extraction for chunks."""
        chunks = [
            self.create_test_chunk("doc1.pdf", 0),
            self.create_test_chunk("doc2.pdf", 1)
        ]
        
        self.mock_renderer.get_page_image.side_effect = [b"image1", b"image2"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            raw_dir = storage_dir / "raw"
            raw_dir.mkdir()
            
            # Create dummy files
            (raw_dir / "doc1.pdf").touch()
            (raw_dir / "doc2.pdf").touch()
            
            images = self.assistant.get_images_for_chunks(chunks, storage_dir)
            
            assert len(images) == 2
            assert images[0] == (b"image1", "doc1.pdf (Page 1)")
            assert images[1] == (b"image2", "doc2.pdf (Page 2)")
    
    def test_get_images_for_chunks_disabled(self):
        """Test image extraction when vision is disabled."""
        self.config["enabled"] = False
        chunks = [self.create_test_chunk("doc1.pdf", 0)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            
            images = self.assistant.get_images_for_chunks(chunks, storage_dir)
            assert images == []
    
    def test_get_images_for_chunks_max_limit(self):
        """Test image extraction with max limit."""
        chunks = [
            self.create_test_chunk("doc1.pdf", 0),
            self.create_test_chunk("doc2.pdf", 1),
            self.create_test_chunk("doc3.pdf", 2),
            self.create_test_chunk("doc4.pdf", 3)
        ]
        
        self.mock_renderer.get_page_image.side_effect = [b"image1", b"image2"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            raw_dir = storage_dir / "raw"
            raw_dir.mkdir()
            
            # Create dummy files
            (raw_dir / "doc1.pdf").touch()
            (raw_dir / "doc2.pdf").touch()
            (raw_dir / "doc3.pdf").touch()
            (raw_dir / "doc4.pdf").touch()
            
            images = self.assistant.get_images_for_chunks(chunks, storage_dir, max_images=2)
            assert len(images) == 2
    
    def test_get_images_for_chunks_with_sheet_number(self):
        """Test image extraction with sheet numbers."""
        chunk = self.create_test_chunk("drawing.pdf", 0)
        chunk["metadata"]["sheet_number"] = "A-101"
        
        self.mock_renderer.get_page_image.return_value = b"image1"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            raw_dir = storage_dir / "raw"
            raw_dir.mkdir()
            
            # Create dummy file
            (raw_dir / "drawing.pdf").touch()
            
            images = self.assistant.get_images_for_chunks([chunk], storage_dir)
            
            assert len(images) == 1
            assert images[0][1] == "drawing.pdf (Page 1) - Sheet A-101"
    
    def test_enhance_answer_with_vision_success(self):
        """Test successful vision enhancement."""
        chunks = [self.create_test_chunk("doc1.pdf", 0)]
        context_packet: ContextPacket = {
            "chunks": chunks,
            "total_tokens": 100,
            "sources": {},
            "project_context": {
                "project_name": "Test Project",
                "description": "Test",
                "project_type": "Commercial",
                "location": None,
                "key_systems": [],
                "disciplines_involved": [],
                "summary": "Test project"
            }
        }
        
        self.mock_renderer.get_page_image.return_value = b"image1"
        self.mock_service.analyze_images_with_query.return_value = "Vision analysis result"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            raw_dir = storage_dir / "raw"
            raw_dir.mkdir()
            
            # Create dummy file
            (raw_dir / "doc1.pdf").touch()
            
            result = self.assistant.enhance_answer_with_vision(
                "What do you see?", 
                context_packet, 
                storage_dir
            )
            
            assert result == "Vision analysis result"
            self.mock_service.analyze_images_with_query.assert_called_once()
    
    def test_enhance_answer_with_vision_disabled(self):
        """Test vision enhancement when disabled."""
        self.config["enabled"] = False
        
        result = self.assistant.enhance_answer_with_vision(
            "Query", 
            {"chunks": [], "total_tokens": 0, "sources": {}, "project_context": {}}, 
            Path("/tmp")
        )
        
        assert result is None
    
    def test_enhance_answer_with_vision_no_images(self):
        """Test vision enhancement with no available images."""
        context_packet: ContextPacket = {
            "chunks": [],
            "total_tokens": 0,
            "sources": {},
            "project_context": {
                "project_name": "Test Project",
                "description": "Test",
                "project_type": "Commercial",
                "location": None,
                "key_systems": [],
                "disciplines_involved": [],
                "summary": "Test project"
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            
            result = self.assistant.enhance_answer_with_vision(
                "Query", 
                context_packet, 
                storage_dir
            )
            
            assert result is None
    
    def test_enhance_answer_with_vision_error(self):
        """Test vision enhancement error handling."""
        chunks = [self.create_test_chunk("doc1.pdf", 0)]
        context_packet: ContextPacket = {
            "chunks": chunks,
            "total_tokens": 100,
            "sources": {},
            "project_context": {
                "project_name": "Test Project",
                "description": "Test",
                "project_type": "Commercial",
                "location": None,
                "key_systems": [],
                "disciplines_involved": [],
                "summary": "Test project"
            }
        }
        
        self.mock_renderer.get_page_image.side_effect = Exception("Render error")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            
            result = self.assistant.enhance_answer_with_vision(
                "Query", 
                context_packet, 
                storage_dir
            )
            
            assert result is None
    
    def test_is_available_true(self):
        """Test availability check when all conditions met."""
        with patch('src.services.vision.OPENAI_AVAILABLE', True), \
             patch('src.services.vision.PIL_AVAILABLE', True):
            
            assert self.assistant.is_available() is True
    
    def test_is_available_disabled(self):
        """Test availability check when disabled."""
        self.config["enabled"] = False
        assert self.assistant.is_available() is False
    
    def test_is_available_no_openai(self):
        """Test availability check without OpenAI."""
        with patch('src.services.vision.OPENAI_AVAILABLE', False):
            assert self.assistant.is_available() is False


class TestCreateVisionAssistant:
    """Test vision assistant factory function."""
    
    @patch('src.services.vision.VisionService')
    def test_create_with_defaults(self, mock_vision_service):
        """Test creation with default configuration."""
        assistant = create_vision_assistant()
        
        assert isinstance(assistant, VisionAssistant)
        assert assistant.config["enabled"] is False
        assert assistant.config["max_images"] == 3
        assert assistant.config["resolution_scale"] == 2.0
    
    @patch('src.services.vision.VisionService')
    def test_create_with_custom_config(self, mock_vision_service):
        """Test creation with custom configuration."""
        config: VisionConfig = {
            "enabled": True,
            "max_images": 5,
            "resolution_scale": 1.5
        }
        
        assistant = create_vision_assistant(config=config)
        
        assert assistant.config == config
    
    @patch('src.services.vision.VisionService')
    def test_create_with_cache_dir(self, mock_vision_service):
        """Test creation with cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            assistant = create_vision_assistant(cache_dir=cache_dir)
            
            assert assistant.image_renderer.cache_dir == cache_dir
    
    @patch('src.services.vision.VisionService')
    def test_create_vision_service_error(self, mock_vision_service):
        """Test creation when vision service fails."""
        mock_vision_service.side_effect = VisionServiceError("Test error")
        
        config: VisionConfig = {
            "enabled": True,
            "max_images": 3,
            "resolution_scale": 2.0
        }
        
        assistant = create_vision_assistant(config=config)
        
        # Should disable vision when service creation fails
        assert assistant.config["enabled"] is False