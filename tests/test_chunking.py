"""
Tests for the chunking system.
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.chunking import DocumentChunker, TokenCounter, TableProcessor, ListProcessor, DrawingProcessor
from src.chunking.context_window import ContextWindow
from src.models.types import (
    PageParse, Block, Chunk, ChunkPolicy, ChunkMetadata, 
    generate_text_hash
)


class TestTokenCounter:
    """Test token counting functionality."""
    
    def test_token_counter_initialization(self):
        """Test token counter initialization."""
        counter = TokenCounter("gpt-4o")
        assert counter is not None
    
    def test_count_tokens_basic(self):
        """Test basic token counting."""
        counter = TokenCounter("gpt-4o")
        
        # Test empty string
        assert counter.count_tokens("") == 0
        
        # Test simple text
        count = counter.count_tokens("Hello world")
        assert count > 0
        assert isinstance(count, int)
    
    def test_truncate_to_tokens(self):
        """Test token truncation."""
        counter = TokenCounter("gpt-4o")
        
        text = "This is a test sentence with multiple words that should be truncated."
        truncated = counter.truncate_to_tokens(text, 5)
        
        assert len(truncated) < len(text)
        assert counter.count_tokens(truncated) <= 5
    
    def test_split_by_sentences(self):
        """Test sentence splitting."""
        counter = TokenCounter("gpt-4o")
        
        text = "First sentence. Second sentence! Third sentence?"
        chunks = counter.split_by_sentences(text, 10)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert counter.count_tokens(chunk) <= 10


class TestTableProcessor:
    """Test table processing functionality."""
    
    def test_table_processor_initialization(self):
        """Test table processor initialization."""
        processor = TableProcessor()
        assert processor is not None
    
    def test_process_table_block_with_html(self):
        """Test processing table block with HTML."""
        processor = TableProcessor()
        
        block = {
            "type": "table",
            "text": "Header1 | Header2\nValue1 | Value2",
            "html": "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Value1</td><td>Value2</td></tr></table>",
            "bbox": [0, 0, 100, 50],
            "spans": [],
            "meta": {}
        }
        
        text, html, csv = processor.process_table_block(block)
        
        assert text == "Header1 | Header2\nValue1 | Value2"
        assert html == block["html"]
        assert csv is not None
        assert "Header1,Header2" in csv
    
    def test_detect_delimiter(self):
        """Test delimiter detection."""
        processor = TableProcessor()
        
        # Test pipe delimiter
        lines = ["Col1 | Col2 | Col3", "Val1 | Val2 | Val3"]
        delimiter = processor._detect_delimiter(lines)
        assert delimiter == "|"
        
        # Test tab delimiter
        lines = ["Col1\tCol2\tCol3", "Val1\tVal2\tVal3"]
        delimiter = processor._detect_delimiter(lines)
        assert delimiter == "\t"
        
        # Test multiple spaces
        lines = ["Col1    Col2    Col3", "Val1    Val2    Val3"]
        delimiter = processor._detect_delimiter(lines)
        assert delimiter == "spaces"


class TestListProcessor:
    """Test list processing functionality."""
    
    def test_list_processor_initialization(self):
        """Test list processor initialization."""
        processor = ListProcessor()
        assert processor is not None
    
    def test_process_list_block(self):
        """Test processing list block."""
        processor = ListProcessor()
        
        block = {
            "type": "list",
            "text": "List Title\n• Item 1\n• Item 2\n• Item 3",
            "html": None,
            "bbox": [0, 0, 100, 100],
            "spans": [],
            "meta": {}
        }
        
        full_text, intro_lines, list_items = processor.process_list_block(block)
        
        assert full_text == "List Title\n• Item 1\n• Item 2\n• Item 3"
        assert intro_lines == ["List Title"]
        assert len(list_items) == 3
        assert "• Item 1" in list_items
    
    def test_detect_list_type(self):
        """Test list type detection."""
        processor = ListProcessor()
        
        # Bullet list
        items = ["• Item 1", "• Item 2"]
        list_type = processor.detect_list_type(items)
        assert list_type == "bullet"
        
        # Numbered list
        items = ["1. Item 1", "2. Item 2"]
        list_type = processor.detect_list_type(items)
        assert list_type == "numbered"
        
        # Mixed list
        items = ["• Item 1", "1. Item 2"]
        list_type = processor.detect_list_type(items)
        assert list_type == "mixed"


class TestDrawingProcessor:
    """Test drawing processing functionality."""
    
    def test_drawing_processor_initialization(self):
        """Test drawing processor initialization."""
        processor = DrawingProcessor()
        assert processor is not None
    
    def test_process_drawing_page_simple(self):
        """Test simple drawing page processing."""
        processor = DrawingProcessor()
        
        blocks = [
            {
                "type": "drawing",
                "text": "Drawing text 1",
                "bbox": [0, 0, 50, 25],
                "spans": [],
                "meta": {}
            },
            {
                "type": "titleblock",
                "text": "Title block text",
                "bbox": [100, 100, 200, 150],
                "spans": [],
                "meta": {}
            }
        ]
        
        page = {
            "page_no": 1,
            "width": 300,
            "height": 200,
            "blocks": blocks,
            "artifacts_removed": []
        }
        
        combined_text, all_bboxes, clustered_regions = processor.process_drawing_page(
            blocks, page, cluster_text=False
        )
        
        assert "Drawing text 1" in combined_text
        assert "Title block text" in combined_text
        assert len(all_bboxes) == 2
        assert len(clustered_regions) == 1
    
    def test_analyze_drawing_layout(self):
        """Test drawing layout analysis."""
        processor = DrawingProcessor()
        
        blocks = [
            {
                "type": "titleblock",
                "text": "FLOOR PLAN - LEVEL 1",
                "bbox": [0, 0, 100, 20],
                "spans": [],
                "meta": {}
            },
            {
                "type": "drawing",
                "text": "Room label",
                "bbox": [50, 50, 100, 70],
                "spans": [],
                "meta": {}
            }
        ]
        
        page = {
            "page_no": 1,
            "width": 200,
            "height": 150,
            "blocks": blocks,
            "artifacts_removed": []
        }
        
        analysis = processor.analyze_drawing_layout(blocks, page)
        
        assert analysis["total_text_regions"] == 2
        assert analysis["title_block_detected"] is True
        assert analysis["drawing_type"] == "floor_plan"


class TestContextWindow:
    """Test context window functionality."""
    
    def test_context_window_initialization(self):
        """Test context window initialization."""
        window = ContextWindow(window_size=2)
        assert window.window_size == 2
        
        # Test negative window size
        window = ContextWindow(window_size=-1)
        assert window.window_size == 0
    
    def test_get_adjacent_chunks(self):
        """Test getting adjacent chunks."""
        window = ContextWindow(window_size=1)
        
        # Create test chunks
        chunks = []
        for i in range(5):
            chunk = {
                "id": f"chunk_{i}",
                "text": f"Text {i}",
                "html": None,
                "metadata": {
                    "project_id": "test",
                    "doc_id": "doc1",
                    "doc_name": "test.pdf",
                    "file_type": "pdf",
                    "page_start": 1,
                    "page_end": 1,
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
                },
                "token_count": 10,
                "text_hash": generate_text_hash(f"Text {i}")
            }
            chunks.append(chunk)
        
        # Test getting adjacent chunks for middle chunk
        previous, next_chunks = window.get_adjacent_chunks(chunks, "chunk_2")
        
        assert len(previous) == 1
        assert len(next_chunks) == 1
        assert previous[0]["id"] == "chunk_1"
        assert next_chunks[0]["id"] == "chunk_3"
    
    def test_build_contextual_chunk(self):
        """Test building contextual chunk."""
        window = ContextWindow(window_size=1)
        
        target_chunk = {
            "id": "target",
            "text": "Target text",
            "html": None,
            "metadata": {
                "project_id": "test",
                "doc_id": "doc1",
                "doc_name": "test.pdf",
                "file_type": "pdf",
                "page_start": 1,
                "page_end": 1,
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
            },
            "token_count": 10,
            "text_hash": generate_text_hash("Target text")
        }
        
        previous_chunks = [{
            "id": "prev",
            "text": "Previous text",
            "metadata": target_chunk["metadata"].copy()
        }]
        
        next_chunks = [{
            "id": "next", 
            "text": "Next text",
            "metadata": target_chunk["metadata"].copy()
        }]
        
        contextual_chunk = window.build_contextual_chunk(
            target_chunk, previous_chunks, next_chunks, "summary"
        )
        
        assert "Previous context:" in contextual_chunk["text"]
        assert "Following context:" in contextual_chunk["text"]
        assert "Target text" in contextual_chunk["text"]
        assert contextual_chunk["metadata"]["has_context"] is True


class TestDocumentChunker:
    """Test main document chunker."""
    
    @pytest.fixture
    def sample_policy(self):
        """Sample chunking policy."""
        return {
            "target_tokens": 500,
            "max_tokens": 900,
            "preserve_tables": True,
            "preserve_lists": True,
            "drawing_cluster_text": True,
            "drawing_max_regions": 8
        }
    
    @pytest.fixture
    def sample_page(self):
        """Sample page data."""
        return {
            "page_no": 1,
            "width": 612,
            "height": 792,
            "blocks": [
                {
                    "type": "heading",
                    "text": "1. Introduction",
                    "html": None,
                    "bbox": [50, 700, 200, 720],
                    "spans": [
                        {
                            "text": "1. Introduction",
                            "bbox": [50, 700, 200, 720],
                            "rot": 0.0,
                            "conf": 1.0
                        }
                    ],
                    "meta": {}
                },
                {
                    "type": "paragraph",
                    "text": "This is a sample paragraph with some content.",
                    "html": None,
                    "bbox": [50, 650, 500, 690],
                    "spans": [
                        {
                            "text": "This is a sample paragraph with some content.",
                            "bbox": [50, 650, 500, 690],
                            "rot": 0.0,
                            "conf": 1.0
                        }
                    ],
                    "meta": {}
                }
            ],
            "artifacts_removed": []
        }
    
    @pytest.fixture
    def sample_doc_metadata(self):
        """Sample document metadata."""
        return {
            "project_id": "test_project",
            "doc_id": "doc_123",
            "doc_name": "test_document.pdf",
            "file_type": "pdf"
        }
    
    def test_chunker_initialization(self, sample_policy):
        """Test chunker initialization."""
        chunker = DocumentChunker(sample_policy)
        assert chunker.policy == sample_policy
        assert chunker.token_counter is not None
        assert chunker.classifier is not None
    
    @patch('src.services.classification.ContentClassifier')
    def test_chunk_page_basic(self, mock_classifier, sample_policy, sample_page, sample_doc_metadata):
        """Test basic page chunking."""
        # Mock the classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_blocks.return_value = ("SpecSection", {})
        mock_classifier.return_value = mock_classifier_instance
        
        chunker = DocumentChunker(sample_policy)
        chunker.classifier = mock_classifier_instance
        
        chunks = chunker.chunk_page(sample_page, sample_doc_metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert "token_count" in chunk
            assert "text_hash" in chunk
    
    def test_group_blocks_by_type(self, sample_policy, sample_page):
        """Test grouping blocks by type."""
        chunker = DocumentChunker(sample_policy)
        
        grouped = chunker._group_blocks_by_type(sample_page["blocks"])
        
        assert "heading" in grouped
        assert "paragraph" in grouped
        assert len(grouped["heading"]) == 1
        assert len(grouped["paragraph"]) == 1
    
    def test_update_heading_context(self, sample_policy):
        """Test heading context updates."""
        chunker = DocumentChunker(sample_policy)
        
        heading_block = {
            "type": "heading",
            "text": "1. Introduction",
            "spans": [],
            "meta": {}
        }
        
        context = chunker._update_heading_context("", heading_block)
        assert context == "1. Introduction"
        
        # Test sub-heading
        sub_heading_block = {
            "type": "heading", 
            "text": "1.1 Overview",
            "spans": [],
            "meta": {}
        }
        
        context = chunker._update_heading_context(context, sub_heading_block)
        assert "1. Introduction > 1.1 Overview" in context


if __name__ == "__main__":
    pytest.main([__file__])