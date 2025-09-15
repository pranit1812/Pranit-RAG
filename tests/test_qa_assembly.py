"""
Tests for QA assembly and citation system.
"""
import pytest
from unittest.mock import Mock, patch
from src.services.qa_assembly import (
    ContextBuilder, CitationGenerator, LLMService, QAAssemblyService
)
from src.models.types import (
    Chunk, Hit, ChunkMetadata, ProjectContext, ContextPacket, SourceInfo
)


@pytest.fixture
def sample_project_context():
    """Sample project context for testing."""
    return ProjectContext(
        project_name="Test Office Building",
        description="A modern office building project",
        project_type="Commercial Office Building",
        location="Downtown",
        key_systems=["HVAC", "Electrical", "Plumbing"],
        disciplines_involved=["A", "S", "M", "E", "P"],
        summary="Modern office building with advanced HVAC and electrical systems"
    )


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        Chunk(
            id="chunk1",
            text="The HVAC system shall include variable air volume units with energy recovery ventilation.",
            html=None,
            metadata=ChunkMetadata(
                project_id="test_project",
                doc_id="spec1",
                doc_name="HVAC Specifications",
                file_type="pdf",
                page_start=15,
                page_end=15,
                content_type="SpecSection",
                division_code="23",
                division_title="Heating, Ventilating and Air Conditioning (HVAC)",
                section_code="23 05 00",
                section_title="Common Work Results for HVAC",
                discipline=None,
                sheet_number=None,
                sheet_title=None,
                bbox_regions=[[100, 200, 500, 250]],
                low_conf=False
            ),
            token_count=20,
            text_hash="hash1"
        ),
        Chunk(
            id="chunk2",
            text="Electrical panel schedule shows 480V main distribution panel with 120/208V sub-panels.",
            html=None,
            metadata=ChunkMetadata(
                project_id="test_project",
                doc_id="drawing1",
                doc_name="Electrical Drawings",
                file_type="pdf",
                page_start=5,
                page_end=5,
                content_type="Drawing",
                division_code=None,
                division_title=None,
                section_code=None,
                section_title=None,
                discipline="E",
                sheet_number="E-101",
                sheet_title="Electrical Panel Schedule",
                bbox_regions=[[50, 100, 400, 300]],
                low_conf=False
            ),
            token_count=18,
            text_hash="hash2"
        )
    ]


@pytest.fixture
def sample_hits(sample_chunks):
    """Sample hits for testing."""
    return [
        Hit(id="chunk1", score=0.95, chunk=sample_chunks[0]),
        Hit(id="chunk2", score=0.87, chunk=sample_chunks[1])
    ]


class TestContextBuilder:
    """Test context building functionality."""
    
    def test_build_context_basic(self, sample_hits, sample_project_context):
        """Test basic context building."""
        builder = ContextBuilder()
        context_packet = builder.build_context(
            sample_hits, sample_project_context, max_tokens=1000
        )
        
        assert len(context_packet['chunks']) == 2
        assert len(context_packet['sources']) == 2
        assert 'S1' in context_packet['sources']
        assert 'S2' in context_packet['sources']
        assert context_packet['project_context'] == sample_project_context
        assert context_packet['total_tokens'] > 0
    
    def test_build_context_empty_hits(self, sample_project_context):
        """Test context building with empty hits."""
        builder = ContextBuilder()
        context_packet = builder.build_context(
            [], sample_project_context, max_tokens=1000
        )
        
        assert len(context_packet['chunks']) == 0
        assert len(context_packet['sources']) == 0
        assert context_packet['total_tokens'] == 0
    
    def test_build_context_token_limit(self, sample_hits, sample_project_context):
        """Test context building with tight token limit."""
        builder = ContextBuilder()
        
        # Set very low token limit to force trimming
        context_packet = builder.build_context(
            sample_hits, sample_project_context, max_tokens=100
        )
        
        # Should still have at least one chunk, but possibly trimmed
        assert len(context_packet['chunks']) >= 1
        assert context_packet['total_tokens'] <= 100
    
    def test_extract_source_info(self, sample_chunks):
        """Test source information extraction."""
        builder = ContextBuilder()
        source_info = builder._extract_source_info(sample_chunks[0])
        
        assert source_info['doc_name'] == "HVAC Specifications"
        assert source_info['page_number'] == 15
        assert source_info['sheet_number'] is None
        
        # Test with sheet number
        source_info2 = builder._extract_source_info(sample_chunks[1])
        assert source_info2['sheet_number'] == "E-101"
    
    def test_optimize_context_for_llm(self, sample_hits, sample_project_context):
        """Test context optimization for LLM."""
        builder = ContextBuilder()
        context_packet = builder.build_context(
            sample_hits, sample_project_context, max_tokens=1000
        )
        
        formatted_context = builder.optimize_context_for_llm(context_packet)
        
        assert "PROJECT CONTEXT:" in formatted_context
        assert "Test Office Building" in formatted_context
        assert "RELEVANT DOCUMENTS:" in formatted_context
        assert "[S1]" in formatted_context
        assert "[S2]" in formatted_context
        assert "HVAC Specifications" in formatted_context
    
    def test_enhance_query_with_project_context(self, sample_project_context):
        """Test query enhancement with project context."""
        builder = ContextBuilder()
        original_query = "What are the electrical requirements?"
        
        enhanced_query = builder.enhance_query_with_project_context(
            original_query, sample_project_context
        )
        
        assert original_query in enhanced_query
        # Should contain some project-specific terms
        assert len(enhanced_query) > len(original_query)


class TestCitationGenerator:
    """Test citation generation functionality."""
    
    def test_generate_citations(self, sample_hits, sample_project_context):
        """Test citation generation."""
        builder = ContextBuilder()
        context_packet = builder.build_context(
            sample_hits, sample_project_context, max_tokens=1000
        )
        
        generator = CitationGenerator()
        citations = generator.generate_citations(context_packet)
        
        assert len(citations) == 2
        assert 'S1' in citations
        assert 'S2' in citations
        assert "HVAC Specifications, Page 15" in citations['S1']
        assert "Electrical Drawings, Sheet E-101, Page 5" in citations['S2']
    
    def test_extract_citations_from_response(self):
        """Test citation extraction from response."""
        generator = CitationGenerator()
        response = "The HVAC system [S1] requires electrical connections [S2]. Additional info [S1]."
        
        citations = generator.extract_citations_from_response(response)
        
        assert citations == ['S1', 'S2']  # Should remove duplicates
    
    def test_validate_citations(self):
        """Test citation validation."""
        generator = CitationGenerator()
        response = "Info from [S1] and [S2] and [S3]."
        available_sources = {'S1': {}, 'S2': {}}
        
        is_valid, missing = generator.validate_citations(response, available_sources)
        
        assert not is_valid
        assert missing == ['S3']
    
    def test_create_source_snippets(self, sample_hits, sample_project_context):
        """Test source snippet creation."""
        builder = ContextBuilder()
        context_packet = builder.build_context(
            sample_hits, sample_project_context, max_tokens=1000
        )
        
        generator = CitationGenerator()
        snippets = generator.create_source_snippets(context_packet, max_snippet_length=50)
        
        assert len(snippets) == 2
        assert 'S1' in snippets
        assert 'S2' in snippets
        
        # Check snippet structure
        snippet = snippets['S1']
        assert 'preview' in snippet
        assert 'full_text' in snippet
        assert 'content_type' in snippet
        assert snippet['content_type'] == 'SpecSection'
    
    def test_format_citations_for_display(self):
        """Test citation formatting for display."""
        generator = CitationGenerator()
        citations = {
            'S1': "Document 1, Page 5",
            'S2': "Document 2, Page 10"
        }
        used_citations = ['S1', 'S2']
        
        formatted = generator.format_citations_for_display(citations, used_citations)
        
        assert "**Sources:**" in formatted
        assert "S1: Document 1, Page 5" in formatted
        assert "S2: Document 2, Page 10" in formatted


class TestLLMService:
    """Test LLM service functionality."""
    
    @patch('openai.OpenAI')
    def test_generate_answer_success(self, mock_openai, sample_hits, sample_project_context):
        """Test successful answer generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The HVAC system [S1] includes VAV units."
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Build context packet
        builder = ContextBuilder()
        context_packet = builder.build_context(
            sample_hits, sample_project_context, max_tokens=1000
        )
        
        # Test LLM service
        llm_service = LLMService()
        result = llm_service.generate_answer("What is the HVAC system?", context_packet)
        
        assert 'answer' in result
        assert 'citations' in result
        assert 'used_citations' in result
        assert result['used_citations'] == ['S1']
        assert result['is_citations_valid'] is True
    
    @patch('openai.OpenAI')
    def test_generate_answer_with_retry(self, mock_openai):
        """Test answer generation with retry on rate limit."""
        # Mock rate limit error then success
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit", response=None, body=None),
            Mock(choices=[Mock(message=Mock(content="Success"))])
        ]
        mock_openai.return_value = mock_client
        
        llm_service = LLMService()
        llm_service.base_delay = 0.01  # Speed up test
        
        # Should succeed after retry
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = llm_service._call_llm_with_retry("system", "user")
            assert result == "Success"


class TestQAAssemblyService:
    """Test complete QA assembly service."""
    
    @patch('openai.OpenAI')
    def test_generate_answer_complete(self, mock_openai, sample_hits, sample_project_context):
        """Test complete answer generation workflow."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The HVAC system [S1] includes VAV units with ERV [S1]."
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test complete service
        qa_service = QAAssemblyService()
        result = qa_service.generate_answer(
            "What is the HVAC system?",
            sample_hits,
            sample_project_context
        )
        
        assert 'answer' in result
        assert 'context_packet' in result
        assert 'citations' in result
        assert 'source_snippets' in result
        assert 'traceability' in result
        assert result['query'] == "What is the HVAC system?"
    
    def test_enhance_query(self, sample_project_context):
        """Test query enhancement."""
        qa_service = QAAssemblyService()
        enhanced = qa_service.enhance_query(
            "What are the requirements?",
            sample_project_context
        )
        
        assert "What are the requirements?" in enhanced
        assert len(enhanced) > len("What are the requirements?")


if __name__ == "__main__":
    pytest.main([__file__])