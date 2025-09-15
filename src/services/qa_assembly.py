"""
QA assembly and citation system for the Construction RAG System.

This module handles context building, citation generation, and LLM integration
for generating answers with traceable sources.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from models.types import (
    Chunk, Hit, ContextPacket, SourceInfo, ProjectContext
)
from chunking.token_counter import TokenCounter
from config import get_config


class ContextBuilder:
    """Builds context packets from retrieved chunks for LLM consumption."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize context builder.
        
        Args:
            model_name: LLM model name for token counting
        """
        self.token_counter = TokenCounter(model_name)
        self.config = get_config()
    
    def build_context(
        self, 
        hits: List[Hit], 
        project_context: ProjectContext,
        max_tokens: int = 8000
    ) -> ContextPacket:
        """
        Build context packet from retrieved chunks with token budget management.
        
        Args:
            hits: Retrieved chunks with scores
            project_context: Project context for query enhancement
            max_tokens: Maximum tokens for context
            
        Returns:
            ContextPacket with chunks, sources, and metadata
        """
        if not hits:
            return ContextPacket(
                chunks=[],
                total_tokens=0,
                sources={},
                project_context=project_context
            )
        
        # Sort hits by score (descending)
        sorted_hits = sorted(hits, key=lambda h: h['score'], reverse=True)
        
        # Build context with intelligent trimming
        selected_chunks = []
        sources = {}
        total_tokens = 0
        source_counter = 1
        
        # Reserve tokens for project context and formatting
        project_context_tokens = self._estimate_project_context_tokens(project_context)
        available_tokens = max_tokens - project_context_tokens - 500  # Buffer for formatting
        
        for hit in sorted_hits:
            chunk = hit['chunk']
            chunk_tokens = chunk['token_count']
            
            # Check if we can fit this chunk
            if total_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                
                # Create source info
                source_key = f"S{source_counter}"
                sources[source_key] = self._extract_source_info(chunk)
                source_counter += 1
            else:
                # Try to fit a trimmed version of the chunk
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 100:  # Only trim if we have reasonable space
                    trimmed_chunk = self._trim_chunk(chunk, remaining_tokens)
                    if trimmed_chunk:
                        selected_chunks.append(trimmed_chunk)
                        total_tokens += trimmed_chunk['token_count']
                        
                        source_key = f"S{source_counter}"
                        sources[source_key] = self._extract_source_info(trimmed_chunk)
                        source_counter += 1
                break
        
        return ContextPacket(
            chunks=selected_chunks,
            total_tokens=total_tokens + project_context_tokens,
            sources=sources,
            project_context=project_context
        )
    
    def _estimate_project_context_tokens(self, project_context: ProjectContext) -> int:
        """Estimate tokens needed for project context."""
        context_text = f"""
        Project: {project_context['project_name']}
        Type: {project_context['project_type']}
        Description: {project_context['description']}
        Key Systems: {', '.join(project_context['key_systems'])}
        Disciplines: {', '.join(project_context['disciplines_involved'])}
        Summary: {project_context['summary']}
        """
        return self.token_counter.count_tokens(context_text)
    
    def _extract_source_info(self, chunk: Chunk) -> SourceInfo:
        """Extract source information from chunk metadata."""
        metadata = chunk['metadata']
        
        # Determine page/sheet number
        page_number = metadata['page_start']
        if metadata['page_end'] != metadata['page_start']:
            page_number = f"{metadata['page_start']}-{metadata['page_end']}"
        
        return SourceInfo(
            doc_name=metadata['doc_name'],
            page_number=page_number,
            sheet_number=metadata.get('sheet_number')
        )
    
    def _trim_chunk(self, chunk: Chunk, max_tokens: int) -> Optional[Chunk]:
        """
        Trim chunk to fit within token budget while preserving meaning.
        
        Args:
            chunk: Chunk to trim
            max_tokens: Maximum tokens allowed
            
        Returns:
            Trimmed chunk or None if cannot be meaningfully trimmed
        """
        if max_tokens < 50:  # Don't trim to very small sizes
            return None
        
        original_text = chunk['text']
        
        # Try to preserve structure by trimming at sentence boundaries
        sentences = self.token_counter.split_by_sentences(original_text, max_tokens)
        if not sentences:
            return None
        
        # Take the first chunk from sentence splitting
        trimmed_text = sentences[0]
        
        # Add ellipsis to indicate truncation
        if len(trimmed_text) < len(original_text):
            trimmed_text += "..."
        
        # Create trimmed chunk
        trimmed_chunk = chunk.copy()
        trimmed_chunk['text'] = trimmed_text
        trimmed_chunk['token_count'] = self.token_counter.count_tokens(trimmed_text)
        
        return trimmed_chunk
    
    def optimize_context_for_llm(self, context_packet: ContextPacket) -> str:
        """
        Optimize context packet for LLM consumption with clear formatting.
        
        Args:
            context_packet: Context packet to format
            
        Returns:
            Formatted context string for LLM
        """
        if not context_packet['chunks']:
            return "No relevant context found."
        
        # Build formatted context
        context_parts = []
        
        # Add project context
        project_context = context_packet['project_context']
        context_parts.append(f"""PROJECT CONTEXT:
Project: {project_context['project_name']}
Type: {project_context['project_type']}
Description: {project_context['description']}
Key Systems: {', '.join(project_context['key_systems'])}
Disciplines: {', '.join(project_context['disciplines_involved'])}
Summary: {project_context['summary']}
""")
        
        # Add retrieved chunks with source references
        context_parts.append("RELEVANT DOCUMENTS:")
        
        source_keys = list(context_packet['sources'].keys())
        for i, chunk in enumerate(context_packet['chunks']):
            source_key = source_keys[i] if i < len(source_keys) else f"S{i+1}"
            source_info = context_packet['sources'].get(source_key, {})
            
            # Format source header
            doc_name = source_info.get('doc_name', 'Unknown Document')
            page_info = source_info.get('page_number', 'Unknown Page')
            sheet_info = source_info.get('sheet_number')
            
            if sheet_info:
                source_header = f"[{source_key}] {doc_name} - Sheet {sheet_info}, Page {page_info}"
            else:
                source_header = f"[{source_key}] {doc_name} - Page {page_info}"
            
            # Add content type and metadata if available
            metadata = chunk['metadata']
            content_type = metadata.get('content_type', '')
            if content_type:
                source_header += f" ({content_type})"
            
            # Add division/section info for specs
            if metadata.get('division_code') and metadata.get('division_title'):
                source_header += f" - Division {metadata['division_code']}: {metadata['division_title']}"
            
            context_parts.append(f"\n{source_header}:")
            
            # Add chunk content
            chunk_text = chunk['text']
            if chunk.get('html') and metadata.get('content_type') == 'Table':
                # For tables, include both HTML and text
                context_parts.append(f"Table Structure:\n{chunk['html']}\n")
                context_parts.append(f"Table Content:\n{chunk_text}")
            else:
                context_parts.append(chunk_text)
        
        return "\n".join(context_parts)
    
    def enhance_query_with_project_context(
        self, 
        query: str, 
        project_context: ProjectContext
    ) -> str:
        """
        Enhance user query with project context for better retrieval.
        
        Args:
            query: Original user query
            project_context: Project context for enhancement
            
        Returns:
            Enhanced query with project context
        """
        # Extract key terms from project context
        project_terms = []
        
        # Add project type terms
        project_type = project_context.get('project_type', '')
        if project_type:
            project_terms.extend(project_type.lower().split())
        
        # Add key systems
        key_systems = project_context.get('key_systems', [])
        for system in key_systems:
            project_terms.extend(system.lower().split())
        
        # Add disciplines
        disciplines = project_context.get('disciplines_involved', [])
        project_terms.extend([d.lower() for d in disciplines])
        
        # Remove duplicates and common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        unique_terms = list(set(project_terms) - common_words)
        
        # Enhance query with relevant project terms
        if unique_terms:
            # Add most relevant terms (limit to avoid query bloat)
            relevant_terms = unique_terms[:5]
            enhanced_query = f"{query} {' '.join(relevant_terms)}"
        else:
            enhanced_query = query
        
        return enhanced_query


class CitationGenerator:
    """Generates citations and manages source references for answers."""
    
    def __init__(self):
        """Initialize citation generator."""
        pass
    
    def generate_citations(self, context_packet: ContextPacket) -> Dict[str, str]:
        """
        Generate citation mappings from context packet.
        
        Args:
            context_packet: Context packet with sources
            
        Returns:
            Dictionary mapping source keys to formatted citations
        """
        citations = {}
        
        for source_key, source_info in context_packet['sources'].items():
            citation = self._format_citation(source_info)
            citations[source_key] = citation
        
        return citations
    
    def _format_citation(self, source_info: SourceInfo) -> str:
        """
        Format a single source into a citation string.
        
        Args:
            source_info: Source information
            
        Returns:
            Formatted citation string
        """
        doc_name = source_info['doc_name']
        page_number = source_info['page_number']
        sheet_number = source_info.get('sheet_number')
        
        if sheet_number:
            return f"{doc_name}, Sheet {sheet_number}, Page {page_number}"
        else:
            return f"{doc_name}, Page {page_number}"
    
    def extract_citations_from_response(self, response: str) -> List[str]:
        """
        Extract citation references from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of citation keys found in response (e.g., ['S1', 'S2'])
        """
        # Pattern to match citation references like [S1], [S2], etc.
        citation_pattern = r'\[S(\d+)\]'
        matches = re.findall(citation_pattern, response)
        
        # Convert to full citation keys
        citations = [f"S{match}" for match in matches]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        return unique_citations
    
    def validate_citations(
        self, 
        response: str, 
        available_sources: Dict[str, SourceInfo]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all citations in response have corresponding sources.
        
        Args:
            response: LLM response with citations
            available_sources: Available source mappings
            
        Returns:
            Tuple of (is_valid, list_of_missing_citations)
        """
        cited_sources = self.extract_citations_from_response(response)
        missing_citations = []
        
        for citation in cited_sources:
            if citation not in available_sources:
                missing_citations.append(citation)
        
        is_valid = len(missing_citations) == 0
        return is_valid, missing_citations
    
    def create_source_snippets(
        self, 
        context_packet: ContextPacket,
        max_snippet_length: int = 300
    ) -> Dict[str, Dict[str, str]]:
        """
        Create expandable source snippets for verification.
        
        Args:
            context_packet: Context packet with chunks
            max_snippet_length: Maximum length for snippet preview
            
        Returns:
            Dictionary mapping source keys to snippet info
        """
        snippets = {}
        source_keys = list(context_packet['sources'].keys())
        
        for i, chunk in enumerate(context_packet['chunks']):
            source_key = source_keys[i] if i < len(source_keys) else f"S{i+1}"
            
            # Create preview snippet
            full_text = chunk['text']
            if len(full_text) <= max_snippet_length:
                preview = full_text
                is_truncated = False
            else:
                # Find a good breaking point (sentence or word boundary)
                preview = full_text[:max_snippet_length]
                last_sentence = preview.rfind('.')
                last_space = preview.rfind(' ')
                
                if last_sentence > max_snippet_length * 0.7:
                    preview = preview[:last_sentence + 1]
                elif last_space > max_snippet_length * 0.7:
                    preview = preview[:last_space]
                
                preview += "..."
                is_truncated = True
            
            # Include metadata for context
            metadata = chunk['metadata']
            content_type = metadata.get('content_type', 'Unknown')
            
            snippets[source_key] = {
                'preview': preview,
                'full_text': full_text,
                'is_truncated': is_truncated,
                'content_type': content_type,
                'html': chunk.get('html'),  # For tables
                'metadata': {
                    'division_code': metadata.get('division_code'),
                    'division_title': metadata.get('division_title'),
                    'section_code': metadata.get('section_code'),
                    'section_title': metadata.get('section_title'),
                    'discipline': metadata.get('discipline'),
                    'sheet_title': metadata.get('sheet_title'),
                    'low_conf': metadata.get('low_conf', False)
                }
            }
        
        return snippets
    
    def format_citations_for_display(
        self, 
        citations: Dict[str, str],
        used_citations: List[str]
    ) -> str:
        """
        Format citations for display at the end of answers.
        
        Args:
            citations: All available citations
            used_citations: Citations actually used in the response
            
        Returns:
            Formatted citation list
        """
        if not used_citations:
            return ""
        
        citation_lines = []
        citation_lines.append("\n**Sources:**")
        
        for citation_key in used_citations:
            if citation_key in citations:
                citation_text = citations[citation_key]
                citation_lines.append(f"- {citation_key}: {citation_text}")
        
        return "\n".join(citation_lines)
    
    def ensure_citation_traceability(
        self, 
        response: str,
        context_packet: ContextPacket
    ) -> Dict[str, Any]:
        """
        Ensure complete traceability from response to source documents.
        
        Args:
            response: LLM response with citations
            context_packet: Original context packet
            
        Returns:
            Traceability information including source paths and page locations
        """
        citations = self.extract_citations_from_response(response)
        traceability = {}
        
        source_keys = list(context_packet['sources'].keys())
        
        for citation in citations:
            if citation in context_packet['sources']:
                source_info = context_packet['sources'][citation]
                
                # Find corresponding chunk for additional metadata
                chunk_index = source_keys.index(citation) if citation in source_keys else -1
                chunk = context_packet['chunks'][chunk_index] if 0 <= chunk_index < len(context_packet['chunks']) else None
                
                traceability[citation] = {
                    'source_info': source_info,
                    'chunk_metadata': chunk['metadata'] if chunk else None,
                    'bbox_regions': chunk['metadata']['bbox_regions'] if chunk else [],
                    'confidence': 'low' if chunk and chunk['metadata'].get('low_conf') else 'high'
                }
        
        return traceability


import openai
import time
import logging
from typing import Union
try:
    from utils.error_handling import (
        safe_api_call,
        retry_with_backoff,
        performance_monitor,
        log_error_with_context,
        APIError,
        create_user_friendly_error
    )
except ImportError:
    from utils.error_handling import (
        safe_api_call,
        retry_with_backoff,
        performance_monitor,
        log_error_with_context,
        APIError,
        create_user_friendly_error
    )


class LLMService:
    """LLM service for answer generation with OpenAI integration and comprehensive error handling."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize LLM service.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (if None, uses environment variable)
        """
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting parameters
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0
    
    @safe_api_call(max_retries=3, backoff_factor=1.0, timeout=120)
    def generate_answer(
        self, 
        query: str,
        context_packet: ContextPacket,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with context injection and comprehensive error handling.
        
        Args:
            query: User query
            context_packet: Context packet with retrieved chunks
            temperature: LLM temperature for response generation
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        with performance_monitor(f"generate_answer_query"):
            return self._generate_answer_internal(query, context_packet, temperature)
    
    def _generate_answer_internal(
        self, 
        query: str,
        context_packet: ContextPacket,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Internal answer generation with error handling."""
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if not context_packet or not context_packet.get('chunks'):
                self.logger.warning("No context chunks provided, generating answer without context")
                return self._generate_fallback_answer(query)
            
            # Build context for LLM
            context_builder = ContextBuilder(self.model_name)
            formatted_context = context_builder.optimize_context_for_llm(context_packet)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context and query
            user_prompt = self._create_user_prompt(query, formatted_context)
            
            # Validate prompt length
            if len(user_prompt) > 100000:  # Conservative limit
                self.logger.warning("User prompt is very long, truncating context")
                user_prompt = self._truncate_prompt(user_prompt, max_length=100000)
            
            # Generate response with retry logic
            response = self._call_llm_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature
            )
            
            if not response or not response.strip():
                raise ValueError("LLM returned empty response")
            
            # Process response and extract citations
            citation_generator = CitationGenerator()
            citations = citation_generator.generate_citations(context_packet)
            used_citations = citation_generator.extract_citations_from_response(response)
            
            # Validate citations
            is_valid, missing_citations = citation_generator.validate_citations(
                response, context_packet['sources']
            )
            
            if missing_citations:
                self.logger.warning(f"Response contains invalid citations: {missing_citations}")
            
            # Format final answer with citations
            formatted_citations = citation_generator.format_citations_for_display(
                citations, used_citations
            )
            
            final_answer = response + formatted_citations
            
            # Create source snippets for verification
            source_snippets = citation_generator.create_source_snippets(context_packet)
            
            # Generate traceability information
            traceability = citation_generator.ensure_citation_traceability(
                response, context_packet
            )
            
            return {
                'answer': final_answer,
                'raw_answer': response,
                'citations': citations,
                'used_citations': used_citations,
                'source_snippets': source_snippets,
                'traceability': traceability,
                'context_tokens': context_packet['total_tokens'],
                'is_citations_valid': is_valid,
                'missing_citations': missing_citations
            }
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for construction RAG assistant."""
        return """You are a construction document analysis assistant. Your role is to provide accurate, helpful answers based on construction documents including specifications, drawings, and ITB (Invitation to Bid) documents.

INSTRUCTIONS:
1. Answer questions using ONLY the provided document context
2. Always cite your sources using the format [S1], [S2], etc. that correspond to the source references
3. If information is not available in the provided context, clearly state this
4. For technical questions, be precise and include relevant details like section numbers, specifications, or drawing references
5. When referencing tables, mention both the content and structure when relevant
6. If a source has low confidence (from OCR), mention this uncertainty
7. Organize your answer clearly with proper formatting
8. Do not make assumptions beyond what is explicitly stated in the documents

CITATION RULES:
- Use [S1], [S2], etc. to reference sources throughout your answer
- Each factual claim should have a citation
- Multiple sources can support the same point: [S1, S2]
- Place citations immediately after the relevant information

Remember: You are helping construction professionals make informed decisions based on their project documents. Accuracy and traceability are critical."""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """
        Create user prompt with query and context.
        
        Args:
            query: User query
            context: Formatted context from documents
            
        Returns:
            Complete user prompt
        """
        return f"""Based on the following construction project documents, please answer this question:

QUESTION: {query}

DOCUMENT CONTEXT:
{context}

Please provide a comprehensive answer with proper citations."""
    
    def _call_llm_with_retry(
        self, 
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1
    ) -> str:
        """
        Call LLM with retry logic for rate limiting and errors.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt with context
            temperature: Response temperature
            
        Returns:
            LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000  # Reasonable limit for answers
                )
                
                return response.choices[0].message.content.strip()
                
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    raise e
                    
            except openai.APIError as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.warning(f"API error, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                else:
                    raise e
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in LLM call: {str(e)}")
                raise e
        
        raise Exception("Max retries exceeded for LLM call")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create error response structure.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            'answer': f"I apologize, but I encountered an error while processing your question: {error_message}",
            'raw_answer': "",
            'citations': {},
            'used_citations': [],
            'source_snippets': {},
            'traceability': {},
            'context_tokens': 0,
            'is_citations_valid': False,
            'missing_citations': [],
            'error': error_message
        }


class QAAssemblyService:
    """Main QA assembly service that coordinates context building, citation, and LLM services."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize QA assembly service.
        
        Args:
            model_name: LLM model name
            api_key: OpenAI API key
        """
        self.context_builder = ContextBuilder(model_name)
        self.citation_generator = CitationGenerator()
        self.llm_service = LLMService(model_name, api_key)
        self.config = get_config()
    
    def generate_answer(
        self,
        query: str,
        hits: List[Hit],
        project_context: ProjectContext,
        max_context_tokens: int = 8000,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate complete answer with context building, LLM generation, and citation.
        
        Args:
            query: User query
            hits: Retrieved chunks with scores
            project_context: Project context for enhancement
            max_context_tokens: Maximum tokens for context
            temperature: LLM temperature
            
        Returns:
            Complete answer with citations and metadata
        """
        # Build context packet
        context_packet = self.context_builder.build_context(
            hits, project_context, max_context_tokens
        )
        
        # Generate answer using LLM
        result = self.llm_service.generate_answer(
            query, context_packet, temperature
        )
        
        # Add context packet info to result
        result['context_packet'] = context_packet
        result['query'] = query
        
        return result
    
    def enhance_query(self, query: str, project_context: ProjectContext) -> str:
        """
        Enhance query with project context.
        
        Args:
            query: Original query
            project_context: Project context
            
        Returns:
            Enhanced query
        """
        return self.context_builder.enhance_query_with_project_context(
            query, project_context
        )