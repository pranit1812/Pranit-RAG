"""
Sliding window context support for chunk relationships.
"""
from typing import List, Dict, Any, Optional, Tuple
from models.types import Chunk, ChunkMetadata


class ContextWindow:
    """Manages sliding window context for chunk relationships."""
    
    def __init__(self, window_size: int = 1):
        """
        Initialize context window.
        
        Args:
            window_size: Number of adjacent chunks to include on each side
        """
        self.window_size = max(0, window_size)
    
    def expand_chunks_with_context(
        self, 
        chunks: List[Chunk], 
        target_chunk_ids: List[str]
    ) -> List[Chunk]:
        """
        Expand target chunks with sliding window context.
        
        Args:
            chunks: All chunks from the document/project
            target_chunk_ids: IDs of chunks to expand with context
            
        Returns:
            List of chunks including target chunks and their context
        """
        if self.window_size == 0 or not target_chunk_ids:
            return [chunk for chunk in chunks if chunk["id"] in target_chunk_ids]
        
        # Group chunks by document and page for proper adjacency
        doc_page_chunks = self._group_chunks_by_doc_page(chunks)
        
        # Find target chunks and their positions
        target_positions = self._find_target_positions(doc_page_chunks, target_chunk_ids)
        
        # Expand with context
        expanded_chunk_ids = set(target_chunk_ids)
        
        for (doc_id, page_no), chunk_list in doc_page_chunks.items():
            target_indices = target_positions.get((doc_id, page_no), [])
            
            for target_idx in target_indices:
                # Add context chunks
                start_idx = max(0, target_idx - self.window_size)
                end_idx = min(len(chunk_list), target_idx + self.window_size + 1)
                
                for i in range(start_idx, end_idx):
                    expanded_chunk_ids.add(chunk_list[i]["id"])
        
        # Return chunks in original order
        return [chunk for chunk in chunks if chunk["id"] in expanded_chunk_ids]
    
    def get_adjacent_chunks(
        self, 
        chunks: List[Chunk], 
        target_chunk_id: str
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Get adjacent chunks (before and after) for a target chunk.
        
        Args:
            chunks: All chunks from the document/project
            target_chunk_id: ID of the target chunk
            
        Returns:
            Tuple of (previous_chunks, next_chunks)
        """
        if self.window_size == 0:
            return [], []
        
        # Group chunks by document and page
        doc_page_chunks = self._group_chunks_by_doc_page(chunks)
        
        # Find target chunk position
        target_position = None
        target_doc_page = None
        
        for (doc_id, page_no), chunk_list in doc_page_chunks.items():
            for i, chunk in enumerate(chunk_list):
                if chunk["id"] == target_chunk_id:
                    target_position = i
                    target_doc_page = (doc_id, page_no)
                    break
            if target_position is not None:
                break
        
        if target_position is None or target_doc_page is None:
            return [], []
        
        chunk_list = doc_page_chunks[target_doc_page]
        
        # Get previous chunks
        start_idx = max(0, target_position - self.window_size)
        previous_chunks = chunk_list[start_idx:target_position]
        
        # Get next chunks
        end_idx = min(len(chunk_list), target_position + self.window_size + 1)
        next_chunks = chunk_list[target_position + 1:end_idx]
        
        return previous_chunks, next_chunks
    
    def build_contextual_chunk(
        self, 
        target_chunk: Chunk, 
        previous_chunks: List[Chunk], 
        next_chunks: List[Chunk],
        context_mode: str = "summary"
    ) -> Chunk:
        """
        Build a contextual chunk that includes adjacent context.
        
        Args:
            target_chunk: The main chunk
            previous_chunks: Chunks that come before
            next_chunks: Chunks that come after
            context_mode: How to include context ("summary", "full", "titles")
            
        Returns:
            Enhanced chunk with context
        """
        if not previous_chunks and not next_chunks:
            return target_chunk
        
        # Create a copy of the target chunk
        contextual_chunk = target_chunk.copy()
        
        # Build context text based on mode
        context_parts = []
        
        if previous_chunks:
            prev_context = self._build_context_text(previous_chunks, context_mode, "previous")
            if prev_context:
                context_parts.append(f"Previous context: {prev_context}")
        
        if next_chunks:
            next_context = self._build_context_text(next_chunks, context_mode, "next")
            if next_context:
                context_parts.append(f"Following context: {next_context}")
        
        # Add context to chunk text
        if context_parts:
            original_text = target_chunk["text"]
            context_text = "\n\n".join(context_parts)
            contextual_chunk["text"] = f"{context_text}\n\n---\n\n{original_text}"
            
            # Update metadata to indicate context inclusion
            contextual_chunk["metadata"] = contextual_chunk["metadata"].copy()
            contextual_chunk["metadata"]["has_context"] = True
            contextual_chunk["metadata"]["context_window_size"] = self.window_size
            contextual_chunk["metadata"]["context_mode"] = context_mode
        
        return contextual_chunk
    
    def _group_chunks_by_doc_page(self, chunks: List[Chunk]) -> Dict[Tuple[str, int], List[Chunk]]:
        """Group chunks by document ID and page number."""
        grouped = {}
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            doc_id = metadata["doc_id"]
            page_start = metadata["page_start"]
            
            key = (doc_id, page_start)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(chunk)
        
        # Sort chunks within each group by creation order (using chunk ID as proxy)
        for key in grouped:
            grouped[key].sort(key=lambda x: x["id"])
        
        return grouped
    
    def _find_target_positions(
        self, 
        doc_page_chunks: Dict[Tuple[str, int], List[Chunk]], 
        target_chunk_ids: List[str]
    ) -> Dict[Tuple[str, int], List[int]]:
        """Find positions of target chunks within their document/page groups."""
        positions = {}
        target_set = set(target_chunk_ids)
        
        for (doc_id, page_no), chunk_list in doc_page_chunks.items():
            target_indices = []
            for i, chunk in enumerate(chunk_list):
                if chunk["id"] in target_set:
                    target_indices.append(i)
            
            if target_indices:
                positions[(doc_id, page_no)] = target_indices
        
        return positions
    
    def _build_context_text(
        self, 
        chunks: List[Chunk], 
        context_mode: str, 
        direction: str
    ) -> str:
        """Build context text from chunks based on mode."""
        if not chunks:
            return ""
        
        if context_mode == "full":
            # Include full text of context chunks
            texts = [chunk["text"] for chunk in chunks]
            return "\n".join(texts)
        
        elif context_mode == "summary":
            # Include first sentence or line of each chunk
            summaries = []
            for chunk in chunks:
                text = chunk["text"].strip()
                if text:
                    # Get first sentence or first 100 characters
                    first_sentence = text.split('.')[0]
                    if len(first_sentence) > 100:
                        first_sentence = text[:100] + "..."
                    summaries.append(first_sentence)
            return " | ".join(summaries)
        
        elif context_mode == "titles":
            # Extract headings or titles from chunks
            titles = []
            for chunk in chunks:
                # Look for heading-like content
                text = chunk["text"].strip()
                lines = text.split('\n')
                for line in lines[:3]:  # Check first 3 lines
                    line = line.strip()
                    if (len(line) < 100 and 
                        (line.isupper() or 
                         line.startswith('#') or 
                         any(char.isdigit() and char in line[:10] for char in line))):
                        titles.append(line)
                        break
            return " | ".join(titles) if titles else self._build_context_text(chunks, "summary", direction)
        
        else:
            # Default to summary mode
            return self._build_context_text(chunks, "summary", direction)
    
    def calculate_context_relevance(
        self, 
        target_chunk: Chunk, 
        context_chunks: List[Chunk]
    ) -> Dict[str, float]:
        """
        Calculate relevance scores for context chunks.
        
        Args:
            target_chunk: The main chunk
            context_chunks: Potential context chunks
            
        Returns:
            Dictionary mapping chunk IDs to relevance scores
        """
        relevance_scores = {}
        target_text = target_chunk["text"].lower()
        target_words = set(target_text.split())
        
        for chunk in context_chunks:
            chunk_text = chunk["text"].lower()
            chunk_words = set(chunk_text.split())
            
            # Simple word overlap score
            if target_words and chunk_words:
                overlap = len(target_words.intersection(chunk_words))
                union = len(target_words.union(chunk_words))
                jaccard_score = overlap / union if union > 0 else 0.0
            else:
                jaccard_score = 0.0
            
            # Boost score for same content type
            target_type = target_chunk["metadata"]["content_type"]
            chunk_type = chunk["metadata"]["content_type"]
            type_bonus = 0.2 if target_type == chunk_type else 0.0
            
            # Boost score for same document section
            target_section = target_chunk["metadata"].get("section_code")
            chunk_section = chunk["metadata"].get("section_code")
            section_bonus = 0.3 if target_section and target_section == chunk_section else 0.0
            
            final_score = jaccard_score + type_bonus + section_bonus
            relevance_scores[chunk["id"]] = min(1.0, final_score)
        
        return relevance_scores
    
    def filter_context_by_relevance(
        self, 
        target_chunk: Chunk, 
        context_chunks: List[Chunk], 
        min_relevance: float = 0.1
    ) -> List[Chunk]:
        """
        Filter context chunks by relevance to target chunk.
        
        Args:
            target_chunk: The main chunk
            context_chunks: Potential context chunks
            min_relevance: Minimum relevance score to include
            
        Returns:
            Filtered list of relevant context chunks
        """
        relevance_scores = self.calculate_context_relevance(target_chunk, context_chunks)
        
        return [
            chunk for chunk in context_chunks 
            if relevance_scores.get(chunk["id"], 0.0) >= min_relevance
        ]