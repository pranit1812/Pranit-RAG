"""
Core chunking logic for layout-aware document processing.
"""
import uuid
import re
from typing import List, Optional, Dict, Any, Tuple, Union
from io import StringIO

# Optional imports with fallbacks
try:
    from sklearn.cluster import DBSCAN
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple fallback for numpy-like operations
    class np:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def zeros(size, dtype=None):
            return [0] * size

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from models.types import (
    PageParse, Block, Chunk, ChunkMetadata, ChunkPolicy, 
    generate_text_hash
)
from services.classification import ContentClassifier
from chunking.token_counter import TokenCounter
from chunking.table_processor import TableProcessor
from chunking.list_processor import ListProcessor
from chunking.drawing_processor import DrawingProcessor
from chunking.context_window import ContextWindow


class DocumentChunker:
    """Layout-aware document chunker that preserves structure and context."""
    
    def __init__(self, policy: ChunkPolicy, model_name: str = "gpt-4o"):
        """
        Initialize document chunker.
        
        Args:
            policy: Chunking policy configuration
            model_name: Model name for token counting
        """
        self.policy = policy
        self.token_counter = TokenCounter(model_name)
        self.classifier = ContentClassifier()
        self.table_processor = TableProcessor()
        self.list_processor = ListProcessor()
        self.drawing_processor = DrawingProcessor()
        self.context_window = ContextWindow(window_size=1)  # Default window size
    
    def chunk_page(
        self, 
        page: Union[PageParse, Dict[str, Any]], 
        doc_metadata: Dict[str, Any],
        heading_context: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk a single page while preserving layout and structure.
        
        Args:
            page: Parsed page data (PageParse object or dict)
            doc_metadata: Document metadata for chunk creation
            heading_context: Current heading context from previous pages
            
        Returns:
            List of chunks for the page
        """
        # Convert dict to proper format if needed
        if isinstance(page, dict):
            if 'blocks' not in page:
                # Create empty blocks if missing
                page['blocks'] = []
            # Ensure required fields exist
            page.setdefault('page_no', 1)
            page.setdefault('width', 800)
            page.setdefault('height', 600)
            page.setdefault('artifacts_removed', [])
        chunks = []
        current_context = heading_context or ""
        
        # Group blocks by type for specialized processing
        page_blocks = page['blocks'] if isinstance(page, dict) else page.blocks
        blocks_by_type = self._group_blocks_by_type(page_blocks)
        
        # Process headings first to establish context
        for block in blocks_by_type.get("heading", []):
            current_context = self._update_heading_context(current_context, block)
        
        # Process different block types with specialized logic
        chunks.extend(self._process_paragraphs(
            blocks_by_type.get("paragraph", []), 
            page, doc_metadata, current_context
        ))
        
        chunks.extend(self._process_tables(
            blocks_by_type.get("table", []), 
            page, doc_metadata, current_context
        ))
        
        chunks.extend(self._process_lists(
            blocks_by_type.get("list", []), 
            page, doc_metadata, current_context
        ))
        
        chunks.extend(self._process_drawings(
            blocks_by_type.get("drawing", []) + blocks_by_type.get("titleblock", []),
            page, doc_metadata, current_context
        ))
        
        # Process remaining block types
        other_blocks = []
        for block_type in ["figure", "caption", "artifact"]:
            other_blocks.extend(blocks_by_type.get(block_type, []))
        
        if other_blocks:
            chunks.extend(self._process_paragraphs(
                other_blocks, page, doc_metadata, current_context
            ))
        
        return chunks

    def chunk_document(
        self,
        pages: List[Union[PageParse, Dict[str, Any]]],
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk an entire document with token-aware paragraphs across pages.
        
        - Carries heading context across pages
        - Streams paragraph-like content across page boundaries
        - Respects target_tokens for preferred chunk size and max_tokens hard cap
        - Processes tables/lists/drawings per page and flushes paragraph buffer before them
        """
        chunks: List[Chunk] = []
        current_context = ""
        current_chunk_text = ""
        current_blocks: List[Block] = []
        current_page_start: Optional[int] = None
        current_page_end: Optional[int] = None

        def finalize_current_chunk():
            nonlocal current_chunk_text, current_blocks, current_page_start, current_page_end, chunks
            if not current_chunk_text:
                return
            # Create chunk spanning page range
            chunk = self._create_chunk_with_pages(
                current_chunk_text,
                current_blocks,
                current_page_start if current_page_start is not None else 1,
                current_page_end if current_page_end is not None else current_page_start or 1,
                doc_metadata,
                current_context
            )
            chunks.append(chunk)
            # Reset buffer
            current_chunk_text = ""
            current_blocks = []
            current_page_start = None
            current_page_end = None

        for page in pages:
            # Normalize page to dict-like access
            if isinstance(page, dict):
                page_obj = page
            else:
                page_obj = page  # TypedDict-like usage below

            page_no = page_obj.get('page_no', 1)
            page_blocks = page_obj.get('blocks', [])

            # Group and update heading context first
            blocks_by_type = self._group_blocks_by_type(page_blocks)
            for heading_block in blocks_by_type.get("heading", []):
                current_context = self._update_heading_context(current_context, heading_block)

            # Helper to stream paragraph-like blocks
            def stream_paragraph_block(block: Block):
                nonlocal current_chunk_text, current_blocks, current_page_start, current_page_end
                block_text = block.get("text", "").strip()
                if not block_text:
                    return

                # If block alone exceeds max, split by sentences and emit
                if self.token_counter.count_tokens(block_text) > self.policy["max_tokens"]:
                    finalize_current_chunk()
                    split_texts = self.token_counter.split_by_sentences(block_text, self.policy["max_tokens"])
                    for split_text in split_texts:
                        chunk = self._create_chunk_with_pages(
                            split_text,
                            [block],
                            page_no,
                            page_no,
                            doc_metadata,
                            current_context
                        )
                        chunks.append(chunk)
                    return

                # Prefer target-sized chunks: if adding this block would push past target, finalize first
                test_text = self._build_chunk_text(current_chunk_text, block_text, current_context)
                token_count = self.token_counter.count_tokens(test_text)

                if current_chunk_text and token_count > self.policy["target_tokens"]:
                    finalize_current_chunk()
                    # Start new with this block
                    current_chunk_text = block_text
                    current_blocks = [block]
                    current_page_start = page_no
                    current_page_end = page_no
                    return

                # If within max, append to current; otherwise finalize and start new
                if token_count <= self.policy["max_tokens"] and current_chunk_text:
                    current_chunk_text = self._combine_text(current_chunk_text, block_text)
                    current_blocks.append(block)
                    current_page_end = page_no if current_page_start is not None else page_no
                else:
                    # Finalize current if exists (case: empty or would exceed max)
                    finalize_current_chunk()
                    current_chunk_text = block_text
                    current_blocks = [block]
                    current_page_start = page_no
                    current_page_end = page_no

            # Stream paragraphs on this page
            for para_block in blocks_by_type.get("paragraph", []):
                stream_paragraph_block(para_block)

            # Other paragraph-like types that can be appended to narrative text
            other_blocks: List[Block] = []
            for block_type in ["figure", "caption", "artifact"]:
                other_blocks.extend(blocks_by_type.get(block_type, []))
            for other_block in other_blocks:
                stream_paragraph_block(other_block)

            # Before processing structured blocks, flush current paragraph chunk to keep boundaries clean
            if current_chunk_text:
                finalize_current_chunk()

            # Process tables
            table_chunks = self._process_tables(
                blocks_by_type.get("table", []), page_obj, doc_metadata, current_context
            )
            chunks.extend(table_chunks)

            # Process lists
            list_chunks = self._process_lists(
                blocks_by_type.get("list", []), page_obj, doc_metadata, current_context
            )
            chunks.extend(list_chunks)

            # Process drawings/titleblocks
            drawing_chunks = self._process_drawings(
                blocks_by_type.get("drawing", []) + blocks_by_type.get("titleblock", []),
                page_obj,
                doc_metadata,
                current_context
            )
            chunks.extend(drawing_chunks)

        # Final flush after last page
        if current_chunk_text:
            finalize_current_chunk()

        return chunks
    
    def _group_blocks_by_type(self, blocks: List[Block]) -> Dict[str, List[Block]]:
        """Group blocks by their type for specialized processing."""
        grouped = {}
        for block in blocks:
            block_type = block["type"]
            if block_type not in grouped:
                grouped[block_type] = []
            grouped[block_type].append(block)
        return grouped
    
    def _update_heading_context(self, current_context: str, heading_block: Block) -> str:
        """Update heading context with new heading information."""
        heading_text = heading_block["text"].strip()
        
        # Simple heading hierarchy detection based on text patterns
        if re.match(r'^\d+\.?\s', heading_text):  # "1. " or "1 "
            # Top-level heading - replace context
            return heading_text
        elif re.match(r'^\d+\.\d+\.?\s', heading_text):  # "1.1. " or "1.1 "
            # Sub-heading - append to context
            if current_context:
                return f"{current_context} > {heading_text}"
            return heading_text
        else:
            # Other headings - append if not too long
            if current_context and len(current_context) < 200:
                return f"{current_context} > {heading_text}"
            return heading_text
    
    def _process_paragraphs(
        self, 
        blocks: List[Block], 
        page: PageParse, 
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Process paragraph blocks with heading context propagation."""
        if not blocks:
            return []
        
        chunks = []
        current_chunk_text = ""
        current_blocks = []
        
        for block in blocks:
            block_text = block["text"].strip()
            if not block_text:
                continue
            
            # Test if adding this block would exceed token limit
            test_text = self._build_chunk_text(current_chunk_text, block_text, context)
            token_count = self.token_counter.count_tokens(test_text)
            
            if token_count <= self.policy["max_tokens"] and current_chunk_text:
                # Add to current chunk
                current_chunk_text = self._combine_text(current_chunk_text, block_text)
                current_blocks.append(block)
            else:
                # Finalize current chunk if it exists
                if current_chunk_text:
                    chunk = self._create_chunk(
                        current_chunk_text, current_blocks, page, doc_metadata, context
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                if self.token_counter.count_tokens(block_text) > self.policy["max_tokens"]:
                    # Split large block by sentences
                    split_texts = self.token_counter.split_by_sentences(
                        block_text, self.policy["max_tokens"]
                    )
                    for split_text in split_texts:
                        chunk = self._create_chunk(
                            split_text, [block], page, doc_metadata, context
                        )
                        chunks.append(chunk)
                    current_chunk_text = ""
                    current_blocks = []
                else:
                    current_chunk_text = block_text
                    current_blocks = [block]
        
        # Finalize remaining chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                current_chunk_text, current_blocks, page, doc_metadata, context
            )
            chunks.append(chunk)
        
        return chunks
    
    def _process_tables(
        self, 
        blocks: List[Block], 
        page: PageParse, 
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Process table blocks as standalone chunks with structure preservation."""
        if not blocks or not self.policy["preserve_tables"]:
            return self._process_paragraphs(blocks, page, doc_metadata, context)
        
        chunks = []
        
        for block in blocks:
            # Process table to get text, HTML, and CSV representations
            table_text, table_html, csv_content = self.table_processor.process_table_block(block)
            
            if not table_text:
                continue
            
            # Check if table fits in single chunk
            full_text = self._build_chunk_text("", table_text, context)
            token_count = self.token_counter.count_tokens(full_text)
            
            if token_count <= self.policy["max_tokens"]:
                # Single chunk for entire table
                chunk = self._create_chunk(
                    table_text, [block], page, doc_metadata, context, table_html
                )
                # Add CSV content to chunk metadata if available
                if csv_content:
                    chunk["metadata"]["csv_content"] = csv_content
                chunks.append(chunk)
            else:
                # Split table with header repetition
                table_chunks = self._split_table_with_headers_enhanced(
                    table_text, table_html, csv_content, block, page, doc_metadata, context
                )
                chunks.extend(table_chunks)
        
        return chunks
    
    def _split_table_with_headers_enhanced(
        self,
        table_text: str,
        table_html: Optional[str],
        csv_content: Optional[str],
        block: Block,
        page: PageParse,
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Split large table while preserving headers using enhanced table processor."""
        # Use table processor for structured splitting
        table_chunks = self.table_processor.split_table_with_headers(
            table_text, table_html, csv_content, 
            self.policy["max_tokens"], self.token_counter
        )
        
        chunks = []
        for chunk_text, chunk_html, chunk_csv in table_chunks:
            chunk = self._create_chunk(
                chunk_text, [block], page, doc_metadata, context, chunk_html
            )
            # Add CSV content to chunk metadata if available
            if chunk_csv:
                chunk["metadata"]["csv_content"] = chunk_csv
            chunks.append(chunk)
        
        return chunks
    
    def _process_lists(
        self, 
        blocks: List[Block], 
        page: PageParse, 
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Process list blocks with title/intro context preservation."""
        if not blocks or not self.policy["preserve_lists"]:
            return self._process_paragraphs(blocks, page, doc_metadata, context)
        
        chunks = []
        
        for block in blocks:
            # Process list to extract structure
            list_text, intro_lines, list_items = self.list_processor.process_list_block(block)
            
            if not list_text:
                continue
            
            # Check if list fits in single chunk
            full_text = self._build_chunk_text("", list_text, context)
            token_count = self.token_counter.count_tokens(full_text)
            
            if token_count <= self.policy["max_tokens"]:
                # Single chunk for entire list
                chunk = self._create_chunk(
                    list_text, [block], page, doc_metadata, context
                )
                # Add list metadata
                list_type = self.list_processor.detect_list_type(list_items)
                chunk["metadata"]["list_type"] = list_type
                chunk["metadata"]["list_item_count"] = len(list_items)
                chunks.append(chunk)
            else:
                # Split list while preserving structure
                list_chunks = self._split_list_with_context_enhanced(
                    intro_lines, list_items, block, page, doc_metadata, context
                )
                chunks.extend(list_chunks)
        
        return chunks
    
    def _split_list_with_context_enhanced(
        self,
        intro_lines: List[str],
        list_items: List[str],
        block: Block,
        page: PageParse,
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Split large list while preserving intro/title context using enhanced processor."""
        # Use list processor for structured splitting
        list_chunks = self.list_processor.split_list_with_context(
            intro_lines, list_items, self.policy["max_tokens"], self.token_counter
        )
        
        chunks = []
        list_type = self.list_processor.detect_list_type(list_items)
        
        for intro_text, items in list_chunks:
            # Combine intro and items with proper structure
            chunk_text = self.list_processor.preserve_list_structure(intro_text, items)
            
            chunk = self._create_chunk(
                chunk_text, [block], page, doc_metadata, context
            )
            
            # Add list metadata
            chunk["metadata"]["list_type"] = list_type
            chunk["metadata"]["list_item_count"] = len(items)
            chunks.append(chunk)
        
        return chunks
    
    def _process_drawings(
        self, 
        blocks: List[Block], 
        page: PageParse, 
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Process drawing blocks with optional regional clustering."""
        if not blocks:
            return []
        
        # Use drawing processor for enhanced processing
        combined_text, all_bboxes, clustered_regions = self.drawing_processor.process_drawing_page(
            blocks, page, 
            self.policy["drawing_cluster_text"], 
            self.policy["drawing_max_regions"]
        )
        
        if not combined_text:
            return []
        
        # Check if we should use clustering
        should_cluster = (
            self.policy["drawing_cluster_text"] and 
            len(clustered_regions) > 1 and
            self.token_counter.count_tokens(combined_text) > self.policy["target_tokens"]
        )
        
        if should_cluster:
            return self._create_drawing_region_chunks(
                clustered_regions, blocks, page, doc_metadata, context
            )
        else:
            # Single page-level chunk
            chunk = self._create_chunk(
                combined_text, blocks, page, doc_metadata, context
            )
            
            # Add drawing analysis metadata
            drawing_analysis = self.drawing_processor.analyze_drawing_layout(blocks, page)
            chunk["metadata"]["drawing_analysis"] = drawing_analysis
            
            return [chunk]
    
    def _create_drawing_region_chunks(
        self,
        clustered_regions: List[Tuple[str, List[List[float]]]],
        blocks: List[Block],
        page: PageParse,
        doc_metadata: Dict[str, Any],
        context: str
    ) -> List[Chunk]:
        """Create chunks from clustered drawing regions."""
        chunks = []
        drawing_analysis = self.drawing_processor.analyze_drawing_layout(blocks, page)
        
        for region_text, region_bboxes in clustered_regions:
            if not region_text.strip():
                continue
            
            # Create chunk for this region
            chunk = self._create_chunk(
                region_text, blocks, page, doc_metadata, context
            )
            
            # Add region-specific metadata
            chunk["metadata"]["bbox_regions"] = region_bboxes
            chunk["metadata"]["drawing_analysis"] = drawing_analysis
            chunk["metadata"]["is_drawing_region"] = True
            
            chunks.append(chunk)
        
        return chunks
    
    def _build_chunk_text(self, base_text: str, new_text: str, context: str) -> str:
        """Build complete chunk text with context."""
        parts = []
        
        if context:
            parts.append(f"Context: {context}")
        
        if base_text:
            parts.append(base_text)
        
        if new_text:
            parts.append(new_text)
        
        return '\n\n'.join(parts)
    
    def _combine_text(self, text1: str, text2: str) -> str:
        """Combine two text segments appropriately."""
        if not text1:
            return text2
        if not text2:
            return text1
        
        # Add appropriate spacing
        if text1.endswith(('.', '!', '?', ':')):
            return f"{text1} {text2}"
        else:
            return f"{text1}\n{text2}"
    
    def _create_chunk(
        self,
        text: str,
        blocks: List[Block],
        page: PageParse,
        doc_metadata: Dict[str, Any],
        context: str,
        html: Optional[str] = None
    ) -> Chunk:
        """Create a chunk with proper metadata and classification."""
        # Build final text with context
        final_text = self._build_chunk_text("", text, context)
        
        # Generate chunk ID and hash
        chunk_id = str(uuid.uuid4())
        text_hash = generate_text_hash(final_text)
        
        # Count tokens
        token_count = self.token_counter.count_tokens(final_text)
        
        # Classify content and extract metadata
        classification_metadata = {}
        try:
            classification_result = self.classifier.classify_blocks(blocks)
            if isinstance(classification_result, tuple) and len(classification_result) >= 1:
                content_type = classification_result[0]
                classification_metadata["content_confidence"] = (
                    classification_result[1] if len(classification_result) > 1 else None
                )
            else:
                content_type = classification_result
        except Exception:
            content_type = "SpecSection"
        
        # Collect bbox regions
        bbox_regions = [block["bbox"] for block in blocks if block.get("bbox")]
        
        # Determine confidence
        low_conf = any(
            span.get("conf", 1.0) < 0.7 
            for block in blocks 
            for span in block.get("spans", [])
        )
        
        # Create chunk metadata
        metadata = ChunkMetadata(
            project_id=doc_metadata["project_id"],
            doc_id=doc_metadata["doc_id"],
            doc_name=doc_metadata["doc_name"],
            file_type=doc_metadata["file_type"],
            page_start=page["page_no"],
            page_end=page["page_no"],
            content_type=content_type,
            division_code=classification_metadata.get("division_code"),
            division_title=classification_metadata.get("division_title"),
            section_code=classification_metadata.get("section_code"),
            section_title=classification_metadata.get("section_title"),
            discipline=classification_metadata.get("discipline"),
            sheet_number=classification_metadata.get("sheet_number"),
            sheet_title=classification_metadata.get("sheet_title"),
            bbox_regions=bbox_regions,
            low_conf=low_conf
        )
        
        return Chunk(
            id=chunk_id,
            text=final_text,
            html=html,
            metadata=metadata,
            token_count=token_count,
            text_hash=text_hash
        )

    def _create_chunk_with_pages(
        self,
        text: str,
        blocks: List[Block],
        page_start: int,
        page_end: int,
        doc_metadata: Dict[str, Any],
        context: str,
        html: Optional[str] = None
    ) -> Chunk:
        """Create a chunk specifying a page range (for cross-page paragraph chunks)."""
        final_text = self._build_chunk_text("", text, context)
        chunk_id = str(uuid.uuid4())
        text_hash = generate_text_hash(final_text)
        token_count = self.token_counter.count_tokens(final_text)

        content_type = self.classifier.classify_blocks(blocks)
        classification_metadata: Dict[str, Any] = {}
        bbox_regions = [block["bbox"] for block in blocks if block.get("bbox")]
        low_conf = any(
            span.get("conf", 1.0) < 0.7 
            for block in blocks 
            for span in block.get("spans", [])
        )

        metadata = ChunkMetadata(
            project_id=doc_metadata["project_id"],
            doc_id=doc_metadata["doc_id"],
            doc_name=doc_metadata["doc_name"],
            file_type=doc_metadata["file_type"],
            page_start=page_start,
            page_end=page_end,
            content_type=content_type,
            division_code=classification_metadata.get("division_code"),
            division_title=classification_metadata.get("division_title"),
            section_code=classification_metadata.get("section_code"),
            section_title=classification_metadata.get("section_title"),
            discipline=classification_metadata.get("discipline"),
            sheet_number=classification_metadata.get("sheet_number"),
            sheet_title=classification_metadata.get("sheet_title"),
            bbox_regions=bbox_regions,
            low_conf=low_conf
        )

        return Chunk(
            id=chunk_id,
            text=final_text,
            html=html,
            metadata=metadata,
            token_count=token_count,
            text_hash=text_hash
        )
    
    def set_context_window_size(self, window_size: int) -> None:
        """
        Set the context window size for sliding window support.
        
        Args:
            window_size: Number of adjacent chunks to include on each side
        """
        self.context_window = ContextWindow(window_size)
    
    def expand_chunks_with_context(
        self, 
        all_chunks: List[Chunk], 
        target_chunk_ids: List[str]
    ) -> List[Chunk]:
        """
        Expand target chunks with sliding window context.
        
        Args:
            all_chunks: All chunks from the document/project
            target_chunk_ids: IDs of chunks to expand with context
            
        Returns:
            List of chunks including target chunks and their context
        """
        return self.context_window.expand_chunks_with_context(all_chunks, target_chunk_ids)
    
    def get_contextual_chunk(
        self, 
        all_chunks: List[Chunk], 
        target_chunk_id: str,
        context_mode: str = "summary"
    ) -> Optional[Chunk]:
        """
        Get a chunk enhanced with sliding window context.
        
        Args:
            all_chunks: All chunks from the document/project
            target_chunk_id: ID of the target chunk
            context_mode: How to include context ("summary", "full", "titles")
            
        Returns:
            Enhanced chunk with context or None if not found
        """
        # Find target chunk
        target_chunk = None
        for chunk in all_chunks:
            if chunk["id"] == target_chunk_id:
                target_chunk = chunk
                break
        
        if not target_chunk:
            return None
        
        # Get adjacent chunks
        previous_chunks, next_chunks = self.context_window.get_adjacent_chunks(
            all_chunks, target_chunk_id
        )
        
        # Build contextual chunk
        return self.context_window.build_contextual_chunk(
            target_chunk, previous_chunks, next_chunks, context_mode
        )


def chunk_page(
    page: PageParse, 
    policy: ChunkPolicy, 
    doc_metadata: Dict[str, Any],
    heading_context: Optional[str] = None,
    model_name: str = "gpt-4o"
) -> List[Chunk]:
    """
    Convenience function to chunk a single page.
    
    Args:
        page: Parsed page data
        policy: Chunking policy
        doc_metadata: Document metadata
        heading_context: Current heading context
        model_name: Model name for token counting
        
    Returns:
        List of chunks for the page
    """
    chunker = DocumentChunker(policy, model_name)
    return chunker.chunk_page(page, doc_metadata, heading_context)