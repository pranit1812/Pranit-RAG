"""
BM25 keyword search implementation using Whoosh for the Construction RAG System.
"""
import logging
import os
import re
import string
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time

from whoosh import fields, index
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Query
from whoosh.searching import Results
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer, Filter
from whoosh.filedb.filestore import FileStorage

from models.types import Chunk, Hit, ChunkMetadata
from config import get_config


logger = logging.getLogger(__name__)


class ConstructionFilter(Filter):
    """
    Custom filter for construction documents with domain-specific preprocessing.
    """
    
    def __init__(self):
        """Initialize construction-specific filter."""
        # Construction-specific terms that should not be stemmed
        self.construction_terms = {
            'hvac', 'electrical', 'plumbing', 'structural', 'mechanical',
            'architectural', 'civil', 'geotechnical', 'environmental',
            'concrete', 'steel', 'masonry', 'wood', 'aluminum', 'copper',
            'pvc', 'cpvc', 'hdpe', 'fiberglass', 'insulation', 'drywall',
            'roofing', 'flooring', 'ceiling', 'foundation', 'framing',
            'rebar', 'conduit', 'ductwork', 'piping', 'wiring', 'fixtures',
            'specifications', 'drawings', 'blueprints', 'schedules',
            'details', 'sections', 'elevations', 'plans', 'layouts'
        }
        
        # Common construction abbreviations
        self.abbreviations = {
            'dwg': 'drawing',
            'spec': 'specification',
            'elev': 'elevation',
            'sect': 'section',
            'det': 'detail',
            'sched': 'schedule',
            'typ': 'typical',
            'sim': 'similar',
            'req': 'required',
            'min': 'minimum',
            'max': 'maximum',
            'dia': 'diameter',
            'thk': 'thickness',
            'ht': 'height',
            'wd': 'width',
            'lg': 'length'
        }
    
    def __call__(self, tokens):
        """
        Process tokens with construction-specific preprocessing.
        
        Args:
            tokens: Input token stream
            
        Yields:
            Processed tokens
        """
        for token in tokens:
            # Expand abbreviations
            text = token.text.lower()
            if text in self.abbreviations:
                token.text = self.abbreviations[text]
            
            yield token


def create_construction_analyzer():
    """
    Create a construction-specific analyzer.
    
    Returns:
        Whoosh analyzer for construction documents
    """
    from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
    
    # Create base analyzer with stemming
    base_analyzer = StemmingAnalyzer()
    
    # Add construction-specific filter
    construction_filter = ConstructionFilter()
    
    # Combine filters
    return base_analyzer | construction_filter


@dataclass
class BM25SearchResult:
    """Result from BM25 search operation."""
    hits: List[Hit]
    total_found: int
    search_time: float
    query_terms: List[str]


class BM25Index:
    """
    BM25 keyword search index using Whoosh for construction documents.
    """
    
    def __init__(self, index_dir: str):
        """
        Initialize BM25 index.
        
        Args:
            index_dir: Directory to store the Whoosh index
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Define schema for construction documents
        construction_analyzer = create_construction_analyzer()
        self.schema = fields.Schema(
            # Core fields
            chunk_id=fields.ID(stored=True, unique=True),
            text=fields.TEXT(analyzer=construction_analyzer, stored=False),
            html=fields.TEXT(analyzer=construction_analyzer, stored=False),
            
            # Metadata fields for filtering
            project_id=fields.ID(stored=True),
            doc_id=fields.ID(stored=True),
            doc_name=fields.TEXT(stored=True),
            file_type=fields.KEYWORD(stored=True),
            page_start=fields.NUMERIC(stored=True),
            page_end=fields.NUMERIC(stored=True),
            content_type=fields.KEYWORD(stored=True),
            division_code=fields.KEYWORD(stored=True),
            division_title=fields.TEXT(stored=True),
            section_code=fields.KEYWORD(stored=True),
            section_title=fields.TEXT(stored=True),
            discipline=fields.KEYWORD(stored=True),
            sheet_number=fields.KEYWORD(stored=True),
            sheet_title=fields.TEXT(stored=True),
            low_conf=fields.BOOLEAN(stored=True),
            
            # Combined search field for better relevance
            searchable_text=fields.TEXT(analyzer=construction_analyzer, stored=False)
        )
        
        self._index = None
        self._ensure_index()
        
        logger.info(f"Initialized BM25 index at {self.index_dir}")
    
    def close(self):
        """Close the index to release file handles."""
        if self._index:
            self._index.close()
            self._index = None
    
    def _ensure_index(self):
        """Ensure the Whoosh index exists and is accessible."""
        try:
            if index.exists_in(str(self.index_dir)):
                self._index = index.open_dir(str(self.index_dir))
                logger.debug("Opened existing Whoosh index")
            else:
                self._index = index.create_in(str(self.index_dir), self.schema)
                logger.info("Created new Whoosh index")
        except Exception as e:
            logger.error(f"Error initializing Whoosh index: {e}")
            raise
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Index a list of chunks for keyword search.
        
        Args:
            chunks: List of chunks to index
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
        
        start_time = time.time()
        
        try:
            writer = self._index.writer()
            
            for chunk in chunks:
                # Prepare searchable text combining main text and metadata
                searchable_parts = [chunk["text"]]
                
                # Add HTML content if available (for tables)
                if chunk.get("html"):
                    # Strip HTML tags for search
                    html_text = re.sub(r'<[^>]+>', ' ', chunk["html"])
                    searchable_parts.append(html_text)
                
                # Add relevant metadata text
                metadata = chunk["metadata"]
                if metadata.get("division_title"):
                    searchable_parts.append(metadata["division_title"])
                if metadata.get("section_title"):
                    searchable_parts.append(metadata["section_title"])
                if metadata.get("sheet_title"):
                    searchable_parts.append(metadata["sheet_title"])
                
                searchable_text = " ".join(searchable_parts)
                
                # Add document to index (ensure all fields are strings)
                writer.add_document(
                    chunk_id=str(chunk["id"]),
                    text=str(chunk["text"]),
                    html=str(chunk.get("html", "")),
                    project_id=str(metadata.get("project_id", "")),
                    doc_id=str(metadata.get("doc_id", "")),
                    doc_name=str(metadata.get("doc_name", "")),
                    file_type=str(metadata.get("file_type", "")),
                    page_start=str(metadata.get("page_start", "")),
                    page_end=str(metadata.get("page_end", "")),
                    content_type=str(metadata.get("content_type", "")),
                    division_code=metadata.get("division_code", ""),
                    division_title=metadata.get("division_title", ""),
                    section_code=metadata.get("section_code", ""),
                    section_title=metadata.get("section_title", ""),
                    discipline=metadata.get("discipline", ""),
                    sheet_number=metadata.get("sheet_number", ""),
                    sheet_title=metadata.get("sheet_title", ""),
                    low_conf=metadata["low_conf"],
                    searchable_text=searchable_text
                )
            
            writer.commit()
            
            index_time = time.time() - start_time
            logger.info(f"Indexed {len(chunks)} chunks in {index_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise
    
    def search(
        self,
        query: str,
        project_id: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> BM25SearchResult:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query string
            project_id: Project identifier for filtering
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            BM25SearchResult with hits and metadata
        """
        start_time = time.time()
        
        try:
            with self._index.searcher() as searcher:
                # Create query parser for multiple fields
                parser = MultifieldParser(
                    ["searchable_text", "text", "division_title", "section_title", "sheet_title"],
                    schema=self._index.schema
                )
                
                # Parse the query
                parsed_query = parser.parse(query)
                
                # Build filter query for project and additional filters
                filter_query = self._build_filter_query(project_id, filters)
                
                # Combine search query with filters
                if filter_query:
                    from whoosh.query import And
                    final_query = And([parsed_query, filter_query])
                else:
                    final_query = parsed_query
                
                # Perform search
                results = searcher.search(final_query, limit=k)
                
                # Convert results to Hit format
                hits = self._convert_results_to_hits(results, searcher)
                
                # Extract query terms for debugging
                query_terms = self._extract_query_terms(parsed_query)
                
                search_time = time.time() - start_time
                
                logger.info(
                    f"BM25 search completed: {len(hits)} hits in {search_time:.3f}s"
                )
                
                return BM25SearchResult(
                    hits=hits,
                    total_found=len(results),
                    search_time=search_time,
                    query_terms=query_terms
                )
                
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            raise
    
    def _build_filter_query(
        self,
        project_id: str,
        filters: Optional[Dict[str, Any]]
    ) -> Optional[Query]:
        """
        Build filter query from project ID and additional filters.
        
        Args:
            project_id: Project identifier
            filters: Optional additional filters
            
        Returns:
            Combined filter query or None
        """
        from whoosh.query import Term, And, Or
        
        filter_parts = [Term("project_id", project_id)]
        
        if not filters:
            return And(filter_parts)
        
        # Content type filters
        if filters.get("content_types"):
            content_filters = [Term("content_type", ct) for ct in filters["content_types"]]
            filter_parts.append(Or(content_filters))
        
        # Division code filters
        if filters.get("division_codes"):
            division_filters = [Term("division_code", dc) for dc in filters["division_codes"]]
            filter_parts.append(Or(division_filters))
        
        # Discipline filters
        if filters.get("disciplines"):
            discipline_filters = [Term("discipline", d) for d in filters["disciplines"]]
            filter_parts.append(Or(discipline_filters))
        
        # Confidence level filter
        if filters.get("low_conf_only") is not None:
            filter_parts.append(Term("low_conf", str(filters["low_conf_only"]).lower()))
        
        # Document name filters
        if filters.get("doc_names"):
            doc_filters = [Term("doc_name", dn) for dn in filters["doc_names"]]
            filter_parts.append(Or(doc_filters))
        
        # File type filters
        if filters.get("file_types"):
            file_type_filters = [Term("file_type", ft) for ft in filters["file_types"]]
            filter_parts.append(Or(file_type_filters))
        
        return And(filter_parts) if len(filter_parts) > 1 else filter_parts[0]
    
    def _convert_results_to_hits(self, results: Results, searcher) -> List[Hit]:
        """
        Convert Whoosh search results to Hit format.
        
        Args:
            results: Whoosh search results
            searcher: Whoosh searcher instance
            
        Returns:
            List of Hit objects
        """
        hits = []
        
        for result in results:
            # Reconstruct chunk metadata with safe field access
            def safe_get(field_name):
                """Safely get field value, returning None if empty or missing."""
                try:
                    value = result[field_name]
                    return value if value else None
                except KeyError:
                    return None
            
            metadata: ChunkMetadata = {
                "project_id": result["project_id"],
                "doc_id": result["doc_id"],
                "doc_name": result["doc_name"],
                "file_type": result["file_type"],
                "page_start": result["page_start"],
                "page_end": result["page_end"],
                "content_type": result["content_type"],
                "division_code": safe_get("division_code"),
                "division_title": safe_get("division_title"),
                "section_code": safe_get("section_code"),
                "section_title": safe_get("section_title"),
                "discipline": safe_get("discipline"),
                "sheet_number": safe_get("sheet_number"),
                "sheet_title": safe_get("sheet_title"),
                "bbox_regions": [],  # Not stored in BM25 index
                "low_conf": result["low_conf"]
            }
            
            # Create minimal chunk (we don't store full text in BM25 index)
            # The actual chunk will need to be retrieved from vector store
            chunk: Chunk = {
                "id": result["chunk_id"],
                "text": "",  # Will be filled by hybrid retriever
                "html": None,
                "metadata": metadata,
                "token_count": 0,  # Will be filled by hybrid retriever
                "text_hash": ""  # Will be filled by hybrid retriever
            }
            
            hit: Hit = {
                "id": result["chunk_id"],
                "score": result.score,
                "chunk": chunk
            }
            
            hits.append(hit)
        
        return hits
    
    def _extract_query_terms(self, query: Query) -> List[str]:
        """
        Extract terms from parsed query for debugging.
        
        Args:
            query: Parsed Whoosh query
            
        Returns:
            List of query terms
        """
        try:
            # Simple extraction of terms from query
            query_str = str(query)
            # Remove field prefixes and extract terms
            terms = re.findall(r'\b\w+\b', query_str)
            # Filter out field names and common words
            field_names = {"searchable_text", "text", "division_title", "section_title", "sheet_title"}
            filtered_terms = [term for term in terms if term.lower() not in field_names]
            return list(set(filtered_terms))  # Remove duplicates
        except Exception:
            return []
    
    def delete_project(self, project_id: str) -> None:
        """
        Delete all chunks for a project from the index.
        
        Args:
            project_id: Project identifier
        """
        try:
            writer = self._index.writer()
            writer.delete_by_term("project_id", project_id)
            writer.commit()
            
            logger.info(f"Deleted project {project_id} from BM25 index")
            
        except Exception as e:
            logger.error(f"Error deleting project from BM25 index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            with self._index.searcher() as searcher:
                doc_count = searcher.doc_count()
                field_names = list(self._index.schema.names())
                
                return {
                    "document_count": doc_count,
                    "field_names": field_names,
                    "index_dir": str(self.index_dir)
                }
        except Exception as e:
            logger.error(f"Error getting BM25 index stats: {e}")
            return {}


def create_bm25_index(project_id: str) -> BM25Index:
    """
    Create a BM25 index for a specific project.
    
    Args:
        project_id: Project identifier
        
    Returns:
        BM25Index instance
    """
    config = get_config()
    index_dir = Path(config.app.data_dir) / project_id / "bm25_index"
    return BM25Index(str(index_dir))