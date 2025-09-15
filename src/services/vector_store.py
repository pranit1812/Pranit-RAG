"""
Vector store implementation with Chroma integration for the Construction RAG System.
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Third-party imports
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from models.types import Chunk, Hit, ChunkMetadata
from config import get_config


logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """
        Insert or update chunks in the vector store.
        
        Args:
            chunks: List of chunks to upsert
        """
        pass
    
    @abstractmethod
    def query(self, vector: List[float], k: int, where: Optional[Dict] = None) -> List[Hit]:
        """
        Query the vector store for similar vectors.
        
        Args:
            vector: Query vector
            k: Number of results to return
            where: Optional metadata filters
            
        Returns:
            List of hits with scores and chunks
        """
        pass
    
    @abstractmethod
    def delete_project(self, project_id: str) -> None:
        """
        Delete all chunks for a project.
        
        Args:
            project_id: Project ID to delete
        """
        pass
    
    @abstractmethod
    def get_chunk_count(self, project_id: str) -> int:
        """
        Get the number of chunks for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Number of chunks
        """
        pass
    
    @abstractmethod
    def chunk_exists(self, text_hash: str) -> bool:
        """
        Check if a chunk with the given text hash exists.
        
        Args:
            text_hash: Text hash to check
            
        Returns:
            True if chunk exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Get chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of chunks
        """
        pass


class ChromaVectorStore(VectorStore):
    """Chroma-based vector store implementation."""
    
    def __init__(self, data_dir: str, collection_prefix: str = "construction_rag"):
        """
        Initialize Chroma vector store.
        
        Args:
            data_dir: Directory for storing Chroma data
            collection_prefix: Prefix for collection names
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB package not available. Install with: pip install chromadb")
        
        self.data_dir = Path(data_dir)
        self.collection_prefix = collection_prefix
        self._collections: Dict[str, Any] = {}
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"Initialized Chroma vector store at: {self.data_dir}")
    
    def _get_collection_name(self, project_id: str) -> str:
        """Get collection name for a project."""
        return f"{self.collection_prefix}_{project_id}"
    
    def _get_or_create_collection(self, project_id: str):
        """Get or create a collection for a project."""
        collection_name = self._get_collection_name(project_id)
        
        if collection_name not in self._collections:
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"Retrieved existing collection: {collection_name}")
            except Exception:
                # Create new collection if it doesn't exist
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"project_id": project_id}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self._collections[collection_name] = collection
        
        return self._collections[collection_name]
    
    def _chunk_to_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Convert chunk metadata to Chroma-compatible format.
        
        Args:
            chunk: Chunk to extract metadata from
            
        Returns:
            Metadata dictionary for Chroma
        """
        metadata = chunk["metadata"].copy()
        
        # Convert lists to JSON strings for Chroma compatibility
        if "bbox_regions" in metadata:
            metadata["bbox_regions"] = json.dumps(metadata["bbox_regions"])
        
        # Add chunk-level metadata
        metadata.update({
            "chunk_id": chunk["id"],
            "text_hash": chunk["text_hash"],
            "token_count": chunk["token_count"],
            "has_html": chunk.get("html") is not None
        })
        
        # Ensure all values are JSON-serializable
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ""
            elif isinstance(value, bool):
                metadata[key] = str(value).lower()
            elif not isinstance(value, (str, int, float)):
                metadata[key] = str(value)
        
        return metadata
    
    def _metadata_to_chunk_metadata(self, metadata: Dict[str, Any]) -> ChunkMetadata:
        """
        Convert Chroma metadata back to ChunkMetadata format.
        
        Args:
            metadata: Metadata from Chroma
            
        Returns:
            ChunkMetadata object
        """
        # Parse bbox_regions back from JSON
        bbox_regions = []
        if "bbox_regions" in metadata and metadata["bbox_regions"]:
            try:
                bbox_regions = json.loads(metadata["bbox_regions"])
            except (json.JSONDecodeError, TypeError):
                bbox_regions = []
        
        # Convert string booleans back to bool
        low_conf = metadata.get("low_conf", "false").lower() == "true"
        
        # Handle optional fields
        def get_optional_str(key: str) -> Optional[str]:
            value = metadata.get(key, "")
            return value if value else None
        
        return ChunkMetadata(
            project_id=metadata["project_id"],
            doc_id=metadata["doc_id"],
            doc_name=metadata["doc_name"],
            file_type=metadata["file_type"],
            page_start=int(metadata["page_start"]),
            page_end=int(metadata["page_end"]),
            content_type=metadata["content_type"],
            division_code=get_optional_str("division_code"),
            division_title=get_optional_str("division_title"),
            section_code=get_optional_str("section_code"),
            section_title=get_optional_str("section_title"),
            discipline=get_optional_str("discipline"),
            sheet_number=get_optional_str("sheet_number"),
            sheet_title=get_optional_str("sheet_title"),
            bbox_regions=bbox_regions,
            low_conf=low_conf
        )
    
    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """
        Insert or update chunks in the vector store.
        
        Args:
            chunks: List of chunks to upsert
        """
        if not chunks:
            return
        
        # Group chunks by project
        chunks_by_project: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            project_id = chunk["metadata"]["project_id"]
            if project_id not in chunks_by_project:
                chunks_by_project[project_id] = []
            chunks_by_project[project_id].append(chunk)
        
        # Process each project separately
        for project_id, project_chunks in chunks_by_project.items():
            collection = self._get_or_create_collection(project_id)
            
            # Prepare data for Chroma
            ids = [chunk["id"] for chunk in project_chunks]
            documents = [chunk["text"] for chunk in project_chunks]
            metadatas = [self._chunk_to_metadata(chunk) for chunk in project_chunks]
            
            # Note: We don't provide embeddings here as they should be generated
            # by the embedding service and passed separately, or Chroma can generate them
            
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Upserted {len(project_chunks)} chunks for project {project_id}")
                
            except Exception as e:
                logger.error(f"Error upserting chunks for project {project_id}: {e}")
                raise
    
    def upsert_chunks_with_embeddings(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Insert or update chunks with pre-computed embeddings.
        
        Args:
            chunks: List of chunks to upsert
            embeddings: Corresponding embeddings for each chunk
        """
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        
        # Group chunks by project
        chunks_by_project: Dict[str, List[tuple]] = {}
        for chunk, embedding in zip(chunks, embeddings):
            project_id = chunk["metadata"]["project_id"]
            if project_id not in chunks_by_project:
                chunks_by_project[project_id] = []
            chunks_by_project[project_id].append((chunk, embedding))
        
        # Process each project separately
        for project_id, project_data in chunks_by_project.items():
            collection = self._get_or_create_collection(project_id)
            
            # Prepare data for Chroma
            project_chunks, project_embeddings = zip(*project_data)
            ids = [chunk["id"] for chunk in project_chunks]
            documents = [chunk["text"] for chunk in project_chunks]
            metadatas = [self._chunk_to_metadata(chunk) for chunk in project_chunks]
            
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=list(project_embeddings)
                )
                logger.info(f"Upserted {len(project_chunks)} chunks with embeddings for project {project_id}")
                
            except Exception as e:
                logger.error(f"Error upserting chunks with embeddings for project {project_id}: {e}")
                raise
    
    def query(self, vector: List[float], k: int, where: Optional[Dict] = None) -> List[Hit]:
        """
        Query the vector store for similar vectors.
        
        Args:
            vector: Query vector
            k: Number of results to return
            where: Optional metadata filters
            
        Returns:
            List of hits with scores and chunks
        """
        # Extract project_id from where clause
        if not where or "project_id" not in where:
            raise ValueError("project_id must be specified in where clause")
        
        project_id = where["project_id"]
        collection = self._get_or_create_collection(project_id)
        
        # Remove project_id from where clause for Chroma query
        chroma_where = {k: v for k, v in where.items() if k != "project_id"}
        
        try:
            results = collection.query(
                query_embeddings=[vector],
                n_results=k,
                where=chroma_where if chroma_where else None
            )
            
            hits = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Reconstruct chunk from stored data
                    metadata = self._metadata_to_chunk_metadata(results["metadatas"][0][i])
                    
                    chunk = Chunk(
                        id=chunk_id,
                        text=results["documents"][0][i],
                        html=None,  # HTML not stored in vector store
                        metadata=metadata,
                        token_count=int(results["metadatas"][0][i].get("token_count", 0)),
                        text_hash=results["metadatas"][0][i]["text_hash"]
                    )
                    
                    hit = Hit(
                        id=chunk_id,
                        score=float(results["distances"][0][i]),
                        chunk=chunk
                    )
                    hits.append(hit)
            
            logger.info(f"Retrieved {len(hits)} hits for project {project_id}")
            return hits
            
        except Exception as e:
            logger.error(f"Error querying vector store for project {project_id}: {e}")
            raise
    
    def delete_project(self, project_id: str) -> None:
        """
        Delete all chunks for a project.
        
        Args:
            project_id: Project ID to delete
        """
        collection_name = self._get_collection_name(project_id)
        
        try:
            # Delete the entire collection
            self.client.delete_collection(name=collection_name)
            
            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logger.info(f"Deleted collection for project {project_id}")
            
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            raise
    
    def get_chunk_count(self, project_id: str) -> int:
        """
        Get the number of chunks for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Number of chunks
        """
        try:
            collection = self._get_or_create_collection(project_id)
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting chunk count for project {project_id}: {e}")
            return 0
    
    def chunk_exists(self, text_hash: str) -> bool:
        """
        Check if a chunk with the given text hash exists.
        
        Args:
            text_hash: Text hash to check
            
        Returns:
            True if chunk exists, False otherwise
        """
        # This is a simplified implementation that checks across all collections
        # In practice, you might want to limit this to a specific project
        try:
            collections = self.client.list_collections()
            for collection_info in collections:
                collection = self.client.get_collection(collection_info.name)
                results = collection.get(where={"text_hash": text_hash})
                if results["ids"]:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking if chunk exists with hash {text_hash}: {e}")
            return False
    
    def get_chunks_by_hashes(self, project_id: str, text_hashes: List[str]) -> List[Chunk]:
        """
        Get chunks by their text hashes.
        
        Args:
            project_id: Project ID
            text_hashes: List of text hashes to retrieve
            
        Returns:
            List of chunks
        """
        if not text_hashes:
            return []
        
        collection = self._get_or_create_collection(project_id)
        chunks = []
        
        try:
            # Query for chunks with matching text hashes
            for text_hash in text_hashes:
                results = collection.get(where={"text_hash": text_hash})
                
                if results["ids"]:
                    for i, chunk_id in enumerate(results["ids"]):
                        metadata = self._metadata_to_chunk_metadata(results["metadatas"][i])
                        
                        chunk = Chunk(
                            id=chunk_id,
                            text=results["documents"][i],
                            html=None,  # HTML not stored in vector store
                            metadata=metadata,
                            token_count=int(results["metadatas"][i].get("token_count", 0)),
                            text_hash=results["metadatas"][i]["text_hash"]
                        )
                        chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks by hash for project {project_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks by hashes for project {project_id}: {e}")
            return []
    
    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Get chunks by their IDs across all projects.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of chunks
        """
        if not chunk_ids:
            return []
        
        chunks = []
        
        try:
            # Search across all collections for the chunk IDs
            collections = self.client.list_collections()
            
            for collection_info in collections:
                collection = self.client.get_collection(collection_info.name)
                
                # Get chunks by IDs from this collection
                results = collection.get(ids=chunk_ids)
                
                if results["ids"]:
                    for i, chunk_id in enumerate(results["ids"]):
                        metadata = self._metadata_to_chunk_metadata(results["metadatas"][i])
                        
                        chunk = Chunk(
                            id=chunk_id,
                            text=results["documents"][i],
                            html=None,  # HTML not stored in vector store
                            metadata=metadata,
                            token_count=int(results["metadatas"][i].get("token_count", 0)),
                            text_hash=results["metadatas"][i]["text_hash"]
                        )
                        chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} chunks by IDs")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks by IDs: {e}")
            return []


class VectorStoreFactory:
    """Factory for creating vector stores based on configuration."""
    
    @staticmethod
    def create_store(store_type: str = "chroma", **kwargs) -> VectorStore:
        """
        Create a vector store based on configuration.
        
        Args:
            store_type: Type of vector store ("chroma")
            **kwargs: Additional arguments for the store
            
        Returns:
            VectorStore instance
        """
        config = get_config()
        
        if store_type == "chroma":
            data_dir = kwargs.get("data_dir", config.app.data_dir)
            collection_prefix = kwargs.get("collection_prefix", "construction_rag")
            return ChromaVectorStore(data_dir=data_dir, collection_prefix=collection_prefix)
        
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")


def create_vector_store(**kwargs) -> VectorStore:
    """
    Convenience function to create a vector store.
    
    Args:
        **kwargs: Arguments passed to VectorStoreFactory.create_store
        
    Returns:
        VectorStore instance
    """
    return VectorStoreFactory.create_store(**kwargs)