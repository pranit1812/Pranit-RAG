"""
Document processing service for Streamlit integration.
Handles multi-file upload and processing through the RAG pipeline.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


def process_uploaded_documents(
    saved_files: List[Dict], 
    project_path: Path,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Process uploaded documents through the RAG pipeline.
    
    Args:
        saved_files: List of saved file information with keys: name, path, size
        project_path: Path to the project directory
        progress_callback: Optional callback for progress updates
        
    Returns:
        Processing result dictionary
    """
    try:
        project_id = project_path.name
        
        # Initialize processing status
        processing_status = {
            "total_files": len(saved_files),
            "completed_files": 0,
            "failed_files": 0,
            "processed_files": 0,
            "current_file": None,
            "status": "initializing",
            "progress": 0.0,
            "start_time": datetime.now(),
            "files": {},
            "documents_indexed": 0,
            "chunks_created": 0,
            "errors": []
        }
        
        if progress_callback:
            progress_callback(processing_status)
        
        # Use the existing file processor system directly
        try:
            from services.file_processor import ProcessingPipeline, FileUpload, ProcessingStatus
            
            # Create processing pipeline for this project
            pipeline = ProcessingPipeline(project_id)
            logger.info(f"Processing pipeline initialized for project: {project_id}")
            
        except Exception as e:
            processing_status["status"] = "failed"
            processing_status["errors"].append(f"Pipeline initialization error: {str(e)}")
            logger.error(f"Pipeline initialization failed: {e}")
            return processing_status
        
        processing_status["status"] = "processing"
        if progress_callback:
            progress_callback(processing_status)
        
        # Process each file
        logger.info(f"Starting to process {len(saved_files)} files")
        
        for i, file_info in enumerate(saved_files):
            try:
                file_path = Path(file_info["path"])
                filename = file_info["name"]
                
                logger.info(f"Processing file {i+1}/{len(saved_files)}: {filename}")
                
                processing_status["current_file"] = filename
                processing_status["progress"] = (i + 0.5) / len(saved_files)
                
                if progress_callback:
                    progress_callback(processing_status)
                
                # Create FileUpload object and process through pipeline
                file_upload = FileUpload(
                    filename=filename,
                    file_path=file_path,
                    file_size=file_info["size"],
                    file_type=file_path.suffix[1:] if file_path.suffix else "unknown",
                    upload_time=datetime.now(),
                    status=ProcessingStatus.PENDING
                )
                
                def file_progress_callback(progress, status):
                    processing_status["files"][filename] = {
                        "progress": progress,
                        "current_status": status
                    }
                    if progress_callback:
                        progress_callback(processing_status)
                
                # Use our working extraction and chunking approach
                try:
                    # Extract document
                    chunks = pipeline._extract_document(file_upload)
                    
                    if chunks:
                        # Save chunks manually (bypass complex indexing for now)
                        import json
                        chunks_file = project_path / "chunks.jsonl"
                        
                        # Append to existing chunks file
                        with open(chunks_file, 'a', encoding='utf-8') as f:
                            for chunk in chunks:
                                # Convert chunk to dict if needed
                                if hasattr(chunk, '__dict__'):
                                    chunk_dict = {
                                        'id': chunk.id,
                                        'text': chunk.text,
                                        'metadata': chunk.metadata.__dict__ if hasattr(chunk.metadata, '__dict__') else chunk.metadata,
                                        'token_count': chunk.token_count,
                                        'text_hash': chunk.text_hash
                                    }
                                else:
                                    chunk_dict = chunk
                                
                                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
                        
                        result = {
                            "success": True,
                            "error": None,
                            "chunks_created": len(chunks),
                            "pages_processed": 1
                        }
                        logger.info(f"Successfully created {len(chunks)} chunks for {filename}")
                    else:
                        result = {
                            "success": False,
                            "error": "No chunks created from document",
                            "chunks_created": 0,
                            "pages_processed": 0
                        }
                        
                except Exception as e:
                    logger.error(f"Processing failed for {filename}: {e}")
                    result = {
                        "success": False,
                        "error": str(e),
                        "chunks_created": 0,
                        "pages_processed": 0
                    }
                
                if result["success"]:
                    processing_status["completed_files"] += 1
                    processing_status["processed_files"] += 1
                    processing_status["chunks_created"] += result["chunks_created"]
                    processing_status["documents_indexed"] += 1
                    
                    processing_status["files"][filename] = {
                        "status": "completed",
                        "chunks_created": result["chunks_created"],
                        "pages_processed": result["pages_processed"]
                    }
                else:
                    processing_status["failed_files"] += 1
                    error_msg = f"Failed to process {filename}: {result.get('error', 'Unknown error')}"
                    processing_status["errors"].append(error_msg)
                    
                    processing_status["files"][filename] = {
                        "status": "failed",
                        "error": result.get("error", "Unknown error")
                    }
                
                # Update progress
                processing_status["progress"] = (i + 1) / len(saved_files)
                if progress_callback:
                    progress_callback(processing_status)
                
            except Exception as e:
                processing_status["failed_files"] += 1
                error_msg = f"Error processing {filename}: {str(e)}"
                processing_status["errors"].append(error_msg)
                
                processing_status["files"][filename] = {
                    "status": "failed",
                    "error": str(e)
                }
                logger.error(error_msg)
        
        # Finalize processing
        processing_status["status"] = "completed"
        processing_status["progress"] = 1.0
        
        if progress_callback:
            progress_callback(processing_status)
        
        return processing_status
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        processing_status = {
            "total_files": len(saved_files),
            "completed_files": 0,
            "failed_files": len(saved_files),
            "status": "failed",
            "errors": [f"Processing failed: {str(e)}"],
            "chunks_created": 0
        }
        return processing_status


def query_project_documents(
    project_id: str,
    project_path: Path,
    query: str,
    top_k: int = 5,
    use_vision: bool = False
) -> Dict[str, Any]:
    """
    Query documents in a project using the RAG system.
    
    Args:
        project_id: Project identifier
        project_path: Path to project directory
        query: User query
        top_k: Number of results to return
        use_vision: Whether to use vision assistance
        
    Returns:
        Query result with answer and sources
    """
    try:
        # Load chunks
        chunks_file = project_path / "chunks.jsonl"
        if not chunks_file.exists() or chunks_file.stat().st_size == 0:
            return {
                "success": False,
                "error": "No processed documents found. Please process documents first.",
                "answer": "",
                "sources": []
            }
        import json
        chunk_list: List[Dict[str, Any]] = []
        id_to_chunk: Dict[str, Dict[str, Any]] = {}
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                chunk_list.append(c)
                id_to_chunk[c["id"]] = c

        # Normalize chunk metadata types for indexing
        def _norm_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (list, tuple)):
                return _norm_str(val[0] if val else "")
            return str(val)
        for c in chunk_list:
            m = c.get("metadata", {})
            # Ensure content_type is a string
            m["content_type"] = _norm_str(m.get("content_type", "SpecSection"))
            # Coerce other optional string fields
            for key in [
                "division_code","division_title","section_code","section_title",
                "discipline","sheet_number","sheet_title","doc_name","file_type","doc_id","project_id"
            ]:
                if key in m:
                    m[key] = _norm_str(m.get(key))
            c["metadata"] = m

        # Ensure BM25 index exists and is populated
        from services.bm25_search import create_bm25_index
        bm25 = create_bm25_index(project_id)
        # Simple heuristic: if index is empty or first run, index chunks now
        try:
            stats = bm25.get_stats()
            needs_index = (stats.get("document_count", 0) == 0)
        except Exception:
            needs_index = True
        if needs_index:
            bm25.index_chunks(chunk_list)

        # Run keyword search
        bm25_result = bm25.search(query=query, project_id=project_id, k=max(top_k, 5))

        # Enrich hits with full chunk data
        hits: List[Dict[str, Any]] = []
        for hit in bm25_result.hits[:top_k]:
            full = id_to_chunk.get(hit["id"]) or hit.get("chunk") or {}
            if full:
                hits.append({
                    "id": hit["id"],
                    "score": hit.get("score", 0.0),
                    "chunk": full
                })

        if not hits:
            # Fallback: simple keyword overlap scoring over chunk texts
            import re
            terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
            scored = []
            for c in chunk_list:
                text = (c.get("text") or "").lower()
                score = sum(text.count(t) for t in terms)
                if score > 0:
                    scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, c in scored[:top_k]:
                hits.append({
                    "id": c["id"],
                    "score": float(score),
                    "chunk": c
                })
            # If still empty, return a gentle message
            if not hits:
                return {
                    "success": True,
                    "answer": "I couldn't find relevant information for your query in the processed documents.",
                    "sources": [],
                    "retrieved_chunks": []
                }

        # Build project context (simple default for now)
        # Load project context if available
        project_context = {
            "project_name": project_id,
            "description": "Construction project documents",
            "project_type": "General Construction Project",
            "location": None,
            "key_systems": [],
            "disciplines_involved": [],
            "summary": "Auto context not yet generated."
        }
        try:
            ctx_file = project_path / "project_context.md"
            if ctx_file.exists():
                content = ctx_file.read_text(encoding='utf-8')
                # Very light parse for title and summary
                import re
                title = re.search(r"^#\s+(.*)$", content, re.MULTILINE)
                summary_section = re.search(r"##\s*Project Summary\s*(.*)$", content, re.DOTALL)
                if title:
                    project_context["project_name"] = title.group(1).strip()
                if summary_section:
                    project_context["summary"] = summary_section.group(1).strip()[:800]
        except Exception:
            pass

        # Generate answer using QA assembly (LLM)
        from services.qa_assembly import QAAssemblyService
        qa = QAAssemblyService()
        qa_result = qa.generate_answer(query, hits, project_context)

        # Optional: vision enhancement if enabled and available
        try:
            from config import get_config
            cfg = get_config()
            if getattr(cfg.vision, 'enabled', False):
                from services.vision import create_vision_assistant
                va = create_vision_assistant(
                    cache_dir=project_path / "images",
                    api_key=None,
                    config={
                        "enabled": True,
                        "max_images": cfg.vision.max_images,
                        "resolution_scale": cfg.vision.resolution_scale,
                    },
                )
                ctx_packet = qa_result.get("context_packet")
                if ctx_packet:
                    vision_text = va.enhance_answer_with_vision(
                        query=query,
                        context_packet=ctx_packet,
                        project_storage_dir=project_path
                    )
                    if vision_text:
                        qa_result["answer"] += "\n\n" + vision_text
        except Exception:
            pass

        # Convert sources for UI
        sources = []
        for idx, h in enumerate(hits, 1):
            meta = h["chunk"]["metadata"]
            sources.append({
                "id": f"S{idx}",
                "doc_name": meta.get("doc_name", "Unknown"),
                "page_number": meta.get("page_start", "?"),
                "sheet_number": meta.get("sheet_number"),
                "content_type": meta.get("content_type", "Unknown"),
                "text": h["chunk"].get("text", "")
            })

        return {
            "success": True,
            "answer": qa_result.get("answer", ""),
            "sources": sources,
            "retrieved_chunks": hits
        }

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            "success": False,
            "error": f"Query failed: {str(e)}",
            "answer": "",
            "sources": []
        }


# ---------------------------
# Project context generation
# ---------------------------
def generate_project_context_from_chunks(
    project_id: str,
    project_path: Path,
    max_samples: int = 300,
    per_type_limit: int = 60
) -> Dict[str, Any]:
    """Generate a concise project context by sampling representative chunks and
    asking the QA assembly to summarize core attributes.

    Returns a structured context dict compatible with the Streamlit UI.
    """
    try:
        chunks_file = project_path / "chunks.jsonl"
        if not chunks_file.exists() or chunks_file.stat().st_size == 0:
            return {
                "success": False,
                "error": "No chunks found. Process documents first.",
            }

        import json
        from collections import defaultdict
        import random

        # Load chunks
        chunk_list: List[Dict[str, Any]] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    c = json.loads(s)
                except Exception:
                    continue
                chunk_list.append(c)

        # Group by content type for diverse sampling
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in chunk_list:
            ct = c.get("metadata", {}).get("content_type", "Unknown")
            # Normalize content_type to a hashable string
            if isinstance(ct, (list, tuple)):
                ct = ct[0] if ct else "Unknown"
            ct = str(ct)
            by_type[ct].append(c)

        sampled: List[Dict[str, Any]] = []
        random.seed(1337)
        for ct, items in by_type.items():
            random.shuffle(items)
            sampled.extend(items[:per_type_limit])
        if len(sampled) > max_samples:
            random.shuffle(sampled)
            sampled = sampled[:max_samples]

        # Build snippets text
        def _norm(s: Optional[str]) -> str:
            return (s or "").strip()

        snippets: List[str] = []
        for c in sampled:
            m = c.get("metadata", {})
            page = m.get("page_number") or m.get("page_start")
            header_parts = [
                _norm(m.get("division_code")),
                _norm(m.get("division_title")),
                _norm(m.get("section_code")),
                _norm(m.get("sheet_number")),
            ]
            header = " | ".join([h for h in header_parts if h])
            prefix = f"[{str(m.get('content_type','?'))}] {m.get('doc_name','?')} pg {page}"
            if header:
                prefix += f" — {header}"
            text = _norm(c.get("text", ""))
            if text:
                snippets.append(f"- {prefix}: {text[:800]}")

        if not snippets:
            return {"success": False, "error": "No textual snippets available to summarize."}

        bundle = "\n".join(snippets)

        # Use QA assembly to summarize
        try:
            from services.qa_assembly import QAAssemblyService
            qa = QAAssemblyService()
        except Exception:
            return {"success": False, "error": "QA assembly service unavailable."}

        system_instruction = (
            "You are a construction project analyst. From mixed construction documents, "
            "extract a concise project context with: project_type, location (if any), "
            "key_systems (HVAC, Electrical, Plumbing, Fire Protection, Structural, etc.), "
            "disciplines_involved, and a 4-6 sentence summary describing scope and notable requirements. "
            "Use only information present in the snippets."
        )

        summarization_query = (
            "Summarize the project using the snippets. Return JSON with keys: "
            "project_type, location, key_systems (array), disciplines_involved (array), summary."
        )

        # Wrap snippets as a single synthetic chunk hit to satisfy QA pipeline
        synthetic_chunk = {
            "id": "context_snippets_chunk",
            "text": bundle,
            "metadata": {
                "project_id": project_id,
                "doc_id": "context_doc",
                "doc_name": "Project Context Snippets",
                "file_type": "txt",
                "page_start": 1,
                "page_end": 1,
                "content_type": "SummaryInput",
            },
            # Rough token estimate is sufficient for budgeting purposes
            "token_count": max(50, len(bundle.split()))
        }

        pseudo_hit = {
            "id": "context_snippets",
            "score": 1.0,
            "chunk": synthetic_chunk,
        }

        # Build a minimal project context required by QA service
        minimal_context = {
            "project_name": project_id,
            "project_type": "Auto-Generated Context",
            "description": "",
            "key_systems": [],
            "disciplines_involved": [],
            "summary": "Context generation task"
        }

        # Inject instruction into the query instead of passing a kwarg
        instruction_preface = (
            "Follow these rules strictly when summarizing: " + system_instruction
        )
        combined_query = f"{summarization_query}\n\n{instruction_preface}"

        qa_result = qa.generate_answer(
            combined_query,
            [pseudo_hit],
            project_context=minimal_context,
        )

        answer_text = (qa_result.get("answer") or "").strip()
        # Try to parse JSON from the answer
        import json as _json
        parsed = None
        try:
            start = answer_text.find("{")
            end = answer_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = _json.loads(answer_text[start:end+1])
        except Exception:
            parsed = None

        context_out = {
            "project_name": project_id.replace("_", " ").title(),
            "project_type": (parsed or {}).get("project_type", "General Construction Project"),
            "location": (parsed or {}).get("location", ""),
            "key_systems": (parsed or {}).get("key_systems", []),
            "disciplines_involved": (parsed or {}).get("disciplines_involved", []),
            "summary": (parsed or {}).get("summary", answer_text[:1000]),
            "description": "",
        }

        return {"success": True, "context": context_out}
    except Exception as e:
        return {"success": False, "error": str(e)}
