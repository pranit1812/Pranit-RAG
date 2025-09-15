"""
Simplified project manager for Streamlit UI.
"""
import os
import json
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


class SimpleProjectInfo:
    """Simple project information for UI."""
    
    def __init__(self, project_id: str, name: str, created_at: datetime, 
                 doc_count: int = 0, chunk_count: int = 0, 
                 last_modified: Optional[datetime] = None):
        self.project_id = project_id
        self.name = name
        self.created_at = created_at
        self.doc_count = doc_count
        self.chunk_count = chunk_count
        self.last_modified = last_modified or created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "last_modified": self.last_modified.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleProjectInfo':
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            doc_count=data.get("doc_count", 0),
            chunk_count=data.get("chunk_count", 0),
            last_modified=datetime.fromisoformat(data.get("last_modified", data["created_at"]))
        )


class SimpleProjectManager:
    """Simplified project manager for UI."""
    
    def __init__(self, storage_dir: str = "./storage"):
        """Initialize project manager."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for project registry
        self.metadata_file = self.storage_dir / "projects.json"
        
        # Initialize metadata file if it doesn't exist
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    def create_project(self, project_name: str, project_id: Optional[str] = None) -> str:
        """Create a new project."""
        # Validate project name
        if not project_name or not project_name.strip():
            raise ValueError("Project name cannot be empty")
        
        # Generate project ID if not provided
        if not project_id:
            project_id = self._generate_project_id(project_name)
        
        # Check if project already exists
        if self.project_exists(project_id):
            raise ValueError(f"Project '{project_id}' already exists")
        
        # Create project directory structure
        project_dir = self.storage_dir / project_id
        self._create_project_structure(project_dir)
        
        # Create project info
        project_info = SimpleProjectInfo(
            project_id=project_id,
            name=project_name.strip(),
            created_at=datetime.now()
        )
        
        # Save to metadata
        metadata = self._load_metadata()
        metadata[project_id] = project_info.to_dict()
        self._save_metadata(metadata)
        
        return project_id
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data."""
        if not self.project_exists(project_id):
            return False
        
        project_dir = self.storage_dir / project_id
        
        try:
            # Remove project directory and all contents
            if project_dir.exists():
                shutil.rmtree(project_dir)
            
            # Remove from metadata
            metadata = self._load_metadata()
            if project_id in metadata:
                del metadata[project_id]
                self._save_metadata(metadata)
            
            return True
        except Exception:
            return False
    
    def list_projects(self) -> List[SimpleProjectInfo]:
        """List all projects with statistics."""
        metadata = self._load_metadata()
        projects = []
        
        for project_id, project_data in metadata.items():
            try:
                # Load project info and update statistics
                project_info = SimpleProjectInfo.from_dict(project_data)
                self._update_project_statistics(project_info)
                projects.append(project_info)
            except Exception:
                # Skip invalid project entries
                continue
        
        # Sort by last modified date (newest first)
        projects.sort(key=lambda p: p.last_modified, reverse=True)
        
        return projects
    
    def get_project_info(self, project_id: str) -> Optional[SimpleProjectInfo]:
        """Get information about a specific project."""
        metadata = self._load_metadata()
        
        if project_id not in metadata:
            return None
        
        try:
            project_info = SimpleProjectInfo.from_dict(metadata[project_id])
            self._update_project_statistics(project_info)
            return project_info
        except Exception:
            return None
    
    def project_exists(self, project_id: str) -> bool:
        """Check if a project exists."""
        project_dir = self.storage_dir / project_id
        metadata = self._load_metadata()
        
        return project_dir.exists() and project_id in metadata
    
    def get_project_path(self, project_id: str) -> Optional[Path]:
        """Get the filesystem path for a project."""
        if not self.project_exists(project_id):
            return None
        
        return self.storage_dir / project_id
    
    def update_project_statistics(self, project_id: str, doc_count: int, chunk_count: int) -> bool:
        """Update project statistics."""
        metadata = self._load_metadata()
        
        if project_id not in metadata:
            return False
        
        try:
            metadata[project_id]["doc_count"] = doc_count
            metadata[project_id]["chunk_count"] = chunk_count
            metadata[project_id]["last_modified"] = datetime.now().isoformat()
            
            self._save_metadata(metadata)
            return True
        except Exception:
            return False
    
    def _generate_project_id(self, project_name: str) -> str:
        """Generate a project ID from project name."""
        # Convert to lowercase and replace spaces/special chars with underscores
        project_id = project_name.lower().strip()
        project_id = ''.join(c if c.isalnum() else '_' for c in project_id)
        
        # Remove multiple consecutive underscores
        import re
        project_id = re.sub(r'_+', '_', project_id)
        project_id = project_id.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it's not empty
        if not project_id:
            project_id = "project"
        
        # Make unique if already exists
        base_id = project_id
        counter = 1
        while self.project_exists(project_id):
            project_id = f"{base_id}_{counter}"
            counter += 1
        
        return project_id
    
    def _create_project_structure(self, project_dir: Path) -> None:
        """Create the directory structure for a new project."""
        # Create main project directory
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "raw",        # Original uploaded files
            "pages",      # Cached page images (PNG)
            "chroma"      # Chroma vector database
        ]
        
        for subdir in subdirs:
            (project_dir / subdir).mkdir(exist_ok=True)
        
        # Create empty chunks.jsonl file
        chunks_file = project_dir / "chunks.jsonl"
        chunks_file.touch()
        
        # Create placeholder project_context.md
        context_file = project_dir / "project_context.md"
        if not context_file.exists():
            placeholder_content = f"""# {project_dir.name}

## Project Information

**Project Type:** General Construction Project

**Description:** A construction project with uploaded documents for analysis.

## Project Summary

This project is ready for document upload and processing.
"""
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(placeholder_content)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load project metadata from JSON file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save project metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to save project metadata: {e}")
    
    def _update_project_statistics(self, project_info: SimpleProjectInfo) -> None:
        """Update project statistics by scanning project directory."""
        project_dir = self.storage_dir / project_info.project_id
        
        # Count documents in raw directory
        raw_dir = project_dir / "raw"
        doc_count = 0
        if raw_dir.exists():
            doc_count = len([f for f in raw_dir.iterdir() if f.is_file()])
        
        # Count chunks in chunks.jsonl
        chunk_count = 0
        chunks_file = project_dir / "chunks.jsonl"
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunk_count = len([line for line in content.split('\n') if line.strip()])
            except Exception:
                chunk_count = 0
        
        # Update project info
        project_info.doc_count = doc_count
        project_info.chunk_count = chunk_count
        
        # Update metadata with new statistics
        metadata = self._load_metadata()
        if project_info.project_id in metadata:
            metadata[project_info.project_id]["doc_count"] = doc_count
            metadata[project_info.project_id]["chunk_count"] = chunk_count
            metadata[project_info.project_id]["last_modified"] = datetime.now().isoformat()
            self._save_metadata(metadata)
        
        # Update last modified time from directory
        try:
            if project_dir.exists():
                mtime = datetime.fromtimestamp(project_dir.stat().st_mtime)
                if mtime > project_info.last_modified:
                    project_info.last_modified = mtime
        except Exception:
            pass