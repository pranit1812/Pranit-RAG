"""
Tests for project management system.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.services.project_manager import ProjectManager, ProjectInfo


class TestProjectManager:
    """Test cases for ProjectManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ProjectManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_project(self):
        """Test project creation."""
        project_id = self.manager.create_project("Test Project")
        
        assert project_id == "test_project"
        assert self.manager.project_exists(project_id)
        
        # Check directory structure
        project_path = self.manager.get_project_path(project_id)
        assert project_path is not None
        assert project_path.exists()
        assert (project_path / "raw").exists()
        assert (project_path / "pages").exists()
        assert (project_path / "chroma").exists()
        assert (project_path / "chunks.jsonl").exists()
        assert (project_path / "project_context.md").exists()
    
    def test_create_project_with_custom_id(self):
        """Test project creation with custom ID."""
        project_id = self.manager.create_project("Test Project", "custom_id")
        
        assert project_id == "custom_id"
        assert self.manager.project_exists(project_id)
    
    def test_create_duplicate_project(self):
        """Test creating duplicate project raises error."""
        project_id = self.manager.create_project("Test Project")
        
        with pytest.raises(ValueError, match="already exists"):
            self.manager.create_project("Another Project", project_id)
    
    def test_create_project_empty_name(self):
        """Test creating project with empty name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            self.manager.create_project("")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            self.manager.create_project("   ")
    
    def test_delete_project(self):
        """Test project deletion."""
        project_id = self.manager.create_project("Test Project")
        assert self.manager.project_exists(project_id)
        
        success = self.manager.delete_project(project_id)
        assert success
        assert not self.manager.project_exists(project_id)
        
        project_path = Path(self.temp_dir) / project_id
        assert not project_path.exists()
    
    def test_delete_nonexistent_project(self):
        """Test deleting non-existent project."""
        success = self.manager.delete_project("nonexistent")
        assert not success
    
    def test_list_projects(self):
        """Test listing projects."""
        # Initially empty
        projects = self.manager.list_projects()
        assert len(projects) == 0
        
        # Create some projects
        id1 = self.manager.create_project("Project 1")
        id2 = self.manager.create_project("Project 2")
        
        projects = self.manager.list_projects()
        assert len(projects) == 2
        
        project_ids = [p.project_id for p in projects]
        assert id1 in project_ids
        assert id2 in project_ids
    
    def test_get_project_info(self):
        """Test getting project information."""
        project_id = self.manager.create_project("Test Project")
        
        info = self.manager.get_project_info(project_id)
        assert info is not None
        assert info.project_id == project_id
        assert info.name == "Test Project"
        assert info.doc_count == 0
        assert info.chunk_count == 0
        assert isinstance(info.created_at, datetime)
    
    def test_get_nonexistent_project_info(self):
        """Test getting info for non-existent project."""
        info = self.manager.get_project_info("nonexistent")
        assert info is None
    
    def test_update_project_statistics(self):
        """Test updating project statistics."""
        project_id = self.manager.create_project("Test Project")
        
        success = self.manager.update_project_statistics(project_id, 5, 100)
        assert success
        
        info = self.manager.get_project_info(project_id)
        assert info.doc_count == 5
        assert info.chunk_count == 100
    
    def test_update_nonexistent_project_statistics(self):
        """Test updating statistics for non-existent project."""
        success = self.manager.update_project_statistics("nonexistent", 5, 100)
        assert not success
    
    def test_generate_project_id(self):
        """Test project ID generation."""
        # Normal case
        project_id = self.manager._generate_project_id("My Test Project")
        assert project_id == "my_test_project"
        
        # Special characters
        project_id = self.manager._generate_project_id("Project #1 (2024)!")
        assert project_id == "project_1_2024"
        
        # Empty/whitespace
        project_id = self.manager._generate_project_id("   ")
        assert project_id == "project"
        
        # Duplicate handling
        self.manager.create_project("Test", "test")
        project_id = self.manager._generate_project_id("Test")
        assert project_id == "test_1"


class TestProjectInfo:
    """Test cases for ProjectInfo."""
    
    def test_project_info_creation(self):
        """Test ProjectInfo creation."""
        created_at = datetime.now()
        info = ProjectInfo("test_id", "Test Project", created_at, 5, 100)
        
        assert info.project_id == "test_id"
        assert info.name == "Test Project"
        assert info.created_at == created_at
        assert info.doc_count == 5
        assert info.chunk_count == 100
        assert info.last_modified == created_at
    
    def test_project_info_serialization(self):
        """Test ProjectInfo serialization."""
        created_at = datetime.now()
        info = ProjectInfo("test_id", "Test Project", created_at, 5, 100)
        
        # Test to_dict
        data = info.to_dict()
        assert data["project_id"] == "test_id"
        assert data["name"] == "Test Project"
        assert data["created_at"] == created_at.isoformat()
        assert data["doc_count"] == 5
        assert data["chunk_count"] == 100
        
        # Test from_dict
        restored = ProjectInfo.from_dict(data)
        assert restored.project_id == info.project_id
        assert restored.name == info.name
        assert restored.created_at == info.created_at
        assert restored.doc_count == info.doc_count
        assert restored.chunk_count == info.chunk_count