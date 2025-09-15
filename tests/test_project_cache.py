"""
Tests for project cache system.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from src.services.project_cache import ProjectCache, ProjectState, ProjectCacheManager
from src.models.types import ProjectContext


class TestProjectState:
    """Test cases for ProjectState."""
    
    def test_project_state_creation(self):
        """Test ProjectState creation."""
        now = datetime.now()
        state = ProjectState(
            project_id="test_project",
            project_name="Test Project",
            last_accessed=now,
            chunk_count=100,
            doc_count=5
        )
        
        assert state.project_id == "test_project"
        assert state.project_name == "Test Project"
        assert state.last_accessed == now
        assert state.chunk_count == 100
        assert state.doc_count == 5
        assert not state.vector_store_loaded
        assert not state.bm25_index_loaded
    
    def test_project_state_serialization(self):
        """Test ProjectState serialization."""
        now = datetime.now()
        context = ProjectContext(
            project_name="Test Project",
            description="A test project",
            project_type="Commercial Office Building",
            location="Test City",
            key_systems=["HVAC", "Electrical"],
            disciplines_involved=["Mechanical", "Electrical"],
            summary="Test project summary"
        )
        
        state = ProjectState(
            project_id="test_project",
            project_name="Test Project",
            last_accessed=now,
            context=context,
            vector_store_loaded=True,
            chunk_count=100
        )
        
        # Test to_dict
        data = state.to_dict()
        assert data["project_id"] == "test_project"
        assert data["last_accessed"] == now.isoformat()
        assert data["context"] == context
        assert data["vector_store_loaded"] is True
        
        # Test from_dict
        restored = ProjectState.from_dict(data)
        assert restored.project_id == state.project_id
        assert restored.last_accessed == state.last_accessed
        assert restored.context == state.context
        assert restored.vector_store_loaded == state.vector_store_loaded


class TestProjectCache:
    """Test cases for ProjectCache."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ProjectCache(max_size=3, cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        state = ProjectState(
            project_id="test_project",
            project_name="Test Project",
            last_accessed=datetime.now()
        )
        
        # Put in cache
        self.cache.put(state)
        
        # Get from cache
        retrieved = self.cache.get("test_project")
        assert retrieved is not None
        assert retrieved.project_id == "test_project"
        assert retrieved.project_name == "Test Project"
    
    def test_cache_get_nonexistent(self):
        """Test getting non-existent project from cache."""
        retrieved = self.cache.get("nonexistent")
        assert retrieved is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache exceeds max size."""
        # Fill cache to capacity
        for i in range(3):
            state = ProjectState(
                project_id=f"project_{i}",
                project_name=f"Project {i}",
                last_accessed=datetime.now() - timedelta(minutes=i)
            )
            self.cache.put(state)
        
        # All should be in cache
        assert self.cache.get("project_0") is not None
        assert self.cache.get("project_1") is not None
        assert self.cache.get("project_2") is not None
        
        # Add one more (should evict oldest)
        state = ProjectState(
            project_id="project_3",
            project_name="Project 3",
            last_accessed=datetime.now()
        )
        self.cache.put(state)
        
        # Oldest should be evicted
        assert self.cache.get("project_0") is None  # Oldest access time
        assert self.cache.get("project_1") is not None
        assert self.cache.get("project_2") is not None
        assert self.cache.get("project_3") is not None
    
    def test_cache_access_order_update(self):
        """Test that accessing items updates their position in LRU order."""
        # Add items
        for i in range(3):
            state = ProjectState(
                project_id=f"project_{i}",
                project_name=f"Project {i}",
                last_accessed=datetime.now() - timedelta(minutes=3-i)
            )
            self.cache.put(state)
        
        # Access oldest item (should move to end)
        self.cache.get("project_0")
        
        # Add new item (should evict project_1, not project_0)
        state = ProjectState(
            project_id="project_3",
            project_name="Project 3",
            last_accessed=datetime.now()
        )
        self.cache.put(state)
        
        assert self.cache.get("project_0") is not None  # Should still be there
        assert self.cache.get("project_1") is None      # Should be evicted
        assert self.cache.get("project_2") is not None
        assert self.cache.get("project_3") is not None
    
    def test_cache_remove(self):
        """Test removing items from cache."""
        state = ProjectState(
            project_id="test_project",
            project_name="Test Project",
            last_accessed=datetime.now()
        )
        
        self.cache.put(state)
        assert self.cache.get("test_project") is not None
        
        # Remove from cache
        removed = self.cache.remove("test_project")
        assert removed is True
        assert self.cache.get("test_project") is None
        
        # Try to remove again
        removed = self.cache.remove("test_project")
        assert removed is False
    
    def test_cache_clear(self):
        """Test clearing cache."""
        # Add some items
        for i in range(2):
            state = ProjectState(
                project_id=f"project_{i}",
                project_name=f"Project {i}",
                last_accessed=datetime.now()
            )
            self.cache.put(state)
        
        assert self.cache.get_cache_size() == 2
        
        # Clear cache
        self.cache.clear()
        assert self.cache.get_cache_size() == 0
        assert self.cache.get("project_0") is None
        assert self.cache.get("project_1") is None
    
    def test_cache_list_cached_projects(self):
        """Test listing cached projects."""
        # Add projects with different access times
        times = [
            datetime.now() - timedelta(minutes=2),
            datetime.now() - timedelta(minutes=1),
            datetime.now()
        ]
        
        for i, time in enumerate(times):
            state = ProjectState(
                project_id=f"project_{i}",
                project_name=f"Project {i}",
                last_accessed=time
            )
            self.cache.put(state)
        
        # List projects (should be in reverse access order)
        projects = self.cache.list_cached_projects()
        assert len(projects) == 3
        assert projects[0].project_id == "project_2"  # Most recent
        assert projects[1].project_id == "project_1"
        assert projects[2].project_id == "project_0"  # Oldest
    
    def test_cache_update_project_state(self):
        """Test updating project state in cache."""
        state = ProjectState(
            project_id="test_project",
            project_name="Test Project",
            last_accessed=datetime.now(),
            chunk_count=50
        )
        
        self.cache.put(state)
        
        # Update state
        updated = self.cache.update_project_state(
            "test_project",
            chunk_count=100,
            vector_store_loaded=True
        )
        assert updated is True
        
        # Check updates
        retrieved = self.cache.get("test_project")
        assert retrieved.chunk_count == 100
        assert retrieved.vector_store_loaded is True
        
        # Try to update non-existent project
        updated = self.cache.update_project_state("nonexistent", chunk_count=200)
        assert updated is False
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        # Initially empty
        stats = self.cache.get_cache_stats()
        assert stats["cached_projects"] == 0
        assert stats["max_size"] == 3
        assert stats["cache_utilization"] == 0.0
        
        # Add some projects
        for i in range(2):
            state = ProjectState(
                project_id=f"project_{i}",
                project_name=f"Project {i}",
                last_accessed=datetime.now()
            )
            self.cache.put(state)
        
        stats = self.cache.get_cache_stats()
        assert stats["cached_projects"] == 2
        assert stats["cache_utilization"] == 2/3
        assert stats["oldest_access"] is not None
        assert stats["newest_access"] is not None
    
    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        # Add project to cache
        state = ProjectState(
            project_id="test_project",
            project_name="Test Project",
            last_accessed=datetime.now(),
            chunk_count=100
        )
        self.cache.put(state)
        
        # Create new cache instance (should load from disk)
        new_cache = ProjectCache(max_size=3, cache_dir=self.temp_dir)
        
        # Check that project was loaded
        retrieved = new_cache.get("test_project")
        assert retrieved is not None
        assert retrieved.project_id == "test_project"
        assert retrieved.chunk_count == 100


class TestProjectCacheManager:
    """Test cases for ProjectCacheManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        cache = ProjectCache(max_size=3, cache_dir=self.temp_dir)
        self.manager = ProjectCacheManager(cache)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_project(self):
        """Test loading project into cache."""
        state = self.manager.load_project("test_project", "Test Project")
        
        assert state.project_id == "test_project"
        assert state.project_name == "Test Project"
        
        # Should be cached now
        assert self.manager.cache.is_cached("test_project")
    
    def test_switch_project(self):
        """Test switching between projects."""
        # Load first project
        state1 = self.manager.switch_project("project_1", "Project 1")
        assert state1.project_id == "project_1"
        
        current = self.manager.get_current_project()
        assert current.project_id == "project_1"
        
        # Switch to second project
        state2 = self.manager.switch_project("project_2", "Project 2")
        assert state2.project_id == "project_2"
        
        current = self.manager.get_current_project()
        assert current.project_id == "project_2"
    
    def test_update_current_project(self):
        """Test updating current project state."""
        # Load project
        self.manager.load_project("test_project", "Test Project")
        
        # Update current project
        updated = self.manager.update_current_project(
            chunk_count=100,
            vector_store_loaded=True
        )
        assert updated is True
        
        # Check updates
        current = self.manager.get_current_project()
        assert current.chunk_count == 100
        assert current.vector_store_loaded is True
    
    def test_invalidate_project(self):
        """Test invalidating project from cache."""
        # Load project
        self.manager.load_project("test_project", "Test Project")
        assert self.manager.cache.is_cached("test_project")
        
        # Invalidate
        self.manager.invalidate_project("test_project")
        assert not self.manager.cache.is_cached("test_project")
        
        # Current project should be cleared if it was the invalidated one
        current = self.manager.get_current_project()
        assert current is None