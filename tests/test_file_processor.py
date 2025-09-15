"""
Tests for file processing system.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.services.file_processor import FileValidator, FileProcessor, ProcessingStatus
from src.services.project_manager import ProjectManager


class TestFileValidator:
    """Test cases for FileValidator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = FileValidator(max_file_size_mb=10)  # 10MB limit for testing
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_supported_file(self):
        """Test validation of supported file types."""
        # Create test files
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"PDF content")
        
        docx_file = Path(self.temp_dir) / "test.docx"
        docx_file.write_bytes(b"DOCX content")
        
        # Test validation
        is_valid, error = self.validator.validate_file(pdf_file)
        assert is_valid
        assert error is None
        
        is_valid, error = self.validator.validate_file(docx_file)
        assert is_valid
        assert error is None
    
    def test_validate_unsupported_file(self):
        """Test validation of unsupported file types."""
        txt_file = Path(self.temp_dir) / "test.txt"
        txt_file.write_text("Text content")
        
        is_valid, error = self.validator.validate_file(txt_file)
        assert not is_valid
        assert "Unsupported file type" in error
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent = Path(self.temp_dir) / "nonexistent.pdf"
        
        is_valid, error = self.validator.validate_file(nonexistent)
        assert not is_valid
        assert "does not exist" in error
    
    def test_validate_large_file(self):
        """Test validation of file exceeding size limit."""
        large_file = Path(self.temp_dir) / "large.pdf"
        # Create file larger than 10MB limit
        large_file.write_bytes(b"x" * (11 * 1024 * 1024))
        
        is_valid, error = self.validator.validate_file(large_file)
        assert not is_valid
        assert "too large" in error
    
    def test_validate_directory(self):
        """Test validation of directory instead of file."""
        directory = Path(self.temp_dir) / "subdir"
        directory.mkdir()
        
        is_valid, error = self.validator.validate_file(directory)
        assert not is_valid
        assert "not a file" in error
    
    def test_get_file_type(self):
        """Test file type detection."""
        assert self.validator.get_file_type(Path("test.pdf")) == "pdf"
        assert self.validator.get_file_type(Path("test.docx")) == "docx"
        assert self.validator.get_file_type(Path("test.xlsx")) == "xlsx"
        assert self.validator.get_file_type(Path("test.png")) == "image"
        assert self.validator.get_file_type(Path("test.jpg")) == "image"
        assert self.validator.get_file_type(Path("test.txt")) == "unknown"


class TestFileProcessor:
    """Test cases for FileProcessor."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_manager = ProjectManager(self.temp_dir)
        self.project_id = self.project_manager.create_project("Test Project")
        # Create processor with same storage directory
        self.processor = FileProcessor(self.project_id)
        self.processor.project_manager = self.project_manager
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_upload_valid_files(self):
        """Test uploading valid files."""
        # Create test files
        test_files = []
        for i, ext in enumerate([".pdf", ".docx", ".xlsx"]):
            file_path = Path(self.temp_dir) / f"test{i}{ext}"
            file_path.write_bytes(b"Test content")
            test_files.append(str(file_path))
        
        uploads = self.processor.upload_files(test_files)
        
        assert len(uploads) == 3
        for upload in uploads:
            assert upload.status == ProcessingStatus.PENDING
            assert upload.error_message is None
            
            # Check file was copied to project
            project_path = self.project_manager.get_project_path(self.project_id)
            copied_file = project_path / "raw" / upload.filename
            assert copied_file.exists()
    
    def test_upload_invalid_files(self):
        """Test uploading invalid files."""
        # Create invalid file
        invalid_file = Path(self.temp_dir) / "test.txt"
        invalid_file.write_text("Invalid content")
        
        uploads = self.processor.upload_files([str(invalid_file)])
        
        assert len(uploads) == 1
        assert uploads[0].status == ProcessingStatus.FAILED
        assert "Unsupported file type" in uploads[0].error_message
    
    def test_upload_duplicate_filenames(self):
        """Test uploading files with duplicate names."""
        # Create files with same name in different directories
        dir1 = Path(self.temp_dir) / "dir1"
        dir2 = Path(self.temp_dir) / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        file1 = dir1 / "test.pdf"
        file2 = dir2 / "test.pdf"
        file1.write_bytes(b"Content 1")
        file2.write_bytes(b"Content 2")
        
        uploads = self.processor.upload_files([str(file1), str(file2)])
        
        assert len(uploads) == 2
        assert uploads[0].status == ProcessingStatus.PENDING
        assert uploads[1].status == ProcessingStatus.PENDING
        
        # Check files have different names in project
        filenames = [upload.filename for upload in uploads]
        assert len(set(filenames)) == 2  # Should be unique
        assert "test.pdf" in filenames
        assert "test_1.pdf" in filenames
    
    def test_upload_nonexistent_project(self):
        """Test uploading to non-existent project."""
        processor = FileProcessor("nonexistent_project")
        
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"Test content")
        
        with pytest.raises(ValueError, match="does not exist"):
            processor.upload_files([str(test_file)])
    
    def test_start_processing(self):
        """Test starting file processing."""
        # Mock the pipeline directly on the processor instance
        mock_pipeline = Mock()
        mock_pipeline.process_file.return_value = True
        self.processor.pipeline = mock_pipeline
        
        # Create test file and upload
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"Test content")
        uploads = self.processor.upload_files([str(test_file)])
        
        # Start processing
        self.processor.start_processing(uploads)
        
        # Wait for completion
        import time
        timeout = 5  # 5 second timeout
        start_time = time.time()
        while self.processor.is_processing() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        # Check that processing completed
        assert not self.processor.is_processing()
        
        # Check that pipeline was called
        mock_pipeline.process_file.assert_called_once()
    
    def test_cancel_processing(self):
        """Test cancelling processing."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"Test content")
        uploads = self.processor.upload_files([str(test_file)])
        
        # Start processing
        self.processor.start_processing(uploads)
        
        # Cancel immediately
        self.processor.cancel_processing()
        
        # Check cancellation
        progress = self.processor.get_progress()
        assert progress is not None
        assert progress.is_cancelled
    
    def test_progress_tracking(self):
        """Test progress tracking."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"Test content")
        uploads = self.processor.upload_files([str(test_file)])
        
        # Start processing
        self.processor.start_processing(uploads)
        
        # Check initial progress
        progress = self.processor.get_progress()
        assert progress is not None
        assert progress.total_files == 1
        assert progress.completed_files == 0
        assert progress.overall_progress >= 0.0
        
        # Cancel to avoid long test
        self.processor.cancel_processing()
    
    def test_progress_callbacks(self):
        """Test progress callbacks."""
        callback_calls = []
        
        def progress_callback(progress):
            callback_calls.append(progress)
        
        self.processor.add_progress_callback(progress_callback)
        
        # Create test file
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"Test content")
        uploads = self.processor.upload_files([str(test_file)])
        
        # Start processing
        self.processor.start_processing(uploads)
        
        # Wait a bit
        import time
        time.sleep(0.1)
        
        # Cancel processing
        self.processor.cancel_processing()
        
        # Check callbacks were called
        assert len(callback_calls) > 0
        
        # Remove callback
        self.processor.remove_progress_callback(progress_callback)