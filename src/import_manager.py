"""
Centralized import manager for the Construction RAG System.
Handles all imports properly to avoid relative import issues when running from Streamlit.
"""
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, Optional


class ImportManager:
    """Manages imports for the RAG system to avoid relative import issues."""
    
    def __init__(self):
        self._ensure_src_in_path()
        self._imported_modules: Dict[str, Any] = {}
    
    def _ensure_src_in_path(self):
        """Ensure src directory is in Python path."""
        src_path = str(Path(__file__).parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    
    def get_config(self):
        """Get config module and classes."""
        if 'config' not in self._imported_modules:
            config_module = importlib.import_module('config')
            self._imported_modules['config'] = {
                'module': config_module,
                'get_config': config_module.get_config,
                'Config': config_module.Config
            }
        return self._imported_modules['config']
    
    def get_models(self):
        """Get models module and types."""
        if 'models' not in self._imported_modules:
            types_module = importlib.import_module('models.types')
            validation_module = importlib.import_module('models.validation')
            self._imported_modules['models'] = {
                'types': types_module,
                'validation': validation_module,
                # Export all common types
                'PageParse': types_module.PageParse,
                'Block': types_module.Block,
                'Chunk': types_module.Chunk,
                'ChunkMetadata': types_module.ChunkMetadata,
                'ChunkPolicy': types_module.ChunkPolicy,
                'Hit': types_module.Hit,
                'ProjectContext': types_module.ProjectContext,
                'VisionConfig': types_module.VisionConfig,
                'ContextPacket': types_module.ContextPacket,
                'generate_text_hash': types_module.generate_text_hash,
                'validate_project_context': validation_module.validate_project_context
            }
        return self._imported_modules['models']
    
    def get_extractors(self):
        """Get extractor modules and classes."""
        if 'extractors' not in self._imported_modules:
            base_module = importlib.import_module('extractors.base')
            
            extractors = {
                'base': base_module,
                'BaseExtractor': base_module.BaseExtractor,
                'ExtractorError': base_module.ExtractorError
            }
            
            # Import extractors with error handling
            try:
                docling_module = importlib.import_module('extractors.docling_extractor')
                extractors['DoclingExtractor'] = docling_module.DoclingExtractor
            except ImportError:
                pass
            
            try:
                unstructured_module = importlib.import_module('extractors.unstructured_extractor')
                extractors['UnstructuredExtractor'] = unstructured_module.UnstructuredExtractor
            except ImportError:
                pass
            
            try:
                native_pdf_module = importlib.import_module('extractors.native_pdf')
                extractors['NativePDFExtractor'] = native_pdf_module.NativePDFExtractor
            except ImportError:
                pass
            
            try:
                office_module = importlib.import_module('extractors.office_extractors')
                extractors['DOCXExtractor'] = office_module.DOCXExtractor
                extractors['XLSXExtractor'] = office_module.XLSXExtractor
            except ImportError:
                pass
            
            try:
                router_module = importlib.import_module('extractors.extraction_router')
                extractors['ExtractionRouter'] = router_module.ExtractionRouter
            except ImportError:
                pass
            
            self._imported_modules['extractors'] = extractors
        
        return self._imported_modules['extractors']
    
    def get_chunking(self):
        """Get chunking modules and classes."""
        if 'chunking' not in self._imported_modules:
            chunker_module = importlib.import_module('chunking.chunker')
            token_counter_module = importlib.import_module('chunking.token_counter')
            table_processor_module = importlib.import_module('chunking.table_processor')
            list_processor_module = importlib.import_module('chunking.list_processor')
            drawing_processor_module = importlib.import_module('chunking.drawing_processor')
            context_window_module = importlib.import_module('chunking.context_window')
            
            self._imported_modules['chunking'] = {
                'DocumentChunker': chunker_module.DocumentChunker,
                'chunk_page': chunker_module.chunk_page,
                'TokenCounter': token_counter_module.TokenCounter,
                'TableProcessor': table_processor_module.TableProcessor,
                'ListProcessor': list_processor_module.ListProcessor,
                'DrawingProcessor': drawing_processor_module.DrawingProcessor,
                'ContextWindow': context_window_module.ContextWindow
            }
        
        return self._imported_modules['chunking']
    
    def get_services(self):
        """Get services modules and classes."""
        if 'services' not in self._imported_modules:
            services = {}
            
            # Import each service module
            service_modules = [
                'embedding', 'vector_store', 'bm25_search', 'classification',
                'project_context', 'retrieval', 'qa_assembly', 'vision',
                'filtering', 'reranking', 'indexing', 'project_manager',
                'file_processor'
            ]
            
            for module_name in service_modules:
                try:
                    module = importlib.import_module(f'services.{module_name}')
                    services[module_name] = module
                except ImportError as e:
                    logger.warning(f"Could not import services.{module_name}: {e}")
            
            self._imported_modules['services'] = services
        
        return self._imported_modules['services']
    
    def get_utils(self):
        """Get utility modules."""
        if 'utils' not in self._imported_modules:
            utils = {}
            
            util_modules = [
                'bbox', 'debug_logging', 'error_handling', 'hashing',
                'io_utils', 'log_management', 'logging_config', 'monitoring',
                'pdf_utils'
            ]
            
            for module_name in util_modules:
                try:
                    module = importlib.import_module(f'utils.{module_name}')
                    utils[module_name] = module
                except ImportError as e:
                    logger.warning(f"Could not import utils.{module_name}: {e}")
            
            self._imported_modules['utils'] = utils
        
        return self._imported_modules['utils']


# Global import manager instance
_import_manager = ImportManager()


def get_import_manager() -> ImportManager:
    """Get the global import manager."""
    return _import_manager


# Convenience functions
def get_config():
    """Get config module."""
    return _import_manager.get_config()

def get_models():
    """Get models module."""
    return _import_manager.get_models()

def get_extractors():
    """Get extractors module."""
    return _import_manager.get_extractors()

def get_chunking():
    """Get chunking module."""
    return _import_manager.get_chunking()

def get_services():
    """Get services module."""
    return _import_manager.get_services()

def get_utils():
    """Get utils module."""
    return _import_manager.get_utils()


