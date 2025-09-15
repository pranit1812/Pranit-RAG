"""
Metadata filtering system for the Construction RAG System.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

from models.types import ChunkMetadata


logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enumeration of content types for filtering."""
    SPEC_SECTION = "SpecSection"
    DRAWING = "Drawing"
    ITB = "ITB"
    TABLE = "Table"
    LIST = "List"


class Discipline(Enum):
    """Enumeration of disciplines for filtering."""
    ARCHITECTURAL = "A"
    STRUCTURAL = "S"
    MECHANICAL = "M"
    ELECTRICAL = "E"
    PLUMBING = "P"
    FIRE_PROTECTION = "FP"
    ELEVATOR = "EL"


# MasterFormat divisions mapping
MASTERFORMAT_DIVISIONS = {
    "00": "Procurement and Contracting Requirements",
    "01": "General Requirements", 
    "02": "Existing Conditions",
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics and Composites",
    "07": "Thermal and Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "11": "Equipment",
    "12": "Furnishings",
    "13": "Special Construction",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "Heating, Ventilating and Air Conditioning (HVAC)",
    "25": "Integrated Automation",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety and Security",
    "31": "Earthwork",
    "32": "Exterior Improvements",
    "33": "Utilities",
    "34": "Transportation",
    "35": "Waterway and Marine Construction",
    "40": "Process Integration",
    "41": "Material Processing and Handling Equipment",
    "42": "Process Heating, Cooling and Drying Equipment",
    "43": "Process Gas and Liquid Handling, Purification and Storage Equipment",
    "44": "Pollution and Waste Control Equipment",
    "45": "Industry-Specific Manufacturing Equipment",
    "46": "Water and Wastewater Equipment",
    "48": "Electrical Power Generation"
}


@dataclass
class FilterCriteria:
    """
    Comprehensive filter criteria for search operations.
    """
    # Content type filters
    content_types: Optional[List[str]] = None
    
    # Division and section filters
    division_codes: Optional[List[str]] = None
    division_titles: Optional[List[str]] = None
    section_codes: Optional[List[str]] = None
    section_titles: Optional[List[str]] = None
    
    # Drawing-specific filters
    disciplines: Optional[List[str]] = None
    sheet_numbers: Optional[List[str]] = None
    sheet_titles: Optional[List[str]] = None
    
    # Document filters
    doc_names: Optional[List[str]] = None
    file_types: Optional[List[str]] = None
    
    # Page range filters
    page_start_min: Optional[int] = None
    page_start_max: Optional[int] = None
    page_end_min: Optional[int] = None
    page_end_max: Optional[int] = None
    
    # Confidence filters
    low_conf_only: Optional[bool] = None
    high_conf_only: Optional[bool] = None
    
    # Project filter
    project_id: Optional[str] = None


class MetadataFilter:
    """
    Metadata filtering system with validation and combination logic.
    """
    
    def __init__(self):
        """Initialize metadata filter."""
        self.valid_content_types = {ct.value for ct in ContentType}
        self.valid_disciplines = {d.value for d in Discipline}
        self.valid_division_codes = set(MASTERFORMAT_DIVISIONS.keys())
        self.valid_file_types = {"pdf", "docx", "xlsx", "image"}
        
        logger.info("Initialized metadata filter")
    
    def validate_criteria(self, criteria: FilterCriteria) -> List[str]:
        """
        Validate filter criteria and return list of validation errors.
        
        Args:
            criteria: Filter criteria to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate content types
        if criteria.content_types:
            invalid_types = set(criteria.content_types) - self.valid_content_types
            if invalid_types:
                errors.append(f"Invalid content types: {invalid_types}")
        
        # Validate disciplines
        if criteria.disciplines:
            invalid_disciplines = set(criteria.disciplines) - self.valid_disciplines
            if invalid_disciplines:
                errors.append(f"Invalid disciplines: {invalid_disciplines}")
        
        # Validate division codes
        if criteria.division_codes:
            invalid_codes = set(criteria.division_codes) - self.valid_division_codes
            if invalid_codes:
                errors.append(f"Invalid division codes: {invalid_codes}")
        
        # Validate file types
        if criteria.file_types:
            invalid_types = set(criteria.file_types) - self.valid_file_types
            if invalid_types:
                errors.append(f"Invalid file types: {invalid_types}")
        
        # Validate page ranges
        if criteria.page_start_min is not None and criteria.page_start_min < 1:
            errors.append("page_start_min must be >= 1")
        
        if criteria.page_end_min is not None and criteria.page_end_min < 1:
            errors.append("page_end_min must be >= 1")
        
        if (criteria.page_start_min is not None and criteria.page_start_max is not None 
            and criteria.page_start_min > criteria.page_start_max):
            errors.append("page_start_min cannot be greater than page_start_max")
        
        if (criteria.page_end_min is not None and criteria.page_end_max is not None 
            and criteria.page_end_min > criteria.page_end_max):
            errors.append("page_end_min cannot be greater than page_end_max")
        
        # Validate confidence filters
        if criteria.low_conf_only and criteria.high_conf_only:
            errors.append("Cannot filter for both low_conf_only and high_conf_only")
        
        return errors
    
    def apply_filters(self, metadata: ChunkMetadata, criteria: FilterCriteria) -> bool:
        """
        Apply filter criteria to chunk metadata.
        
        Args:
            metadata: Chunk metadata to test
            criteria: Filter criteria to apply
            
        Returns:
            True if metadata passes all filters, False otherwise
        """
        # Project ID filter (required)
        if criteria.project_id and metadata["project_id"] != criteria.project_id:
            return False
        
        # Content type filter
        if criteria.content_types and metadata["content_type"] not in criteria.content_types:
            return False
        
        # Division code filter
        if criteria.division_codes:
            if not metadata.get("division_code") or metadata["division_code"] not in criteria.division_codes:
                return False
        
        # Division title filter (partial match)
        if criteria.division_titles:
            if not metadata.get("division_title"):
                return False
            title_match = any(
                title.lower() in metadata["division_title"].lower()
                for title in criteria.division_titles
            )
            if not title_match:
                return False
        
        # Section code filter
        if criteria.section_codes:
            if not metadata.get("section_code") or metadata["section_code"] not in criteria.section_codes:
                return False
        
        # Section title filter (partial match)
        if criteria.section_titles:
            if not metadata.get("section_title"):
                return False
            title_match = any(
                title.lower() in metadata["section_title"].lower()
                for title in criteria.section_titles
            )
            if not title_match:
                return False
        
        # Discipline filter
        if criteria.disciplines:
            if not metadata.get("discipline") or metadata["discipline"] not in criteria.disciplines:
                return False
        
        # Sheet number filter
        if criteria.sheet_numbers:
            if not metadata.get("sheet_number") or metadata["sheet_number"] not in criteria.sheet_numbers:
                return False
        
        # Sheet title filter (partial match)
        if criteria.sheet_titles:
            if not metadata.get("sheet_title"):
                return False
            title_match = any(
                title.lower() in metadata["sheet_title"].lower()
                for title in criteria.sheet_titles
            )
            if not title_match:
                return False
        
        # Document name filter (partial match)
        if criteria.doc_names:
            doc_match = any(
                name.lower() in metadata["doc_name"].lower()
                for name in criteria.doc_names
            )
            if not doc_match:
                return False
        
        # File type filter
        if criteria.file_types and metadata["file_type"] not in criteria.file_types:
            return False
        
        # Page range filters
        if criteria.page_start_min is not None and metadata["page_start"] < criteria.page_start_min:
            return False
        
        if criteria.page_start_max is not None and metadata["page_start"] > criteria.page_start_max:
            return False
        
        if criteria.page_end_min is not None and metadata["page_end"] < criteria.page_end_min:
            return False
        
        if criteria.page_end_max is not None and metadata["page_end"] > criteria.page_end_max:
            return False
        
        # Confidence filters
        if criteria.low_conf_only is not None:
            if criteria.low_conf_only and not metadata["low_conf"]:
                return False
            if not criteria.low_conf_only and metadata["low_conf"]:
                return False
        
        if criteria.high_conf_only is not None:
            if criteria.high_conf_only and metadata["low_conf"]:
                return False
            if not criteria.high_conf_only and not metadata["low_conf"]:
                return False
        
        return True
    
    def build_vector_store_filter(self, criteria: FilterCriteria) -> Dict[str, Any]:
        """
        Build filter dictionary for vector store queries.
        
        Args:
            criteria: Filter criteria
            
        Returns:
            Dictionary suitable for vector store where clause
        """
        where = {}
        
        # Project ID (required)
        if criteria.project_id:
            where["project_id"] = criteria.project_id
        
        # Content type filters
        if criteria.content_types:
            if len(criteria.content_types) == 1:
                where["content_type"] = criteria.content_types[0]
            else:
                where["content_type"] = {"$in": criteria.content_types}
        
        # Division code filters
        if criteria.division_codes:
            if len(criteria.division_codes) == 1:
                where["division_code"] = criteria.division_codes[0]
            else:
                where["division_code"] = {"$in": criteria.division_codes}
        
        # Discipline filters
        if criteria.disciplines:
            if len(criteria.disciplines) == 1:
                where["discipline"] = criteria.disciplines[0]
            else:
                where["discipline"] = {"$in": criteria.disciplines}
        
        # Document name filters (exact match for vector store)
        if criteria.doc_names:
            if len(criteria.doc_names) == 1:
                where["doc_name"] = criteria.doc_names[0]
            else:
                where["doc_name"] = {"$in": criteria.doc_names}
        
        # File type filters
        if criteria.file_types:
            if len(criteria.file_types) == 1:
                where["file_type"] = criteria.file_types[0]
            else:
                where["file_type"] = {"$in": criteria.file_types}
        
        # Confidence filter
        if criteria.low_conf_only is not None:
            where["low_conf"] = str(criteria.low_conf_only).lower()
        elif criteria.high_conf_only is not None:
            where["low_conf"] = str(not criteria.high_conf_only).lower()
        
        # Note: Vector stores typically don't support range queries well,
        # so page range and partial text matches would need post-processing
        
        return where
    
    def build_bm25_filter(self, criteria: FilterCriteria) -> Dict[str, Any]:
        """
        Build filter dictionary for BM25 search queries.
        
        Args:
            criteria: Filter criteria
            
        Returns:
            Dictionary suitable for BM25 search filters
        """
        filters = {}
        
        # Content type filters
        if criteria.content_types:
            filters["content_types"] = criteria.content_types
        
        # Division code filters
        if criteria.division_codes:
            filters["division_codes"] = criteria.division_codes
        
        # Discipline filters
        if criteria.disciplines:
            filters["disciplines"] = criteria.disciplines
        
        # Document name filters
        if criteria.doc_names:
            filters["doc_names"] = criteria.doc_names
        
        # File type filters
        if criteria.file_types:
            filters["file_types"] = criteria.file_types
        
        # Confidence filter
        if criteria.low_conf_only is not None:
            filters["low_conf_only"] = criteria.low_conf_only
        elif criteria.high_conf_only is not None:
            filters["low_conf_only"] = not criteria.high_conf_only
        
        return filters
    
    def get_available_values(self, project_id: str, field: str) -> List[str]:
        """
        Get available values for a specific metadata field in a project.
        This would typically query the vector store or index to get actual values.
        
        Args:
            project_id: Project identifier
            field: Metadata field name
            
        Returns:
            List of available values for the field
        """
        # This is a placeholder implementation
        # In a real implementation, this would query the actual data
        
        if field == "content_type":
            return list(self.valid_content_types)
        elif field == "discipline":
            return list(self.valid_disciplines)
        elif field == "division_code":
            return list(self.valid_division_codes)
        elif field == "file_type":
            return list(self.valid_file_types)
        else:
            logger.warning(f"Unknown field for available values: {field}")
            return []
    
    def get_division_title(self, division_code: str) -> Optional[str]:
        """
        Get division title for a given division code.
        
        Args:
            division_code: MasterFormat division code
            
        Returns:
            Division title or None if not found
        """
        return MASTERFORMAT_DIVISIONS.get(division_code)
    
    def get_all_divisions(self) -> Dict[str, str]:
        """
        Get all MasterFormat divisions.
        
        Returns:
            Dictionary mapping division codes to titles
        """
        return MASTERFORMAT_DIVISIONS.copy()
    
    def suggest_filters(self, query: str) -> FilterCriteria:
        """
        Suggest filter criteria based on query content.
        
        Args:
            query: User query string
            
        Returns:
            Suggested filter criteria
        """
        query_lower = query.lower()
        criteria = FilterCriteria()
        
        # Suggest content types based on query keywords
        content_type_keywords = {
            "drawing": [ContentType.DRAWING.value],
            "plan": [ContentType.DRAWING.value],
            "elevation": [ContentType.DRAWING.value],
            "section": [ContentType.DRAWING.value],
            "detail": [ContentType.DRAWING.value],
            "specification": [ContentType.SPEC_SECTION.value],
            "spec": [ContentType.SPEC_SECTION.value],
            "table": [ContentType.TABLE.value],
            "schedule": [ContentType.TABLE.value],
            "list": [ContentType.LIST.value],
            "itb": [ContentType.ITB.value],
            "instruction": [ContentType.ITB.value]
        }
        
        for keyword, types in content_type_keywords.items():
            if keyword in query_lower:
                criteria.content_types = (criteria.content_types or []) + types
        
        # Suggest disciplines based on query keywords
        discipline_keywords = {
            "architectural": [Discipline.ARCHITECTURAL.value],
            "structural": [Discipline.STRUCTURAL.value],
            "mechanical": [Discipline.MECHANICAL.value],
            "hvac": [Discipline.MECHANICAL.value],
            "electrical": [Discipline.ELECTRICAL.value],
            "plumbing": [Discipline.PLUMBING.value],
            "fire": [Discipline.FIRE_PROTECTION.value],
            "elevator": [Discipline.ELEVATOR.value]
        }
        
        for keyword, disciplines in discipline_keywords.items():
            if keyword in query_lower:
                criteria.disciplines = (criteria.disciplines or []) + disciplines
        
        # Suggest division codes based on query keywords
        division_keywords = {
            "concrete": ["03"],
            "masonry": ["04"],
            "steel": ["05"],
            "metal": ["05"],
            "wood": ["06"],
            "roofing": ["07"],
            "waterproofing": ["07"],
            "insulation": ["07"],
            "door": ["08"],
            "window": ["08"],
            "opening": ["08"],
            "finish": ["09"],
            "paint": ["09"],
            "flooring": ["09"],
            "ceiling": ["09"],
            "fire suppression": ["21"],
            "sprinkler": ["21"],
            "plumbing": ["22"],
            "hvac": ["23"],
            "heating": ["23"],
            "ventilation": ["23"],
            "air conditioning": ["23"],
            "electrical": ["26"],
            "lighting": ["26"],
            "power": ["26"]
        }
        
        for keyword, codes in division_keywords.items():
            if keyword in query_lower:
                criteria.division_codes = (criteria.division_codes or []) + codes
        
        # Remove duplicates
        if criteria.content_types:
            criteria.content_types = list(set(criteria.content_types))
        if criteria.disciplines:
            criteria.disciplines = list(set(criteria.disciplines))
        if criteria.division_codes:
            criteria.division_codes = list(set(criteria.division_codes))
        
        return criteria


def create_metadata_filter() -> MetadataFilter:
    """
    Create a metadata filter instance.
    
    Returns:
        MetadataFilter instance
    """
    return MetadataFilter()