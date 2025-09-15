"""
Project context generation and management for construction RAG system.
"""
import os
import re
from typing import List, Dict, Set, Optional, Counter
from collections import Counter
from pathlib import Path

from models.types import Chunk, ProjectContext, ChunkMetadata, validate_project_context
from services.filtering import MASTERFORMAT_DIVISIONS
from utils.io_utils import read_text_file, write_text_file


class ProjectContextGenerator:
    """Generates project context from document analysis."""
    
    # Project type detection patterns
    PROJECT_TYPE_PATTERNS = {
        "Commercial Office Building": [
            r"office\s+building", r"commercial\s+office", r"office\s+complex",
            r"corporate\s+headquarters", r"business\s+center"
        ],
        "Residential Complex": [
            r"residential\s+complex", r"apartment\s+building", r"condominium",
            r"housing\s+development", r"multi-family", r"townhome"
        ],
        "Industrial Facility": [
            r"industrial\s+facility", r"manufacturing\s+plant", r"warehouse",
            r"distribution\s+center", r"factory", r"production\s+facility"
        ],
        "Healthcare Facility": [
            r"hospital", r"medical\s+center", r"clinic", r"healthcare\s+facility",
            r"surgery\s+center", r"medical\s+office"
        ],
        "Educational Facility": [
            r"school", r"university", r"college", r"educational\s+facility",
            r"classroom\s+building", r"academic\s+center"
        ],
        "Retail Facility": [
            r"retail\s+store", r"shopping\s+center", r"mall", r"restaurant",
            r"retail\s+facility", r"commercial\s+retail"
        ],
        "Mixed-Use Development": [
            r"mixed-use", r"mixed\s+use", r"multi-use", r"combined\s+use"
        ]
    }
    
    # Key systems detection patterns
    SYSTEMS_PATTERNS = {
        "HVAC": [
            r"hvac", r"heating", r"ventilation", r"air\s+conditioning",
            r"mechanical\s+system", r"climate\s+control", r"ductwork"
        ],
        "Electrical": [
            r"electrical", r"power", r"lighting", r"electrical\s+system",
            r"wiring", r"electrical\s+panel", r"circuit"
        ],
        "Plumbing": [
            r"plumbing", r"water", r"sewer", r"drainage", r"piping",
            r"plumbing\s+system", r"water\s+supply"
        ],
        "Fire Protection": [
            r"fire\s+protection", r"sprinkler", r"fire\s+suppression",
            r"fire\s+alarm", r"fire\s+safety"
        ],
        "Structural": [
            r"structural", r"foundation", r"framing", r"steel", r"concrete",
            r"structural\s+system", r"load\s+bearing"
        ],
        "Security": [
            r"security", r"access\s+control", r"surveillance", r"alarm",
            r"security\s+system"
        ],
        "Communications": [
            r"communications", r"data", r"telecommunications", r"network",
            r"fiber\s+optic", r"phone\s+system"
        ]
    }
    
    def __init__(self):
        """Initialize the project context generator."""
        pass
    
    def generate_context(self, chunks: List[Chunk], project_name: str) -> ProjectContext:
        """
        Generate project context from analyzed chunks.
        
        Args:
            chunks: List of document chunks to analyze
            project_name: Name of the project
            
        Returns:
            Generated ProjectContext
        """
        # Combine all text for analysis (preserve original case for location)
        all_text_original = " ".join([chunk["text"] for chunk in chunks])
        all_text_lower = all_text_original.lower()
        
        # Detect project type
        project_type = self._detect_project_type(all_text_lower)
        
        # Identify key systems
        key_systems = self._identify_key_systems(all_text_lower, chunks)
        
        # Detect disciplines involved
        disciplines_involved = self._detect_disciplines(chunks)
        
        # Extract location if possible (use original case)
        location = self._extract_location(all_text_original)
        
        # Generate summary
        summary = self._generate_summary(chunks, project_type, key_systems, disciplines_involved)
        
        # Create description
        description = self._generate_description(project_type, key_systems, disciplines_involved)
        
        return ProjectContext(
            project_name=project_name,
            description=description,
            project_type=project_type,
            location=location,
            key_systems=key_systems,
            disciplines_involved=disciplines_involved,
            summary=summary
        )
    
    def _detect_project_type(self, text: str) -> str:
        """
        Detect project type from text content.
        
        Args:
            text: Combined text content (lowercase)
            
        Returns:
            Detected project type
        """
        type_scores = {}
        
        for project_type, patterns in self.PROJECT_TYPE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            type_scores[project_type] = score
        
        # Return the type with highest score, or default
        if type_scores and max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return "General Construction Project"
    
    def _identify_key_systems(self, text: str, chunks: List[Chunk]) -> List[str]:
        """
        Identify key building systems from content.
        
        Args:
            text: Combined text content (lowercase)
            chunks: Document chunks for additional analysis
            
        Returns:
            List of identified key systems
        """
        system_scores = {}
        
        # Score based on text patterns
        for system, patterns in self.SYSTEMS_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            system_scores[system] = score
        
        # Additional scoring based on division codes
        division_systems = {
            "21": "Fire Protection",
            "22": "Plumbing", 
            "23": "HVAC",
            "26": "Electrical",
            "27": "Communications",
            "28": "Security"
        }
        
        for chunk in chunks:
            division_code = chunk["metadata"].get("division_code")
            if division_code and division_code in division_systems:
                system = division_systems[division_code]
                if system in system_scores:
                    system_scores[system] += 5  # Boost score for division presence
        
        # Return systems with score > 0, sorted by score
        identified_systems = [
            system for system, score in system_scores.items() 
            if score > 0
        ]
        
        # Sort by score descending
        identified_systems.sort(key=lambda s: system_scores[s], reverse=True)
        
        return identified_systems
    
    def _detect_disciplines(self, chunks: List[Chunk]) -> List[str]:
        """
        Detect involved disciplines from chunk metadata.
        
        Args:
            chunks: Document chunks to analyze
            
        Returns:
            List of discipline names
        """
        discipline_codes = set()
        discipline_names = {
            "A": "Architectural",
            "S": "Structural", 
            "M": "Mechanical",
            "E": "Electrical",
            "P": "Plumbing",
            "FP": "Fire Protection",
            "EL": "Electrical"  # Alternative electrical code
        }
        
        # Collect discipline codes from metadata
        for chunk in chunks:
            discipline = chunk["metadata"].get("discipline")
            if discipline:
                discipline_codes.add(discipline)
        
        # Also infer from division codes
        division_to_discipline = {
            "03": "Structural",  # Concrete
            "04": "Structural",  # Masonry
            "05": "Structural",  # Metals
            "21": "Fire Protection",
            "22": "Plumbing",
            "23": "Mechanical",
            "26": "Electrical",
            "27": "Electrical",  # Communications
            "28": "Electrical"   # Electronic Safety
        }
        
        inferred_disciplines = set()
        for chunk in chunks:
            division_code = chunk["metadata"].get("division_code")
            if division_code and division_code in division_to_discipline:
                inferred_disciplines.add(division_to_discipline[division_code])
        
        # Combine explicit disciplines and inferred ones
        all_disciplines = set()
        
        # Add explicit disciplines
        for code in discipline_codes:
            if code in discipline_names:
                all_disciplines.add(discipline_names[code])
        
        # Add inferred disciplines
        all_disciplines.update(inferred_disciplines)
        
        return sorted(list(all_disciplines))
    
    def _extract_location(self, text: str) -> Optional[str]:
        """
        Extract project location from text if possible.
        
        Args:
            text: Combined text content (original case)
            
        Returns:
            Extracted location or None
        """
        # Look for common location patterns
        location_patterns = [
            r"project\s+location[:\s]+([^.\n]+?)(?:\s+building|\s+specifications|$)",
            r"site\s+address[:\s]+([^.\n]+?)(?:\s+building|\s+specifications|$)",
            r"located\s+(?:at|in)\s+([^.\n]+?)(?:\s+building|\s+specifications|$)",
            r"address[:\s]+([^.\n]+?)(?:\s+building|\s+specifications|$)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up the location string
                location = re.sub(r'\s+', ' ', location)
                # Remove trailing punctuation
                location = location.rstrip('.,;')
                if len(location) > 5 and len(location) < 100:  # Reasonable length
                    return location
        
        return None
    
    def _generate_summary(self, chunks: List[Chunk], project_type: str, 
                         key_systems: List[str], disciplines: List[str]) -> str:
        """
        Generate project summary from analysis.
        
        Args:
            chunks: Document chunks
            project_type: Detected project type
            key_systems: Identified key systems
            disciplines: Involved disciplines
            
        Returns:
            Generated summary text
        """
        # Count document types
        doc_counts = Counter()
        division_counts = Counter()
        
        for chunk in chunks:
            content_type = chunk["metadata"]["content_type"]
            doc_counts[content_type] += 1
            
            division_code = chunk["metadata"].get("division_code")
            if division_code:
                division_counts[division_code] += 1
        
        # Build summary
        summary_parts = []
        
        # Project type and scope
        summary_parts.append(f"This is a {project_type.lower()} project")
        
        # Document composition
        if doc_counts:
            doc_types = []
            if doc_counts.get("Drawing", 0) > 0:
                doc_types.append(f"{doc_counts['Drawing']} drawing sheets")
            if doc_counts.get("SpecSection", 0) > 0:
                doc_types.append(f"{doc_counts['SpecSection']} specification sections")
            if doc_counts.get("Table", 0) > 0:
                doc_types.append(f"{doc_counts['Table']} tables")
            
            if doc_types:
                summary_parts.append(f"containing {', '.join(doc_types)}")
        
        # Key systems
        if key_systems:
            if len(key_systems) <= 3:
                systems_text = ", ".join(key_systems)
            else:
                systems_text = f"{', '.join(key_systems[:3])}, and {len(key_systems) - 3} other systems"
            summary_parts.append(f"The project involves {systems_text}")
        
        # Disciplines
        if disciplines:
            if len(disciplines) <= 3:
                disciplines_text = ", ".join(disciplines)
            else:
                disciplines_text = f"{', '.join(disciplines[:3])}, and {len(disciplines) - 3} other disciplines"
            summary_parts.append(f"with {disciplines_text} disciplines involved")
        
        # Major divisions
        if division_counts:
            top_divisions = division_counts.most_common(3)
            division_names = []
            for div_code, count in top_divisions:
                div_title = MASTERFORMAT_DIVISIONS.get(div_code, f"Division {div_code}")
                division_names.append(div_title)
            
            if division_names:
                summary_parts.append(f"Major specification divisions include {', '.join(division_names)}")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_description(self, project_type: str, key_systems: List[str], 
                            disciplines: List[str]) -> str:
        """
        Generate project description.
        
        Args:
            project_type: Detected project type
            key_systems: Identified key systems
            disciplines: Involved disciplines
            
        Returns:
            Generated description text
        """
        desc_parts = [f"A {project_type.lower()} project"]
        
        if key_systems:
            systems_text = ", ".join(key_systems[:5])  # Limit to top 5
            desc_parts.append(f"involving {systems_text} systems")
        
        if disciplines:
            disciplines_text = ", ".join(disciplines[:5])  # Limit to top 5
            desc_parts.append(f"with {disciplines_text} design disciplines")
        
        return " ".join(desc_parts) + "."


class ProjectContextManager:
    """Manages project context persistence and loading."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize project context manager.
        
        Args:
            storage_dir: Base storage directory for projects
        """
        self.storage_dir = Path(storage_dir)
    
    def save_context(self, project_id: str, context: ProjectContext) -> None:
        """
        Save project context to markdown file.
        
        Args:
            project_id: Project identifier
            context: ProjectContext to save
            
        Raises:
            ValueError: If context validation fails
            IOError: If file write fails
        """
        # Validate context before saving
        if not validate_project_context(context):
            raise ValueError("Invalid project context structure")
        
        project_dir = self.storage_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        context_file = project_dir / "project_context.md"
        
        # Generate markdown content
        markdown_content = self._context_to_markdown(context)
        
        # Write to file
        try:
            write_text_file(str(context_file), markdown_content)
        except Exception as e:
            raise IOError(f"Failed to save project context: {e}")
    
    def load_context(self, project_id: str) -> Optional[ProjectContext]:
        """
        Load project context from markdown file.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Loaded ProjectContext or None if not found
        """
        context_file = self.storage_dir / project_id / "project_context.md"
        
        if not context_file.exists():
            return None
        
        try:
            markdown_content = read_text_file(str(context_file))
            context = self._markdown_to_context(markdown_content)
            
            # Validate loaded context
            if not validate_project_context(context):
                return None
                
            return context
        except Exception:
            return None
    
    def context_exists(self, project_id: str) -> bool:
        """
        Check if project context file exists.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if context file exists
        """
        context_file = self.storage_dir / project_id / "project_context.md"
        return context_file.exists()
    
    def update_context(self, project_id: str, updates: Dict[str, any]) -> bool:
        """
        Update specific fields in project context.
        
        Args:
            project_id: Project identifier
            updates: Dictionary of field updates
            
        Returns:
            True if update successful, False otherwise
        """
        # Load existing context
        context = self.load_context(project_id)
        if not context:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if key in context:
                context[key] = value
        
        # Validate and save
        try:
            self.save_context(project_id, context)
            return True
        except (ValueError, IOError):
            return False
    
    def delete_context(self, project_id: str) -> bool:
        """
        Delete project context file.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        context_file = self.storage_dir / project_id / "project_context.md"
        
        try:
            if context_file.exists():
                context_file.unlink()
            return True
        except Exception:
            return False
    
    def _context_to_markdown(self, context: ProjectContext) -> str:
        """
        Convert ProjectContext to markdown format.
        
        Args:
            context: ProjectContext to convert
            
        Returns:
            Markdown formatted string
        """
        lines = [
            f"# {context['project_name']}",
            "",
            "## Project Information",
            "",
            f"**Project Type:** {context['project_type']}",
            "",
            f"**Description:** {context['description']}",
            ""
        ]
        
        if context.get('location'):
            lines.extend([
                f"**Location:** {context['location']}",
                ""
            ])
        
        if context['key_systems']:
            lines.extend([
                "## Key Systems",
                ""
            ])
            for system in context['key_systems']:
                lines.append(f"- {system}")
            lines.append("")
        
        if context['disciplines_involved']:
            lines.extend([
                "## Disciplines Involved",
                ""
            ])
            for discipline in context['disciplines_involved']:
                lines.append(f"- {discipline}")
            lines.append("")
        
        lines.extend([
            "## Project Summary",
            "",
            context['summary'],
            ""
        ])
        
        return "\n".join(lines)
    
    def _markdown_to_context(self, markdown_content: str) -> ProjectContext:
        """
        Parse markdown content back to ProjectContext.
        
        Args:
            markdown_content: Markdown formatted string
            
        Returns:
            Parsed ProjectContext
        """
        lines = markdown_content.split('\n')
        
        # Initialize context with defaults
        context = ProjectContext(
            project_name="",
            description="",
            project_type="General Construction Project",
            location=None,
            key_systems=[],
            disciplines_involved=[],
            summary=""
        )
        
        current_section = None
        summary_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Extract project name from title
            if line.startswith('# '):
                context['project_name'] = line[2:].strip()
            
            # Track sections
            elif line.startswith('## '):
                current_section = line[3:].strip().lower()
                if current_section == "project summary":
                    summary_lines = []
            
            # Extract project information
            elif line.startswith('**Project Type:**'):
                value = line.split(':', 1)[1].strip()
                # Remove markdown formatting
                value = value.replace('**', '').strip()
                context['project_type'] = value
            elif line.startswith('**Description:**'):
                value = line.split(':', 1)[1].strip()
                # Remove markdown formatting
                value = value.replace('**', '').strip()
                context['description'] = value
            elif line.startswith('**Location:**'):
                value = line.split(':', 1)[1].strip()
                # Remove markdown formatting
                value = value.replace('**', '').strip()
                context['location'] = value
            
            # Extract lists
            elif line.startswith('- '):
                item = line[2:].strip()
                if current_section == "key systems":
                    context['key_systems'].append(item)
                elif current_section == "disciplines involved":
                    context['disciplines_involved'].append(item)
            
            # Collect summary lines
            elif current_section == "project summary" and line and not line.startswith('#'):
                summary_lines.append(line)
        
        # Join summary lines
        if summary_lines:
            context['summary'] = ' '.join(summary_lines)
        
        return context


class ProjectContextCache:
    """Cache for project contexts to improve performance."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize context cache.
        
        Args:
            max_size: Maximum number of contexts to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, ProjectContext] = {}
        self._access_order: List[str] = []
    
    def get(self, project_id: str) -> Optional[ProjectContext]:
        """
        Get context from cache.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Cached ProjectContext or None
        """
        if project_id in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(project_id)
            self._access_order.append(project_id)
            return self._cache[project_id]
        
        return None
    
    def put(self, project_id: str, context: ProjectContext) -> None:
        """
        Put context in cache.
        
        Args:
            project_id: Project identifier
            context: ProjectContext to cache
        """
        # Remove if already exists
        if project_id in self._cache:
            self._access_order.remove(project_id)
        
        # Add to cache
        self._cache[project_id] = context
        self._access_order.append(project_id)
        
        # Evict oldest if over limit
        while len(self._cache) > self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
    
    def remove(self, project_id: str) -> None:
        """
        Remove context from cache.
        
        Args:
            project_id: Project identifier
        """
        if project_id in self._cache:
            del self._cache[project_id]
            self._access_order.remove(project_id)
    
    def clear(self) -> None:
        """Clear all cached contexts."""
        self._cache.clear()
        self._access_order.clear()


class QueryEnhancer:
    """Enhances user queries with project context for better retrieval."""
    
    # Construction terminology mappings
    CONSTRUCTION_TERMS = {
        # HVAC terms
        "hvac": ["heating", "ventilation", "air conditioning", "mechanical system", "climate control"],
        "ductwork": ["ducts", "air distribution", "ventilation system"],
        "vav": ["variable air volume", "air handling"],
        
        # Electrical terms
        "electrical": ["power", "wiring", "circuits", "electrical system"],
        "panel": ["electrical panel", "distribution panel", "switchboard"],
        "conduit": ["electrical conduit", "wiring raceway"],
        
        # Plumbing terms
        "plumbing": ["water", "sewer", "drainage", "piping system"],
        "fixture": ["plumbing fixture", "water fixture"],
        "valve": ["shutoff valve", "control valve"],
        
        # Structural terms
        "structural": ["framing", "foundation", "load bearing", "structural system"],
        "beam": ["structural beam", "support beam"],
        "column": ["structural column", "support column"],
        
        # Fire protection terms
        "sprinkler": ["fire sprinkler", "fire suppression", "fire protection"],
        "alarm": ["fire alarm", "smoke detector", "fire safety"],
        
        # General construction terms
        "spec": ["specification", "technical specification"],
        "drawing": ["plan", "blueprint", "construction drawing"],
        "detail": ["construction detail", "technical detail"],
        "schedule": ["construction schedule", "material schedule"]
    }
    
    # Project type specific terminology
    PROJECT_TYPE_TERMS = {
        "Commercial Office Building": {
            "office": ["workspace", "commercial space", "tenant space"],
            "lobby": ["entrance", "reception area", "main entrance"],
            "elevator": ["vertical transportation", "passenger elevator"],
            "parking": ["parking garage", "vehicle parking", "parking structure"]
        },
        "Healthcare Facility": {
            "patient": ["patient room", "patient care", "medical care"],
            "surgery": ["operating room", "surgical suite", "OR"],
            "medical": ["healthcare", "clinical", "medical equipment"],
            "isolation": ["isolation room", "negative pressure", "infection control"]
        },
        "Educational Facility": {
            "classroom": ["learning space", "teaching space", "academic space"],
            "lab": ["laboratory", "science lab", "computer lab"],
            "cafeteria": ["dining hall", "food service", "kitchen"],
            "gym": ["gymnasium", "athletic facility", "sports facility"]
        },
        "Industrial Facility": {
            "production": ["manufacturing", "assembly", "industrial process"],
            "warehouse": ["storage", "distribution", "material handling"],
            "equipment": ["machinery", "industrial equipment", "process equipment"],
            "safety": ["industrial safety", "worker safety", "hazard control"]
        }
    }
    
    def __init__(self):
        """Initialize the query enhancer."""
        pass
    
    def enhance_query(self, query: str, project_context: ProjectContext) -> str:
        """
        Enhance user query with project context for better retrieval.
        
        Args:
            query: Original user query
            project_context: Project context for enhancement
            
        Returns:
            Enhanced query string
        """
        enhanced_parts = [query]
        
        # Add project type context
        enhanced_parts.append(f"Project type: {project_context['project_type']}")
        
        # Add relevant system context
        query_lower = query.lower()
        relevant_systems = self._find_relevant_systems(query_lower, project_context['key_systems'])
        if relevant_systems:
            enhanced_parts.append(f"Related systems: {', '.join(relevant_systems)}")
        
        # Add discipline context
        relevant_disciplines = self._find_relevant_disciplines(query_lower, project_context['disciplines_involved'])
        if relevant_disciplines:
            enhanced_parts.append(f"Related disciplines: {', '.join(relevant_disciplines)}")
        
        # Expand technical terms
        expanded_terms = self._expand_technical_terms(query_lower, project_context['project_type'])
        if expanded_terms:
            enhanced_parts.extend(expanded_terms)
        
        return " ".join(enhanced_parts)
    
    def disambiguate_terms(self, query: str, project_context: ProjectContext) -> str:
        """
        Disambiguate technical terms based on project context.
        
        Args:
            query: User query with potentially ambiguous terms
            project_context: Project context for disambiguation
            
        Returns:
            Query with disambiguated terms
        """
        disambiguated = query
        query_lower = query.lower()
        
        # Disambiguate based on project type
        project_type = project_context['project_type']
        
        # Common ambiguous terms
        disambiguations = {
            "panel": {
                "default": "electrical panel",
                "Healthcare Facility": "medical gas panel or electrical panel",
                "Industrial Facility": "control panel or electrical panel"
            },
            "system": {
                "default": "building system",
                "Healthcare Facility": "medical system or building system",
                "Industrial Facility": "process system or building system"
            },
            "equipment": {
                "default": "building equipment",
                "Healthcare Facility": "medical equipment",
                "Industrial Facility": "process equipment or industrial equipment"
            }
        }
        
        for term, contexts in disambiguations.items():
            if term in query_lower:
                if project_type in contexts:
                    replacement = contexts[project_type]
                else:
                    replacement = contexts["default"]
                
                # Replace with more specific term
                disambiguated = re.sub(
                    rf'\b{term}\b', 
                    replacement, 
                    disambiguated, 
                    flags=re.IGNORECASE
                )
        
        return disambiguated
    
    def add_domain_knowledge(self, query: str, project_context: ProjectContext) -> str:
        """
        Add construction domain knowledge to enhance query understanding.
        
        Args:
            query: User query
            project_context: Project context
            
        Returns:
            Query enhanced with domain knowledge
        """
        enhanced_parts = [query]
        query_lower = query.lower()
        
        # Add MasterFormat context for specification queries
        if any(word in query_lower for word in ["spec", "specification", "section", "division"]):
            enhanced_parts.append("MasterFormat specification document")
        
        # Add drawing context for visual queries
        if any(word in query_lower for word in ["drawing", "plan", "detail", "sheet"]):
            enhanced_parts.append("construction drawing or architectural plan")
        
        # Add system-specific context
        for system in project_context['key_systems']:
            system_lower = system.lower()
            if system_lower in query_lower:
                if system_lower == "hvac":
                    enhanced_parts.append("heating ventilation air conditioning mechanical system")
                elif system_lower == "electrical":
                    enhanced_parts.append("electrical power lighting system")
                elif system_lower == "plumbing":
                    enhanced_parts.append("water supply drainage plumbing system")
                elif system_lower == "fire protection":
                    enhanced_parts.append("fire suppression sprinkler alarm system")
        
        return " ".join(enhanced_parts)
    
    def _find_relevant_systems(self, query: str, key_systems: List[str]) -> List[str]:
        """
        Find systems relevant to the query.
        
        Args:
            query: Query text (lowercase)
            key_systems: List of project key systems
            
        Returns:
            List of relevant systems
        """
        relevant = []
        
        for system in key_systems:
            system_lower = system.lower()
            
            # Direct mention
            if system_lower in query:
                relevant.append(system)
                continue
            
            # Check related terms
            if system_lower in self.CONSTRUCTION_TERMS:
                related_terms = self.CONSTRUCTION_TERMS[system_lower]
                if any(term in query for term in related_terms):
                    relevant.append(system)
        
        return relevant
    
    def _find_relevant_disciplines(self, query: str, disciplines: List[str]) -> List[str]:
        """
        Find disciplines relevant to the query.
        
        Args:
            query: Query text (lowercase)
            disciplines: List of project disciplines
            
        Returns:
            List of relevant disciplines
        """
        relevant = []
        
        discipline_keywords = {
            "architectural": ["architectural", "building", "space", "room", "layout"],
            "structural": ["structural", "foundation", "beam", "column", "load"],
            "mechanical": ["mechanical", "hvac", "heating", "cooling", "ventilation"],
            "electrical": ["electrical", "power", "lighting", "wiring", "circuit"],
            "plumbing": ["plumbing", "water", "sewer", "drainage", "pipe"],
            "fire protection": ["fire", "sprinkler", "suppression", "alarm", "safety"]
        }
        
        for discipline in disciplines:
            discipline_lower = discipline.lower()
            
            # Direct mention
            if discipline_lower in query:
                relevant.append(discipline)
                continue
            
            # Check keywords
            if discipline_lower in discipline_keywords:
                keywords = discipline_keywords[discipline_lower]
                if any(keyword in query for keyword in keywords):
                    relevant.append(discipline)
        
        return relevant
    
    def _expand_technical_terms(self, query: str, project_type: str) -> List[str]:
        """
        Expand technical terms based on project context.
        
        Args:
            query: Query text (lowercase)
            project_type: Type of construction project
            
        Returns:
            List of expanded term phrases
        """
        expansions = []
        
        # General construction term expansions
        for term, related in self.CONSTRUCTION_TERMS.items():
            if term in query:
                expansions.extend(related)
        
        # Project-specific term expansions
        if project_type in self.PROJECT_TYPE_TERMS:
            project_terms = self.PROJECT_TYPE_TERMS[project_type]
            for term, related in project_terms.items():
                if term in query:
                    expansions.extend(related)
        
        return expansions[:5]  # Limit to avoid over-expansion