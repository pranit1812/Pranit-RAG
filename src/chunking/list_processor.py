"""
Specialized list processing for chunking with context preservation.
"""
import re
from typing import List, Tuple, Optional
from models.types import Block


class ListProcessor:
    """Specialized processor for list blocks with context preservation."""
    
    def __init__(self):
        """Initialize list processor."""
        # Regex patterns for different list types
        self.list_patterns = {
            'bullet': re.compile(r'^\s*[-*•]\s+'),
            'numbered': re.compile(r'^\s*\d+[\.)]\s+'),
            'lettered': re.compile(r'^\s*[a-zA-Z][\.)]\s+'),
            'roman': re.compile(r'^\s*[ivxlcdm]+[\.)]\s+', re.IGNORECASE),
            'nested': re.compile(r'^\s{2,}[-*•]\s+|^\s{2,}\d+[\.)]\s+')
        }
    
    def process_list_block(self, block: Block) -> Tuple[str, List[str], List[str]]:
        """
        Process a list block to extract intro, items, and structure.
        
        Args:
            block: List block to process
            
        Returns:
            Tuple of (full_text, intro_lines, list_items)
        """
        list_text = block["text"].strip()
        if not list_text:
            return "", [], []
        
        lines = list_text.split('\n')
        intro_lines = []
        list_items = []
        
        # Separate intro from list items
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
                
            if self._is_list_item(line):
                list_items.append(line)
            elif not list_items:  # Before any list items
                intro_lines.append(line)
            else:  # After list items started - could be continuation
                if self._is_continuation_line(line):
                    # Append to last list item
                    if list_items:
                        list_items[-1] += '\n' + line
                else:
                    list_items.append(line)
        
        return list_text, intro_lines, list_items
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        for pattern in self.list_patterns.values():
            if pattern.match(line):
                return True
        return False
    
    def _is_continuation_line(self, line: str) -> bool:
        """Check if line is a continuation of previous list item."""
        # Lines that start with significant indentation but no list marker
        if re.match(r'^\s{4,}[^\s\-\*•\d]', line):
            return True
        return False
    
    def split_list_with_context(
        self, 
        intro_lines: List[str], 
        list_items: List[str], 
        max_tokens: int, 
        token_counter
    ) -> List[Tuple[str, List[str]]]:
        """
        Split large list while preserving intro context.
        
        Args:
            intro_lines: Introduction/title lines
            list_items: List item lines
            max_tokens: Maximum tokens per chunk
            token_counter: Token counter instance
            
        Returns:
            List of (intro_text, items) tuples for each chunk
        """
        if not list_items:
            intro_text = '\n'.join(intro_lines)
            return [(intro_text, [])]
        
        intro_text = '\n'.join(intro_lines)
        chunks = []
        current_items = []
        
        for item in list_items:
            test_items = current_items + [item]
            
            # Build test chunk with intro
            if intro_text:
                test_text = intro_text + '\n\n' + '\n'.join(test_items)
            else:
                test_text = '\n'.join(test_items)
            
            if token_counter.count_tokens(test_text) <= max_tokens:
                current_items.append(item)
            else:
                # Finalize current chunk
                if current_items:
                    chunks.append((intro_text, current_items.copy()))
                
                # Start new chunk
                # Check if single item with intro exceeds limit
                single_item_text = intro_text + '\n\n' + item if intro_text else item
                if token_counter.count_tokens(single_item_text) > max_tokens:
                    # Split the item itself
                    split_items = self._split_long_item(item, intro_text, max_tokens, token_counter)
                    for split_item in split_items:
                        chunks.append((intro_text, [split_item]))
                    current_items = []
                else:
                    current_items = [item]
        
        # Add final chunk
        if current_items:
            chunks.append((intro_text, current_items))
        
        return chunks
    
    def _split_long_item(
        self, 
        item: str, 
        intro_text: str, 
        max_tokens: int, 
        token_counter
    ) -> List[str]:
        """Split a long list item that exceeds token limit."""
        # Try to split by sentences within the item
        sentences = self._split_item_sentences(item)
        
        if len(sentences) <= 1:
            # Can't split further, truncate
            available_tokens = max_tokens - token_counter.count_tokens(intro_text + '\n\n')
            return [token_counter.truncate_to_tokens(item, max(available_tokens, 50))]
        
        # Group sentences into sub-items
        sub_items = []
        current_sub_item = ""
        
        # Extract list marker from original item
        marker_match = None
        for pattern in self.list_patterns.values():
            match = pattern.match(item)
            if match:
                marker_match = match
                break
        
        marker = marker_match.group() if marker_match else "• "
        
        for sentence in sentences:
            if not current_sub_item:
                test_sub_item = marker + sentence
            else:
                test_sub_item = current_sub_item + " " + sentence
            
            test_text = intro_text + '\n\n' + test_sub_item if intro_text else test_sub_item
            
            if token_counter.count_tokens(test_text) <= max_tokens:
                current_sub_item = test_sub_item
            else:
                if current_sub_item:
                    sub_items.append(current_sub_item)
                current_sub_item = marker + sentence
        
        if current_sub_item:
            sub_items.append(current_sub_item)
        
        return sub_items
    
    def _split_item_sentences(self, item: str) -> List[str]:
        """Split list item into sentences."""
        # Remove list marker first
        for pattern in self.list_patterns.values():
            match = pattern.match(item)
            if match:
                item = item[match.end():]
                break
        
        # Simple sentence splitting
        sentences = []
        current_sentence = ""
        
        for char in item:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def detect_list_type(self, list_items: List[str]) -> str:
        """
        Detect the type of list based on items.
        
        Args:
            list_items: List of item strings
            
        Returns:
            List type ('bullet', 'numbered', 'lettered', 'roman', 'mixed')
        """
        if not list_items:
            return 'unknown'
        
        types_found = set()
        
        for item in list_items[:5]:  # Check first 5 items
            for list_type, pattern in self.list_patterns.items():
                if pattern.match(item):
                    types_found.add(list_type)
                    break
        
        if len(types_found) == 1:
            return list(types_found)[0]
        elif len(types_found) > 1:
            return 'mixed'
        else:
            return 'unknown'
    
    def preserve_list_structure(self, intro_text: str, items: List[str]) -> str:
        """
        Combine intro and items while preserving list structure.
        
        Args:
            intro_text: Introduction text
            items: List items
            
        Returns:
            Combined text with proper formatting
        """
        parts = []
        
        if intro_text.strip():
            parts.append(intro_text.strip())
        
        if items:
            # Ensure proper spacing between intro and list
            if parts:
                parts.append("")  # Empty line
            
            parts.extend(items)
        
        return '\n'.join(parts)