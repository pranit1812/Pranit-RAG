"""
Specialized table processing for chunking with CSV generation.
"""
from io import StringIO
from typing import List, Optional, Tuple, Dict, Any
import re
import csv

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from models.types import Block, Chunk


class TableProcessor:
    """Specialized processor for table blocks with structure preservation."""
    
    def __init__(self):
        """Initialize table processor."""
        pass
    
    def process_table_block(self, block: Block) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Process a table block to extract text, HTML, and CSV representations.
        
        Args:
            block: Table block to process
            
        Returns:
            Tuple of (text, html, csv) representations
        """
        table_text = block["text"].strip()
        table_html = block.get("html", "")
        
        if not table_text:
            return "", None, None
        
        # Generate CSV from HTML if available
        csv_content = None
        if table_html:
            csv_content = self._html_to_csv(table_html)
        
        # If no HTML or CSV generation failed, try to parse from text
        if not csv_content:
            csv_content = self._text_to_csv(table_text)
        
        return table_text, table_html, csv_content
    
    def _html_to_csv(self, html: str) -> Optional[str]:
        """
        Convert HTML table to CSV format.
        
        Args:
            html: HTML table string
            
        Returns:
            CSV string or None if conversion fails
        """
        if not PANDAS_AVAILABLE:
            return None
            
        try:
            # Parse HTML table using pandas
            dfs = pd.read_html(StringIO(html))
            if not dfs:
                return None
            
            df = dfs[0]
            
            # Convert to CSV
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_MINIMAL)
            return csv_buffer.getvalue()
            
        except Exception:
            return None
    
    def _text_to_csv(self, text: str) -> Optional[str]:
        """
        Convert text table to CSV format using heuristics.
        
        Args:
            text: Text representation of table
            
        Returns:
            CSV string or None if conversion fails
        """
        try:
            lines = text.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Detect delimiter (tab, pipe, multiple spaces)
            delimiter = self._detect_delimiter(lines)
            if not delimiter:
                return None
            
            # Parse rows
            rows = []
            for line in lines:
                if delimiter == 'spaces':
                    # Split on multiple spaces
                    row = re.split(r'\s{2,}', line.strip())
                else:
                    row = [cell.strip() for cell in line.split(delimiter)]
                
                if row and any(cell for cell in row):  # Skip empty rows
                    rows.append(row)
            
            if not rows:
                return None
            
            # Normalize row lengths
            max_cols = max(len(row) for row in rows)
            normalized_rows = []
            for row in rows:
                while len(row) < max_cols:
                    row.append("")
                normalized_rows.append(row)
            
            # Convert to CSV
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(normalized_rows)
            return csv_buffer.getvalue()
            
        except Exception:
            return None
    
    def _detect_delimiter(self, lines: List[str]) -> Optional[str]:
        """
        Detect the delimiter used in text table.
        
        Args:
            lines: Lines of text table
            
        Returns:
            Delimiter character or 'spaces' for multiple spaces, None if not detected
        """
        if not lines:
            return None
        
        # Test common delimiters
        delimiters = ['|', '\t', ',', ';']
        
        for delimiter in delimiters:
            # Check if delimiter appears consistently across lines
            delimiter_counts = [line.count(delimiter) for line in lines[:5]]  # Check first 5 lines
            
            if (len(set(delimiter_counts)) == 1 and  # Same count in all lines
                delimiter_counts[0] > 0):  # At least one delimiter
                return delimiter
        
        # Check for multiple spaces as delimiter
        space_pattern = re.compile(r'\s{2,}')
        space_counts = [len(space_pattern.findall(line)) for line in lines[:5]]
        
        if (len(set(space_counts)) == 1 and 
            space_counts[0] > 0):
            return 'spaces'
        
        return None
    
    def split_table_with_headers(
        self, 
        table_text: str, 
        table_html: Optional[str], 
        csv_content: Optional[str],
        max_tokens: int,
        token_counter
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Split large table while preserving headers.
        
        Args:
            table_text: Text representation
            table_html: HTML representation
            csv_content: CSV representation
            max_tokens: Maximum tokens per chunk
            token_counter: Token counter instance
            
        Returns:
            List of (text, html, csv) tuples for each chunk
        """
        chunks = []
        
        # Try structured splitting with CSV
        if csv_content:
            csv_chunks = self._split_csv_with_headers(csv_content, max_tokens, token_counter)
            
            for csv_chunk in csv_chunks:
                # Convert CSV chunk back to text and HTML
                chunk_text = self._csv_to_text(csv_chunk)
                chunk_html = self._csv_to_html(csv_chunk) if table_html else None
                
                chunks.append((chunk_text, chunk_html, csv_chunk))
            
            return chunks
        
        # Fallback to text-based splitting
        return self._split_text_table_with_headers(table_text, max_tokens, token_counter)
    
    def _split_csv_with_headers(
        self, 
        csv_content: str, 
        max_tokens: int, 
        token_counter
    ) -> List[str]:
        """Split CSV content while preserving headers."""
        try:
            lines = csv_content.strip().split('\n')
            if len(lines) < 2:
                return [csv_content]
            
            header_line = lines[0]
            data_lines = lines[1:]
            
            chunks = []
            current_chunk_lines = [header_line]
            
            for line in data_lines:
                test_chunk = '\n'.join(current_chunk_lines + [line])
                
                if token_counter.count_tokens(test_chunk) <= max_tokens:
                    current_chunk_lines.append(line)
                else:
                    # Finalize current chunk
                    if len(current_chunk_lines) > 1:
                        chunks.append('\n'.join(current_chunk_lines))
                    
                    # Start new chunk with header
                    current_chunk_lines = [header_line, line]
            
            # Add final chunk
            if len(current_chunk_lines) > 1:
                chunks.append('\n'.join(current_chunk_lines))
            
            return chunks
            
        except Exception:
            return [csv_content]
    
    def _split_text_table_with_headers(
        self, 
        table_text: str, 
        max_tokens: int, 
        token_counter
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """Split text table while preserving headers."""
        lines = table_text.split('\n')
        if len(lines) < 2:
            return [(table_text, None, None)]
        
        # Assume first line is header
        header = lines[0]
        data_lines = lines[1:]
        
        chunks = []
        current_chunk_lines = [header]
        
        for line in data_lines:
            test_text = '\n'.join(current_chunk_lines + [line])
            
            if token_counter.count_tokens(test_text) <= max_tokens:
                current_chunk_lines.append(line)
            else:
                # Finalize current chunk
                if len(current_chunk_lines) > 1:
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunks.append((chunk_text, None, None))
                
                # Start new chunk with header
                current_chunk_lines = [header, line]
        
        # Add final chunk
        if len(current_chunk_lines) > 1:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append((chunk_text, None, None))
        
        return chunks
    
    def _csv_to_text(self, csv_content: str) -> str:
        """Convert CSV content to readable text format."""
        try:
            lines = csv_content.strip().split('\n')
            reader = csv.reader(lines)
            rows = list(reader)
            
            if not rows:
                return csv_content
            
            # Calculate column widths
            col_widths = []
            for i in range(len(rows[0])):
                max_width = max(len(str(row[i])) if i < len(row) else 0 for row in rows)
                col_widths.append(max(max_width, 3))  # Minimum width of 3
            
            # Format as aligned text table
            formatted_lines = []
            for row in rows:
                formatted_cells = []
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        formatted_cells.append(str(cell).ljust(col_widths[i]))
                    else:
                        formatted_cells.append(str(cell))
                formatted_lines.append(' | '.join(formatted_cells))
            
            return '\n'.join(formatted_lines)
            
        except Exception:
            return csv_content
    
    def _csv_to_html(self, csv_content: str) -> str:
        """Convert CSV content to HTML table format."""
        try:
            lines = csv_content.strip().split('\n')
            reader = csv.reader(lines)
            rows = list(reader)
            
            if not rows:
                return f"<table><tr><td>{csv_content}</td></tr></table>"
            
            html_parts = ["<table>"]
            
            # Header row
            if rows:
                header_cells = [f"<th>{cell}</th>" for cell in rows[0]]
                html_parts.append(f"<tr>{''.join(header_cells)}</tr>")
            
            # Data rows
            for row in rows[1:]:
                data_cells = [f"<td>{cell}</td>" for cell in row]
                html_parts.append(f"<tr>{''.join(data_cells)}</tr>")
            
            html_parts.append("</table>")
            return ''.join(html_parts)
            
        except Exception:
            return f"<table><tr><td>{csv_content}</td></tr></table>"