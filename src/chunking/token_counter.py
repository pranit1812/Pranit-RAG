"""
Token counting utilities using tiktoken for OpenAI compatibility.
"""
import tiktoken
from typing import Optional


class TokenCounter:
    """Token counter using tiktoken for OpenAI model compatibility."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize token counter for specific model.
        
        Args:
            model_name: OpenAI model name for encoding
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception:
            # Fallback to rough estimation if encoding fails
            return len(text.split()) * 1.3  # Rough approximation
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        if not text or max_tokens <= 0:
            return ""
        
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        except Exception:
            # Fallback to character-based truncation
            words = text.split()
            estimated_tokens = len(words) * 1.3
            if estimated_tokens <= max_tokens:
                return text
            
            # Rough truncation based on word count
            target_words = int(max_tokens / 1.3)
            return " ".join(words[:target_words])
    
    def split_by_sentences(self, text: str, max_tokens: int) -> list[str]:
        """
        Split text by sentences while respecting token limits.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Simple sentence splitting (can be enhanced with nltk/spacy)
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text as last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence exceeds limit, truncate it
                if self.count_tokens(sentence) > max_tokens:
                    current_chunk = self.truncate_to_tokens(sentence, max_tokens)
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks