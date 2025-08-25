#!/usr/bin/env python3
"""
Arabic Text Preprocessing Utilities

This module provides comprehensive text preprocessing functions specifically
designed for Arabic language processing in the context of voice cloning and TTS.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ArabicTextStats:
    """Statistics about Arabic text processing."""
    original_length: int
    processed_length: int
    num_diacritics_removed: int
    num_punctuation_normalized: int
    num_numbers_normalized: int
    num_whitespace_normalized: int


class ArabicTextProcessor:
    """
    Comprehensive Arabic text processor for voice cloning applications.
    
    Handles diacritics removal, punctuation normalization, number processing,
    and other Arabic-specific text transformations.
    """
    
    # Arabic diacritics (tashkeel) pattern
    ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0640]')
    
    # Arabic punctuation mappings
    ARABIC_PUNCTUATION_MAP = {
        '،': ',',    # Arabic comma
        '؟': '?',    # Arabic question mark  
        '؛': ';',    # Arabic semicolon
        '٪': '%',    # Arabic percent sign
        '٫': '.',    # Arabic decimal separator
        '٬': ',',    # Arabic thousands separator
        '؍': ',',    # Arabic date separator
        '؎': ',',    # Arabic footnote marker
        '؏': '*',    # Arabic sign safha
        '؞': '...',  # Arabic triple dot punctuation
        '؎': '.',    # Arabic footnote marker
        '؏': '*',    # Arabic sign safha
    }
    
    # Arabic quotation marks
    ARABIC_QUOTES_MAP = {
        '«': '"',    # Left-pointing double angle quotation mark
        '»': '"',    # Right-pointing double angle quotation mark
        '„': '"',    # Double low-9 quotation mark
        '"': '"',    # Left double quotation mark
        '"': '"',    # Right double quotation mark
        ''': "'",    # Left single quotation mark
        ''': "'",    # Right single quotation mark
    }
    
    # Arabic number mappings (Eastern Arabic numerals to Western)
    ARABIC_NUMBERS_MAP = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    
    # Persian/Urdu number mappings  
    PERSIAN_NUMBERS_MAP = {
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
    }
    
    # Common Arabic abbreviations and expansions
    ARABIC_ABBREVIATIONS = {
        'د.': 'دكتور',
        'أ.د.': 'أستاذ دكتور', 
        'م.': 'متر',
        'كم': 'كيلومتر',
        'كغ': 'كيلوغرام',
        'ص': 'صفحة',
        'ج': 'جزء',
        'ق.م': 'قبل الميلاد',
        'م.': 'ميلادي',
        'هـ': 'هجري',
    }
    
    # Arabic letters that should be normalized
    ARABIC_LETTER_NORMALIZATION = {
        'إ': 'ا',    # Alef with hamza below to alef
        'أ': 'ا',    # Alef with hamza above to alef
        'آ': 'ا',    # Alef with madda to alef
        'ة': 'ه',    # Teh marbuta to heh
        'ى': 'ي',    # Alef maksura to yeh
    }
    
    def __init__(
        self,
        remove_diacritics: bool = True,
        normalize_punctuation: bool = True,
        normalize_numbers: bool = True,
        normalize_whitespace: bool = True,
        expand_abbreviations: bool = False,
        normalize_letters: bool = False,
        preserve_structure: bool = True,
    ):
        """
        Initialize Arabic text processor.
        
        Args:
            remove_diacritics: Remove Arabic diacritics (tashkeel)
            normalize_punctuation: Convert Arabic punctuation to standard forms
            normalize_numbers: Convert Arabic/Persian numerals to Western numerals
            normalize_whitespace: Clean up whitespace
            expand_abbreviations: Expand common Arabic abbreviations
            normalize_letters: Normalize similar Arabic letters
            preserve_structure: Keep paragraph and sentence structure
        """
        self.remove_diacritics = remove_diacritics
        self.normalize_punctuation = normalize_punctuation
        self.normalize_numbers = normalize_numbers
        self.normalize_whitespace = normalize_whitespace
        self.expand_abbreviations = expand_abbreviations
        self.normalize_letters = normalize_letters
        self.preserve_structure = preserve_structure
    
    def process(self, text: str) -> Tuple[str, ArabicTextStats]:
        """
        Process Arabic text with selected normalization options.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Tuple of (processed_text, statistics)
        """
        if not text or not isinstance(text, str):
            return "", ArabicTextStats(0, 0, 0, 0, 0, 0)
        
        original_text = text
        stats = ArabicTextStats(
            original_length=len(original_text),
            processed_length=0,
            num_diacritics_removed=0,
            num_punctuation_normalized=0,
            num_numbers_normalized=0,
            num_whitespace_normalized=0,
        )
        
        # Remove diacritics
        if self.remove_diacritics:
            text, stats.num_diacritics_removed = self._remove_diacritics(text)
        
        # Normalize punctuation
        if self.normalize_punctuation:
            text, stats.num_punctuation_normalized = self._normalize_punctuation(text)
        
        # Normalize numbers
        if self.normalize_numbers:
            text, stats.num_numbers_normalized = self._normalize_numbers(text)
        
        # Expand abbreviations
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        # Normalize letters
        if self.normalize_letters:
            text = self._normalize_letters(text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text, stats.num_whitespace_normalized = self._normalize_whitespace(text)
        
        # Final cleanup
        text = self._final_cleanup(text)
        
        stats.processed_length = len(text)
        return text, stats
    
    def _remove_diacritics(self, text: str) -> Tuple[str, int]:
        """Remove Arabic diacritics (tashkeel)."""
        original_length = len(text)
        text = self.ARABIC_DIACRITICS.sub('', text)
        removed_count = original_length - len(text)
        return text, removed_count
    
    def _normalize_punctuation(self, text: str) -> Tuple[str, int]:
        """Normalize Arabic punctuation to standard forms."""
        changes = 0
        
        # Convert Arabic punctuation
        for arab_punct, standard_punct in self.ARABIC_PUNCTUATION_MAP.items():
            if arab_punct in text:
                text = text.replace(arab_punct, standard_punct)
                changes += 1
        
        # Convert Arabic quotes
        for arab_quote, standard_quote in self.ARABIC_QUOTES_MAP.items():
            if arab_quote in text:
                text = text.replace(arab_quote, standard_quote)
                changes += 1
        
        return text, changes
    
    def _normalize_numbers(self, text: str) -> Tuple[str, int]:
        """Convert Arabic and Persian numerals to Western numerals."""
        changes = 0
        
        # Convert Arabic-Indic digits
        for arab_digit, western_digit in self.ARABIC_NUMBERS_MAP.items():
            if arab_digit in text:
                text = text.replace(arab_digit, western_digit)
                changes += 1
        
        # Convert Persian digits
        for persian_digit, western_digit in self.PERSIAN_NUMBERS_MAP.items():
            if persian_digit in text:
                text = text.replace(persian_digit, western_digit)
                changes += 1
        
        return text, changes
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Arabic abbreviations."""
        for abbrev, expansion in self.ARABIC_ABBREVIATIONS.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text)
        
        return text
    
    def _normalize_letters(self, text: str) -> str:
        """Normalize similar Arabic letters."""
        for original, normalized in self.ARABIC_LETTER_NORMALIZATION.items():
            text = text.replace(original, normalized)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> Tuple[str, int]:
        """Normalize whitespace while preserving structure if requested."""
        original_whitespace_count = len(re.findall(r'\s+', text))
        
        if self.preserve_structure:
            # Preserve paragraph breaks (double newlines)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            # Normalize other whitespace
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r' *\n *', '\n', text)  # Clean spaces around newlines
        else:
            # Aggressive whitespace normalization
            text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        final_whitespace_count = len(re.findall(r'\s+', text))
        changes = abs(original_whitespace_count - final_whitespace_count)
        
        return text, changes
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and validation."""
        # Remove any remaining problematic characters
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0020-\u007F\n\r]', '', text)
        
        # Ensure proper sentence endings for TTS
        text = text.strip()
        if text and not text[-1] in '.!?،؟':
            text += '.'
        
        return text


def normalize_arabic_text(
    text: str,
    remove_diacritics: bool = True,
    normalize_punctuation: bool = True,
    normalize_numbers: bool = True,
    normalize_whitespace: bool = True,
    expand_abbreviations: bool = False,
    normalize_letters: bool = False,
) -> str:
    """
    Convenience function for basic Arabic text normalization.
    
    Args:
        text: Input Arabic text
        remove_diacritics: Remove Arabic diacritics
        normalize_punctuation: Normalize punctuation
        normalize_numbers: Normalize numbers
        normalize_whitespace: Normalize whitespace
        expand_abbreviations: Expand abbreviations
        normalize_letters: Normalize similar letters
        
    Returns:
        Normalized Arabic text
    """
    processor = ArabicTextProcessor(
        remove_diacritics=remove_diacritics,
        normalize_punctuation=normalize_punctuation,
        normalize_numbers=normalize_numbers,
        normalize_whitespace=normalize_whitespace,
        expand_abbreviations=expand_abbreviations,
        normalize_letters=normalize_letters,
    )
    
    processed_text, _ = processor.process(text)
    return processed_text


def detect_arabic_content(text: str) -> Dict[str, any]:
    """
    Analyze Arabic content in text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with analysis results
    """
    if not text:
        return {
            "is_arabic": False,
            "arabic_ratio": 0.0,
            "total_chars": 0,
            "arabic_chars": 0,
            "has_diacritics": False,
            "has_arabic_numerals": False,
            "has_arabic_punctuation": False,
        }
    
    # Count Arabic characters
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    total_chars = len([c for c in text if not c.isspace()])
    arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0.0
    
    # Check for diacritics
    has_diacritics = bool(re.search(r'[\u064B-\u065F\u0670\u0640]', text))
    
    # Check for Arabic numerals
    has_arabic_numerals = bool(re.search(r'[٠-٩۰-۹]', text))
    
    # Check for Arabic punctuation
    has_arabic_punctuation = bool(re.search(r'[،؟؛٪٫٬]', text))
    
    return {
        "is_arabic": arabic_ratio > 0.5,  # More than 50% Arabic characters
        "arabic_ratio": arabic_ratio,
        "total_chars": total_chars,
        "arabic_chars": arabic_chars,
        "has_diacritics": has_diacritics,
        "has_arabic_numerals": has_arabic_numerals,
        "has_arabic_punctuation": has_arabic_punctuation,
    }


def clean_arabic_for_tts(text: str) -> str:
    """
    Clean Arabic text specifically for TTS/voice synthesis.
    
    This function applies TTS-optimized normalization including:
    - Remove diacritics (they can interfere with pronunciation models)
    - Normalize punctuation for better prosody
    - Convert numbers to Western format
    - Ensure proper sentence endings
    
    Args:
        text: Input Arabic text
        
    Returns:
        TTS-optimized Arabic text
    """
    processor = ArabicTextProcessor(
        remove_diacritics=True,
        normalize_punctuation=True,
        normalize_numbers=True,
        normalize_whitespace=True,
        expand_abbreviations=True,  # Better for pronunciation
        normalize_letters=False,   # Keep authentic pronunciation
        preserve_structure=True,
    )
    
    cleaned_text, stats = processor.process(text)
    
    # Additional TTS-specific cleaning
    # Remove parentheses and brackets (often contain non-speech content)
    cleaned_text = re.sub(r'[\(\)\[\]{}]', ' ', cleaned_text)
    
    # Normalize multiple punctuation marks
    cleaned_text = re.sub(r'[.]{2,}', '...', cleaned_text)
    cleaned_text = re.sub(r'[!]{2,}', '!', cleaned_text)
    cleaned_text = re.sub(r'[?]{2,}', '?', cleaned_text)
    
    # Clean up whitespace again
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def validate_arabic_text(text: str) -> Dict[str, any]:
    """
    Validate Arabic text quality for voice cloning.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Validation results
    """
    if not text:
        return {
            "is_valid": False,
            "issues": ["Text is empty"],
            "suggestions": ["Provide non-empty text"],
        }
    
    issues = []
    suggestions = []
    
    # Check length
    if len(text.strip()) < 10:
        issues.append("Text too short")
        suggestions.append("Provide longer text for better voice cloning")
    
    if len(text.strip()) > 1000:
        issues.append("Text very long")
        suggestions.append("Consider splitting into shorter segments")
    
    # Analyze content
    analysis = detect_arabic_content(text)
    
    if not analysis["is_arabic"]:
        issues.append("Low Arabic content ratio")
        suggestions.append("Ensure text is primarily in Arabic")
    
    # Check for problematic characters
    if re.search(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0020-\u007F\n\r]', text):
        issues.append("Contains non-Arabic/non-ASCII characters")
        suggestions.append("Remove or replace special characters")
    
    # Check sentence structure
    sentences = re.split(r'[.!?،؟]', text)
    avg_sentence_length = sum(len(s.strip().split()) for s in sentences) / len(sentences) if sentences else 0
    
    if avg_sentence_length > 30:
        issues.append("Very long sentences")
        suggestions.append("Break down into shorter sentences for better prosody")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
        "analysis": analysis,
        "avg_sentence_length": avg_sentence_length,
    }


# Utility functions for batch processing
def process_arabic_text_batch(
    texts: List[str],
    processor_config: Optional[Dict] = None
) -> List[Tuple[str, ArabicTextStats]]:
    """
    Process a batch of Arabic texts.
    
    Args:
        texts: List of Arabic texts
        processor_config: Configuration for ArabicTextProcessor
        
    Returns:
        List of (processed_text, stats) tuples
    """
    config = processor_config or {}
    processor = ArabicTextProcessor(**config)
    
    results = []
    for text in texts:
        processed_text, stats = processor.process(text)
        results.append((processed_text, stats))
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "مَرْحَباً بِكُمْ فِي نِظَامِ التَّعَرُّفِ عَلَى الصَّوْتِ",  # With diacritics
        "هذا نص باللغة العربية يحتوي على أرقام ١٢٣٤٥ وعلامات ترقيم،",  # Arabic numerals
        "سؤال: هل يمكن تحويل النص العربي؟ الجواب: نعم!",  # Arabic punctuation
        "د. محمد أحمد - أ.د. في الجامعة (قسم الهندسة)",  # Abbreviations and symbols
    ]
    
    print("Arabic Text Preprocessing Test")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        
        # Analyze content
        analysis = detect_arabic_content(text)
        print(f"Arabic content: {analysis['arabic_ratio']:.2%}")
        
        # Process text
        processor = ArabicTextProcessor()
        processed, stats = processor.process(text)
        print(f"Processed: {processed}")
        print(f"Stats: {stats}")
        
        # Clean for TTS
        tts_text = clean_arabic_for_tts(text)
        print(f"TTS-optimized: {tts_text}")
        
        # Validate
        validation = validate_arabic_text(text)
        print(f"Valid: {validation['is_valid']}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")