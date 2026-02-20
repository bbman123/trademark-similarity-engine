"""
Utility functions for data preprocessing and model operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def validate_wordmark(text: str) -> bool:
    """
    Check if a wordmark is valid (not empty/null)
    
    Args:
        text: Wordmark text to validate
        
    Returns:
        True if valid, False otherwise
    """
    if pd.isna(text):
        return False
    text = str(text).strip()
    return len(text) > 0 and text.lower() not in ['nan', 'none', 'null', '']


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and fixing common issues
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def normalize_text(text: str, lowercase: bool = True, remove_punct: bool = False) -> str:
    """
    Normalize text for processing
    
    Args:
        text: Text to normalize
        lowercase: Whether to convert to lowercase
        remove_punct: Whether to remove punctuation
        
    Returns:
        Normalized text
    """
    if not text or pd.isna(text):
        return ""
    
    text = clean_text(text)
    
    if lowercase:
        text = text.lower()
    
    if remove_punct:
        import re
        text = re.sub(r'[^\w\s]', '', text)
    
    return text


def batch_process(
    items: List[Any],
    func: callable,
    batch_size: int = 32,
    show_progress: bool = False
) -> List[Any]:
    """
    Process items in batches
    
    Args:
        items: List of items to process
        func: Function to apply to each batch
        batch_size: Size of each batch
        show_progress: Whether to show progress
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        
        if show_progress:
            print(f"Processing batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}...")
        
        batch_results = func(batch)
        results.extend(batch_results)
    
    return results


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary of statistics
    """
    arr = np.array(values)
    
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75))
    }


def format_probability(prob: float, decimals: int = 2) -> str:
    """
    Format probability as percentage
    
    Args:
        prob: Probability value (0-1)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{prob*100:.{decimals}f}%"


def get_risk_level(probability: float) -> tuple:
    """
    Determine risk level from probability
    
    Args:
        probability: Similarity probability (0-1)
        
    Returns:
        Tuple of (risk_level, emoji, color)
    """
    if probability >= 0.7:
        return ("HIGH", "🔴", "red")
    elif probability >= 0.5:
        return ("MEDIUM", "🟡", "yellow")
    else:
        return ("LOW", "🟢", "green")
