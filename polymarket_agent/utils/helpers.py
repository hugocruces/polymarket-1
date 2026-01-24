"""
Helper Utilities

Common utility functions used throughout the agent.
"""

import asyncio
import json
import logging
import sys
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the agent.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Optional custom format string
        log_file: Optional file to write logs to
        
    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)
    
    # Set specific loggers to less verbose
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return root_logger


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_currency(value: float, currency: str = "$", decimals: int = 0) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Value to format
        currency: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if value >= 1_000_000:
        return f"{currency}{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{currency}{value / 1_000:.1f}K"
    else:
        return f"{currency}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as a percentage.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON, returning default on failure.
    
    Args:
        text: JSON string to parse
        default: Value to return on parse failure
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_async(max_retries=3, delay=1.0)
        ... async def fetch_data():
        ...     # This will be retried up to 3 times on failure
        ...     return await some_api_call()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deduplicate_by_key(items: list, key: Callable) -> list:
    """
    Remove duplicates from a list based on a key function.
    
    Args:
        items: List of items
        key: Function to extract key for comparison
        
    Returns:
        Deduplicated list (preserves first occurrence)
    """
    seen = set()
    result = []
    
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    
    return result


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    
    Example:
        >>> tracker = ProgressTracker(total=100, description="Processing")
        >>> for i in range(100):
        ...     do_work(i)
        ...     tracker.update()
        >>> tracker.finish()
    """
    
    def __init__(
        self,
        total: int,
        description: str = "Progress",
        show_bar: bool = True,
    ):
        self.total = total
        self.current = 0
        self.description = description
        self.show_bar = show_bar
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current = min(self.total, self.current + n)
        self._display()
    
    def _display(self) -> None:
        """Display current progress."""
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        if self.show_bar:
            bar_length = 30
            filled = int(bar_length * self.current / self.total) if self.total > 0 else 0
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r{self.description}: [{bar}] {self.current}/{self.total} ({percent:.1f}%)", end="")
        else:
            print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%)", end="")
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.current = self.total
        self._display()
        print()  # New line
