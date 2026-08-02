"""
Universal Logger Template - D:\\Scripts\\log_templates\\universal_logger.py

A comprehensive, production-ready logging utility with multiple backends.
Usage: import universal_logger as log or from universal_logger import setup_logger
"""

import logging
from functools import wraps
from typing import Optional, Dict, Any

# ==================== CONFIGURATION CLASS ====================
class LogConfig:
    """Centralized logging configuration for consistent setups."""
    
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - [%(levelname)-8s] - %(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(name)s - [%(levelname)-8s] - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    
    # Log level names for quick reference
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


# ==================== LOGGER DECORATORS ====================
def log_with(level: str = "INFO"):
    """Decorator to add context to function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger_name = args[0].__class__.__name__ if args else __name__
            print(f"[{level}] Calling: {func.__name__}()")
            result = func(*args, **kwargs)
            print(f"[{level}] Completed: {func.__name__}() -> {type(result).__name__}")
            return result
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations with logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        import time
                        time.sleep(delay)
                    else:
                        raise Exception(f"Attempt {attempt} failed") from e
        return wrapper
    return decorator


# ==================== UNIVERSAL LOGGER CLASS ====================
class UniversalLogger(logging.Logger):
    """Extended logger with additional utility methods."""
    
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._extra_context = {}
        
    @property
    def context(self):
        return self._extra_context
    
    def set_context(self, **kwargs):
        """Set logging context (thread-safe)."""
        self._extra_context.update(kwargs)
        
    def get_context(self) -> Dict[str, Any]:
        return self._extra_context.copy()
    
    def clear_context(self):
        """Clear all context data."""
        self._extra_context.clear()


# ==================== LOGGER FACTORY FUNCTIONS ====================
def setup_logger(
    name: str = "__main__",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    detailed_format: bool = False,
    add_handlers: bool = True
) -> UniversalLogger:
    """
    Factory function to create and configure a logger.
    
    Args:
        name: Logger name (module path or custom name)
        level: Log level string ('DEBUG', 'INFO', 'WARNING', etc.)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        detailed_format: Use detailed format with filename/line info
        add_handlers: Whether to add handlers (True by default)
    
    Returns:
        Configured UniversalLogger instance
    
    Examples:
        >>> logger = setup_logger("my_app", level="DEBUG")
        >>> logger.info("Application started")
        
        >>> # With file logging
        >>> logger = setup_logger("my_app", log_file="/var/log/myapp.log", level="INFO")
    """
    
    fmt = LogConfig.DETAILED_FORMAT if detailed_format else LogConfig.DEFAULT_FORMAT
    
    handlers = []
    
    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level))
        ch.setFormatter(logging.Formatter(fmt))
        handlers.append(ch)
    
    if log_file and add_handlers:
        try:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, level))
            fh.setFormatter(logging.Formatter(fmt))
            handlers.append(fh)
        except Exception as e:
            print(f"Warning: Could not create file handler for {log_file}")
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str = "__main__", config: Optional[Dict] = None) -> logging.Logger:
    """Simple wrapper with dict-based configuration."""
    if config is None:
        config = {}
    
    kwargs = {
        "name": name,
        "level": config.get("level", "INFO"),
        "log_file": config.get("log_file", None),
        "console_output": config.get("console_output", True),
        "detailed_format": config.get("detailed_format", False)
    }
    
    return setup_logger(**kwargs)


# ==================== CONVENIENCE FUNCTIONS ====================
def init_logging(level: str = "INFO", log_file: Optional[str] = None, name: str = "__main__"):
    """Convenience function to initialize logging in a single line."""
    return setup_logger(name=name, level=level, log_file=log_file)


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    logger = setup_logger(level="INFO", name="example")
    logger.info("This is the universal logger template!")
