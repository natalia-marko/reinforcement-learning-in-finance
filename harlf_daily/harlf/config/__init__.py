"""
Configuration management for HARLF
"""

from .defaults import *
try:
    from .paths import *
except ImportError:
    pass

__all__ = ['defaults', 'paths']
