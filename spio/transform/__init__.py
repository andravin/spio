"""Functions for transforming and scanning PyTorch modules.

The transform function replaces PyTorch modules with Spio modules when
when they match. The scan_modules reports all modules that have a
matching Spio module without. .
"""

from ._transform import transform, scan_modules
