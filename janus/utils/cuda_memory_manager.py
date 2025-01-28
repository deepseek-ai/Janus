# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any, Tuple, List
from functools import wraps
import warnings
import time
import math
import logging
from enum import Enum

"""Memory monitoring utilities for Janus.

This module provides essential memory management for multi-modal operations,
focusing on preventing OOM issues and optimizing resource usage for
vision-language tasks.
"""

class JanusMemoryManager:
    """Memory manager tailored for multi-modal operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.warning_threshold_gb = config.get('warning_threshold_gb', 2.0)
        self.oom_threshold_gb = config.get('oom_threshold_gb', 1.0)
        self.peak_tracking = config.get('peak_tracking', True)

    def check_memory(self) -> Dict[str, float]:
        """Get current CUDA memory status."""
        if not torch.cuda.is_available():
            return {}
        return {
            'free': torch.cuda.mem_get_info()[0] / 1024**3,
            'peak': torch.cuda.max_memory_allocated() / 1024**3
        }

def monitor_memory(
    threshold_gb: float = 2.0,
    track_peak: bool = True
) -> Callable:
    """Decorator for monitoring memory in critical paths.
    
    Designed specifically for multi-modal operations where
    memory usage can spike during modality fusion.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not torch.cuda.is_available():
                return func(*args, **kwargs)

            # Track initial state
            free_before = torch.cuda.mem_get_info()[0] / 1024**3

            try:
                if free_before < threshold_gb:
                    torch.cuda.empty_cache()
                    free_after_cleanup = torch.cuda.mem_get_info()[0] / 1024**3
                    if free_after_cleanup < threshold_gb:
                        warnings.warn(
                            f"Critical memory state in {func.__name__}: "
                            f"{free_after_cleanup:.2f}GB free"
                        )

                result = func(*args, **kwargs)

                if track_peak:
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    free_after = torch.cuda.mem_get_info()[0] / 1024**3
                    print(
                        f"Memory stats for {func.__name__}:\n"
                        f"Peak usage: {peak:.2f}GB\n"
                        f"Memory delta: {free_before - free_after:.2f}GB"
                    )

                return result

            except RuntimeError as e:
                if "out of memory" in str(e):
                    free = torch.cuda.mem_get_info()[0] / 1024**3
                    raise RuntimeError(
                        f"OOM in {func.__name__}. Free memory: {free:.2f}GB\n"
                        f"Consider reducing batch size or image resolution"
                    ) from e
                raise
            finally:
                if track_peak:
                    torch.cuda.reset_peak_memory_stats()

        return wrapper
    return decorator