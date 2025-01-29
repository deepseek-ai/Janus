from functools import wraps
from typing import Callable, Any
import torch
import warnings


def monitor_memory(
    warning_threshold_gb: float = 2.0,
    track_stats: bool = True,
    cleanup_on_warning: bool = True,
) -> Callable:
    """Memory monitoring decorator for CUDA operations.

    Args:
        warning_threshold_gb: Memory threshold in GB to trigger warnings
        track_stats: Whether to track and print memory statistics
        cleanup_on_warning: Whether to attempt memory cleanup when threshold is reached

    Returns:
        Decorator function that monitors memory usage
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not torch.cuda.is_available():
                return func(*args, **kwargs)

            # Get initial memory state
            free_before = torch.cuda.mem_get_info()[0] / 1024**3

            try:
                # Check memory state and cleanup if needed
                if free_before < warning_threshold_gb and cleanup_on_warning:
                    torch.cuda.empty_cache()
                    free_after_cleanup = torch.cuda.mem_get_info()[0] / 1024**3

                    if free_after_cleanup < warning_threshold_gb:
                        warnings.warn(
                            f"Low memory in {func.__name__}: {free_after_cleanup:.2f}GB free"
                        )

                result = func(*args, **kwargs)

                # Track memory statistics if enabled
                if track_stats:
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    free_after = torch.cuda.mem_get_info()[0] / 1024**3
                    print(
                        f"Memory stats for {func.__name__}:\n"
                        f"Peak: {peak:.2f}GB | Delta: {free_before - free_after:.2f}GB"
                    )
                    torch.cuda.reset_peak_memory_stats()

                return result

            except RuntimeError as e:
                if "out of memory" in str(e):
                    free = torch.cuda.mem_get_info()[0] / 1024**3
                    raise RuntimeError(
                        f"OOM in {func.__name__} with {free:.2f}GB free. "
                        "Consider reducing batch size or image resolution."
                    ) from e
                raise

        return wrapper

    return decorator
