import time
from functools import wraps
from typing import Callable, Any

def measure_exec_time(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.execution_time = time.perf_counter() - start
        return result
    
    wrapper.execution_time = 0.0
    return wrapper