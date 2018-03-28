AVAILABLE_CONTEXT = []
try:
    import torch  # noqa: F401
    AVAILABLE_CONTEXT += ['torch']
except ImportError:
    pass
try:
    import tensorflow as tf  # noqa: F401
    AVAILABLE_CONTEXT += ['tf']
except ImportError:
    pass
assert len(AVAILABLE_CONTEXT) > 0, (
    "Should have at least one deep-learning framework in "
    "{'tensorflow' | 'pytorch'}"
)
