import importlib

AVAILABLE_CONTEXT = []


if importlib.util.find_spec('torch'):
    AVAILABLE_CONTEXT += ['torch']

if importlib.util.find_spec('tensorflow'):
    AVAILABLE_CONTEXT += ['tf']


assert len(AVAILABLE_CONTEXT) > 0, (
    "Should have at least one deep-learning framework in "
    "{'tensorflow' | 'pytorch'}"
)
