"""
Shared model configuration for sub-agents.

The main agent module calls set_model() before importing the sub-agents,
so every agent in the pipeline uses the same LLM provider/model.
"""

_model = None


def set_model(model):
    global _model
    _model = model


def get_model():
    if _model is None:
        raise RuntimeError(
            "agent_config.set_model() must be called before importing sub-agents"
        )
    return _model
