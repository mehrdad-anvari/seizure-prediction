from .binary import register_binary_evaluators  # noqa: F401
from .mil import register_mil_evaluators  # noqa: F401

register_binary_evaluators()
register_mil_evaluators()
