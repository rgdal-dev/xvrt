"""Single exception hierarchy so the rules R1..R7 raise something catchable."""
from __future__ import annotations


class VrtWriterError(ValueError):
    """Raised when the writer's rules are violated.

    Subclasses ``ValueError`` so that naive except-clauses still catch it;
    specific enough that callers can ``except VrtWriterError`` to distinguish
    rule violations from generic bad input.
    """
