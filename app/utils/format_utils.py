"""
JSON formatting utilities.

Provides consistent JSON serialization and formatting across the project.
"""

import json
from typing import Any, Dict


def format_state(state: Dict[str, Any], indent: int = 4) -> str:
    """
    Format a workflow state object such as `ImportGraphState`.

    Args:
        state: Workflow state dictionary
        indent: Number of spaces used for JSON indentation, default 4

    Returns:
        Formatted JSON string

    Example:
        >>> state = {"task_id": "001", "pdf_path": "test.pdf"}
        >>> print(format_state(state))
        {
            "task_id": "001",
            "pdf_path": "test.pdf"
        }
    """

    return json.dumps(state, indent=indent, ensure_ascii=False)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """
    Generic JSON formatting helper.

    Args:
        data: Serializable data to format
        indent: Number of spaces used for JSON indentation, default 4
        ensure_ascii: Whether to escape non-ASCII characters, default False

    Returns:
        Formatted JSON string

    Example:
        >>> data = {"name": "test", "value": 123}
        >>> print(format_json(data))
        {
            "name": "test",
            "value": 123
        }
    """
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

