# ===================== Core helper function =====================
def escape_milvus_string(value: str) -> str:
    """
    Safely escape a string for use in a Milvus filter expression.
    Purpose:
        Prevent filter parsing errors when the original string contains special
        characters so CRUD operations can run normally.
    Escaping rules:
        1. Backslash (`\`) -> double backslash (`\\`)
        2. Double quote (`"`) -> escaped double quote (`\"`)
        3. Newlines, carriage returns, and tabs -> spaces
    Args:
        value: Raw string to escape, such as an item name or file title
    Returns:
        str: Escaped string safe to use in `filter_expr`
    """
    if value is None:
        return ""
    # Ensure the input is treated as a string.
    s = str(value)
    # Escape special characters according to Milvus rules.
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    # Replace control whitespace with spaces to keep the expression on one line.
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
