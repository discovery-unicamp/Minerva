def tree_like_formating(d: dict, initial_indent: str = "") -> str:
    """Format a dictionary into a tree-like structure.

    Parameters
    ----------
    d : dict
        The dictionary to format.
    initial_indent : str, optional
        The initial indentation for the first level of the tree structure
        (default is no indentation), by default "". This is useful for
        controlling the starting point of the tree structure. For example,
        if you want to start the tree structure with a specific indentation
        level, you can pass a string of spaces or other characters to this
        parameter. This allows for more flexibility in formatting the tree
        structure according to your preferences.
    Returns
    -------
    str
        The formatted tree-like structure as a string.
    """

    lines = []

    def recursive_tree_structure(d, indent):
        keys = list(d.keys())
        for i, key in enumerate(keys):
            value = d[key]
            is_last_child = i == len(keys) - 1
            branch = "└── " if is_last_child else "├── "
            child_indent = "    " if is_last_child else "│   "

            if isinstance(value, dict):
                lines.append(f"{indent}{branch}{key}")
                recursive_tree_structure(value, indent + child_indent)
            else:
                lines.append(f"{indent}{branch}{key}: {value}")

    recursive_tree_structure(d, initial_indent)
    return "\n".join(lines)


def indent_text(text: str, spaces: int = 4, char: str = " ") -> str:
    """Add indentation to each line of a given text.

    Parameters
    ----------
    text : str
        The text to be indented.
    spaces : int, optional
        The number of spaces to indent each line, by default 4
    char : str, optional
        The character to use for indentation, by default " "

    Returns
    -------
    str
        The indented text.
    """
    indent = char * spaces
    return "\n".join(
        f"{indent}{line}" if line.strip() else line for line in text.splitlines()
    )
