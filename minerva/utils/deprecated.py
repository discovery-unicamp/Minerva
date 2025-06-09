import functools
import warnings


def deprecated(msg: str, version=None):
    """Decorator to mark a function as deprecated. It will print a warning
    message when the function is called, indicating that it is deprecated and
    will be removed in a future version.

    Parameters
    ----------
    msg : str
        A message to display when the function is called.
    version : str, optional
        The version in which the function will be removed. If provided, it will
        be included in the warning message.

    Example
    -------
    >>> @deprecated(msg="Use new_function() instead.", version="2.0")
    ... def old_function():
    ...     return "This is the old function."
    >>> old_function()
    """

    def decorator(func):
        message = f"Warning: '{func.__name__}' is deprecated"
        if version:
            message += f" and will be removed in version {version}"
        if msg:
            message += f". {msg}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_class(msg: str, version=None):
    """Decorator to mark a class as deprecated. It will print a warning
    message when an instance of the class is created, indicating that it is
    deprecated and will be removed in a future version.

    Parameters
    ----------
    msg : str
        A message to display when the class is instantiated.
    version : str, optional
        The version in which the class will be removed. If provided, it will
        be included in the warning message.

    Example
    -------
    >>> @deprecated_class(msg="Use NewClass() instead.", version="3.0")
    ... class OldClass:
    ...     def __init__(self):
    ...         self.value = 10
    >>> obj = OldClass()
    """

    def decorator(cls):
        message = f"Warning: Class '{cls.__name__}' is deprecated"
        if version:
            message += f" and will be removed in version {version}"
        if msg:
            message += f". {msg}"

        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            orig_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


def warn_if_used(
    param,
    param_name: str,
    msg: str,
    version=None,
):
    """Warn if a parameter is used, indicating that it is deprecated and will
    be removed in a future version.

    Parameters
    ----------
    param : any
        The parameter to check. If it is not None, a warning will be issued.
    param_name : str
        The name of the parameter to include in the warning message.
    msg : str
        A message to display when the parameter is used.
    version : str, optional
        The version in which the parameter will be removed. If provided, it will
        be included in the warning message.

    Example
    -------
    >>> class MyClass:
    ...     def __init__(self, new_param, old_param=None):
    ...         warn_if_used(old_param, "old_param", msg="It will be removed in the next version.", version="1.0")
    ...         self.new_param = new_param
    ...         self.old_param = old_param
    >>> obj = MyClass(new_param="new", old_param="old")

    """
    if param is not None:
        msg = f"Warning: parameter '{param_name}' is deprecated"
        if version:
            msg += f" and will be removed in version {version}"
        if msg:
            msg += f". {msg}"
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
