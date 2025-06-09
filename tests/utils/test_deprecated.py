import pytest
import warnings

# Import the functions and decorators from the module
from minerva.utils.deprecated import deprecated, deprecated_class, warn_if_used


# Test for deprecated function
def test_deprecated_function():
    @deprecated(msg="Please use 'new_function' instead.")
    def old_function():
        return "This is the old function."

    with pytest.warns(
        DeprecationWarning, match="Warning: 'old_function' is deprecated"
    ):
        old_function()

    @deprecated(msg="Please use 'new_function' instead", version="1.0")
    def old_function_with_version():
        return "This is the old function with version."

    with pytest.warns(DeprecationWarning, match="and will be removed in version 1.0"):
        old_function_with_version()


# Test for deprecated class
def test_deprecated_class():
    @deprecated_class(msg="Use NewClass() instead.")
    class OldClass:
        def __init__(self):
            self.value = 10

    with pytest.warns(
        DeprecationWarning, match="Warning: Class 'OldClass' is deprecated"
    ):
        obj = OldClass()

    @deprecated_class(msg="Use NewClass() instead", version="1.0")
    class OldClassWithVersion:
        def __init__(self):
            self.value = 20

    with pytest.warns(
        DeprecationWarning, match="Warning: Class 'OldClassWithVersion' is deprecated"
    ):
        obj_with_version = OldClassWithVersion()


# Test for warn_if_used with deprecated parameters
def test_warn_if_used():
    class MyClass:
        def __init__(self, new_param, old_param=None):
            warn_if_used(
                old_param,
                "old_param",
                msg="It will be removed in the next version.",
                version="1.0",
            )
            self.new_param = new_param
            self.old_param = old_param

    with pytest.warns(
        DeprecationWarning, match="Warning: parameter 'old_param' is deprecated"
    ):
        obj = MyClass(new_param="new", old_param="old")

    with pytest.warns(DeprecationWarning, match="and will be removed in version 1.0"):
        obj = MyClass(new_param="new", old_param="old")


# Test deprecated function with custom version
def test_deprecated_function_with_version():
    @deprecated(msg="Please use 'new_function' instead.", version="2.0")
    def old_function():
        return "This is the old function."

    with pytest.warns(DeprecationWarning, match="and will be removed in version 2.0"):
        old_function()


# Test deprecated class with custom version
def test_deprecated_class_with_version():
    @deprecated_class(msg="Use NewClass() instead.", version="3.0")
    class OldClass:
        def __init__(self):
            self.value = 10

    with pytest.warns(DeprecationWarning, match="and will be removed in version 3.0"):
        obj = OldClass()


# Test warn_if_used with custom version
def test_warn_if_used_with_version():
    class MyClass:
        def __init__(self, new_param, old_param=None):
            warn_if_used(
                old_param,
                "old_param",
                msg="It will be removed in the next version.",
                version="2.0",
            )
            self.new_param = new_param
            self.old_param = old_param

    with pytest.warns(DeprecationWarning, match="and will be removed in version 2.0"):
        obj = MyClass(new_param="new", old_param="old")
