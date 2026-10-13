"""
tests/conftest.py
=================
Global pytest configuration and compatibility shims.
"""

import sys
import dataclasses

# Python < 3.10 compatibility shim for dataclass(slots=True)
if sys.version_info < (3, 10):
    _orig_dataclass = dataclasses.dataclass

    def _compat_dataclass(*args, **kwargs):
        kwargs.pop("slots", None)
        return _orig_dataclass(*args, **kwargs)

    dataclasses.dataclass = _compat_dataclass
