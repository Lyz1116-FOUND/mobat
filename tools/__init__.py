"""
Thin compatibility package for local project wrappers.

This package keeps `from tools.xxx import ...` working by extending the
package search path to the framework-provided `tools` package available in
the container image.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
