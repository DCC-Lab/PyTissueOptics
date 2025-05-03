from importlib.metadata import version

from .rayscattering import *  # noqa: F403
from .scene import *  # noqa: F403

__version__ = version("pytissueoptics")
