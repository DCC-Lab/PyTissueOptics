import sys
import os
from pathlib import Path

directory = Path(os.path.dirname(os.path.abspath(__file__)))
module_dir = directory.parts[0:directory.parts.index('pytissueoptics')]

sys.path.insert(
    0, str(Path(*module_dir))
)
