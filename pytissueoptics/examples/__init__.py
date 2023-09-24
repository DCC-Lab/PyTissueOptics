import os
import re
import importlib
from dataclasses import dataclass
from typing import List

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

EXAMPLE_MODULE = "rayscattering"
EXAMPLE_FILE_PATTERN = r"^(ex\d+)\.py$"
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), EXAMPLE_MODULE)
EXAMPLE_FILES = [file for file in os.listdir(EXAMPLE_DIR) if re.match(EXAMPLE_FILE_PATTERN, file)]
EXAMPLE_FILES.sort()


@dataclass
class Example:
    name: str
    title: str
    description: str
    func: callable
    sourceCode: str


def loadExamples() -> List[Example]:
    allExamples = []
    for file in EXAMPLE_FILES:
        name = re.match(EXAMPLE_FILE_PATTERN, file).group(1)
        module = importlib.import_module(f"pytissueoptics.examples.{EXAMPLE_MODULE}.{name}")
        with open(os.path.join(EXAMPLE_DIR, file), 'r') as f:
            srcCode = f.read()
        pattern = r"def exampleCode\(\):\s*(.*?)\s*if __name__ == \"__main__\":"
        srcCode = re.search(pattern, srcCode, re.DOTALL).group(1)
        srcCode = re.sub(r"^    ", "", srcCode, flags=re.MULTILINE)
        srcCode = "from pytissueoptics import *\n" + srcCode
        srcCode = highlight(srcCode, PythonLexer(), TerminalFormatter())
        allExamples.append(Example(name, module.TITLE, module.DESCRIPTION, module.exampleCode, srcCode))
    return allExamples
