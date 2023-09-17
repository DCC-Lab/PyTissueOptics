import unittest

from examples import *


class TestExamples(unittest.TestCase):
    def testExampleFormat(self):
        self.assertTrue(len(EXAMPLE_FILES) > 0)
        for file in EXAMPLE_FILES:
            name = re.match(EXAMPLE_FILE_PATTERN, file).group(1)
            module = importlib.import_module(f"examples.{EXAMPLE_MODULE}.{name}")
            with open(os.path.join(EXAMPLE_DIR, file), 'r') as f:
                srcCode = f.read()
            with self.subTest(name):
                self.assertTrue(hasattr(module, "TITLE"))
                self.assertTrue(hasattr(module, "DESCRIPTION"))
                self.assertTrue(hasattr(module, "exampleCode"))
                self.assertTrue(srcCode.startswith("from pytissueoptics import *"))
                self.assertTrue(srcCode.endswith("if __name__ == \"__main__\":\n" +
                                                 "    import env\n" + "    exampleCode()\n"))

    def testLoadExamples(self):
        allExamples = loadExamples()
        self.assertTrue(len(allExamples) > 0)
        for example in allExamples:
            with self.subTest(example.name):
                self.assertTrue(example.name.startswith("ex"))
                self.assertTrue(example.title)
                self.assertTrue(example.description)
                self.assertTrue(example.func)
                self.assertTrue(example.sourceCode)


if __name__ == '__main__':
    unittest.main()
