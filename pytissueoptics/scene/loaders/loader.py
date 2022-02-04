
class Loader:
    """
    Base class to manage the conversion between files and Scene() or Solid() from
    various types of files.
    """
    def __init__(self, file, parser: 'Parser'):
        self._file = file
        self.parser = parser
        self._parse()
        self._convert()

    def _parse(self):
        raise NotImplementedError

    def _convert(self):
        raise NotImplementedError
