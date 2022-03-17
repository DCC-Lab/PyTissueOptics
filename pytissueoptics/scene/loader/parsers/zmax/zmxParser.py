from pytissueoptics.scene.loader.parsers import Parser


class ZMXParser(Parser):
    def _checkFileExtension(self):
        if self._filepath.endswith('.zmx'):
            return
        else:
            raise TypeError

    def _parse(self):
        raise NotImplementedError
