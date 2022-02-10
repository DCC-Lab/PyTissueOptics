from pytissueoptics.scene.loader.parsers import Parser


class ColladaParser(Parser):

    def _checkFileExtension(self):
        if self._filepath.endswith('.dae'):
            return
        else:
            raise TypeError

    def _parse(self):
        raise NotImplementedError
