from pytissueoptics.scene.loaders.parsers import Parser


class StepParser(Parser):

    def _checkFileExtension(self):
        if self._filepath.endswith('.step'):
            return
        else:
            raise TypeError

    def _parse(self):
        raise NotImplementedError