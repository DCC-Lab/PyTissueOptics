import json
import os
from typing import Optional


class IPPTable:
    TABLE_PATH = os.path.join(os.path.dirname(__file__), "ipp.json")

    def __init__(self):
        with open(self.TABLE_PATH, "r") as f:
            self._table = json.load(f)

    def getIPP(self, experimentHash: int) -> Optional[float]:
        if str(experimentHash) not in self._table:
            return None
        return self._table[str(experimentHash)][1]

    def updateIPP(self, experimentHash: int, photonCount: int, IPP: float):
        if str(experimentHash) not in self._table:
            self._table[str(experimentHash)] = [photonCount, IPP]
        else:
            oldN, oldIPP = self._table[str(experimentHash)]
            newN = oldN + photonCount
            newIPP = (oldN * oldIPP + photonCount * IPP) / newN
            self._table[str(experimentHash)] = [newN, round(newIPP, 3)]
        self._save()

    def _save(self):
        with open(self.TABLE_PATH, "w") as f:
            json.dump(self._table, f, indent=4)

    def __contains__(self, experimentHash: int):
        return str(experimentHash) in self._table
