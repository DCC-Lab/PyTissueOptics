import json
import os
from typing import Optional


class IPPTable:
    TABLE_PATH = os.path.join(os.path.dirname(__file__), "../ipp.json")

    def __init__(self):
        self._assertExists()

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

    def _assertExists(self):
        if not os.path.exists(self.TABLE_PATH):
            self._table = DEFAULT_IPP
            self._save()


DEFAULT_IPP = {
    "-6887487125664597075": [
        40000,
        144.571
    ],
    "2325526903167451785": [
        11601000,
        17.43
    ],
    "-6919696058704085231": [
        3901000,
        17.915
    ],
    "-2058547611842612924": [
        11000,
        2843.68
    ],
    "3321215562015198964": [
        223000,
        17.973
    ],
    "-7640516670101814241": [
        21000,
        2846.064
    ],
    "6019286260145673908": [
        246000,
        23.237
    ],
    "7995554833896372811": [
        256000,
        86.142
    ],
    "-181984282493345035": [
        11812662,
        41.95
    ]
}
