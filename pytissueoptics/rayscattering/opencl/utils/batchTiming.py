class BatchTiming:
    """
    Used to record and display the progress of a batched photon propagation.
    """
    def __init__(self, totalPhotons: int):
        self._photonCount = 0
        self._totalPhotons = totalPhotons
        self._batchCount = 0

        self._propagationTime = 0
        self._dataTransferTime = 0
        self._dataConversionTime = 0
        self._totalTime = 0

        self._title = "SIMULATION PROGRESS"
        self._header = ["BATCH #", "PHOTON COUNT", "SPEED (ph/ms)", "TIME ELAPSED", "TIME LEFT"]

        self._columnWidths = [len(value) + 3 for value in self._header]
        photonCountWidth = len(str(self._totalPhotons)) + 11
        self._columnWidths[1] = max(self._columnWidths[1], photonCountWidth)
        self._width = sum(self._columnWidths) + 3 * (len(self._columnWidths) - 1)

        self._printHeader()

    def recordBatch(self, photonCount: int, propagationTime: float, dataTransferTime: float, dataConversionTime: float, totalTime: float):
        """
        Photon count is the number of photons that were propagated in the batch. The other times are in nanoseconds.
        Propagation time is the time it took to run the propagation kernel. Data transfer time is the time it took to
        transfer the raw 3D data from the GPU. Data conversion time is the time it took to sort and convert the 
        interactions IDs into proper InteractionKey points. 
        """
        self._photonCount += photonCount
        self._propagationTime += propagationTime
        self._dataTransferTime += dataTransferTime
        self._dataConversionTime += dataConversionTime
        self._totalTime += totalTime
        self._batchCount += 1

        self._printProgress()
        if self._photonCount == self._totalPhotons:
            self._printFooter()

    def _printHeader(self):
        halfBanner = "=" * ((self._width - len(self._title) - 1) // 2)
        print(f"\n{halfBanner} {self._title} {halfBanner}")
        formatted_header = [f"[{title}] ".center(width) for title, width in zip(self._header, self._columnWidths)]
        print(":: ".join(formatted_header))

    def _printProgress(self):
        progress = self._photonCount / self._totalPhotons
        speed = self._photonCount / (self._totalTime / 1e6)
        timeElapsed = self._totalTime / 1e9
        timeLeft = timeElapsed * (self._totalPhotons / self._photonCount) - timeElapsed

        progressString = f"{self._photonCount} ({progress * 100:.1f}%)"
        speedString = f"{speed:.2f}"
        timeElapsedString = f"{timeElapsed:.2f} s"
        timeLeftString = f"{timeLeft:.2f} s"

        formatted_values = [f" {value} ".center(width) for value, width in
                            zip([self._batchCount, progressString, speedString, timeElapsedString, timeLeftString], self._columnWidths)]
        print(":: ".join(formatted_values))

    def _printFooter(self):
        print("\nComputation splits:")
        splits = {"Propagation": self._propagationTime, "Data transfer": self._dataTransferTime,
                  "Data conversion": self._dataConversionTime}
        for key, value in splits.items():
            print(f"\t{key}: {value / self._totalTime * 100:.1f}%")
        print("".join(["=" * self._width]) + "\n")
