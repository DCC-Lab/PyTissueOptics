from matplotlib import pyplot as plt

SHOW_VISUAL_TESTS = False


def compareVisuals(expectedImagePath, currentImagePath, title: str) -> bool:
    visualIsOK = True

    def _visualOK(event, OK: bool = True):
        plt.close()
        nonlocal visualIsOK
        visualIsOK = OK

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(plt.imread(expectedImagePath))
    ax[1].imshow(plt.imread(currentImagePath))
    ax[0].set_title('Expected view')
    ax[1].set_title('Current view')
    axOK = plt.axes([0.7, 0.05, 0.1, 0.075])
    axFAIL = plt.axes([0.81, 0.05, 0.1, 0.075])
    btnOK = plt.Button(axOK, 'OK')
    btnFAIL = plt.Button(axFAIL, 'FAIL')
    btnOK.on_clicked(_visualOK)
    btnFAIL.on_clicked(lambda event: _visualOK(event, False))
    plt.suptitle(title)
    plt.show()
    return visualIsOK
