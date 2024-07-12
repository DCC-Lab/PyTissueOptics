import numpy as np
import scipy.io as sio

spectral_lib = sio.loadmat("spectralLIB.mat")

def makeTissueList(nm):
    # Load spectral library
    muaoxy = spectral_lib["muaoxy"].flatten()
    muadeoxy = spectral_lib["muadeoxy"].flatten()
    muawater = spectral_lib["muawater"].flatten()
    muamel = spectral_lib["muamel"].flatten()
    nmLIB = spectral_lib["nmLIB"].flatten()

    MU = np.zeros((len(nmLIB), 4))
    MU[:, 0] = np.interp(nm, nmLIB, muaoxy)
    MU[:, 1] = np.interp(nm, nmLIB, muadeoxy)
    MU[:, 2] = np.interp(nm, nmLIB, muawater)
    MU[:, 3] = np.interp(nm, nmLIB, muamel)

    # Create tissue list
    tissue = []

    # air 2
    tissue.append({"name": "air", "mua": 0.0001, "mus": 1.0, "g": 1.0})

    # water
    tissue.append({"name": "water", "mua": MU[2], "mus": 10, "g": 1.0})

    # blood 3
    B = 1.00
    S = 0.75
    W = 0.95
    M = 0
    musp500 = 10
    fray = 0.0
    bmie = 1.0
    gg = 0.90
    musp = musp500 * (fray * (nm / 500) ** -4 + (1 - fray) * (nm / 500) ** -bmie)
    X = [B * S, B * (1 - S), W, M]
    tissue.append({"name": "blood", "mua": np.dot(MU, X), "mus": musp / (1 - gg), "g": gg})

    # dermis (4)
    B = 0.002
    S = 0.67
    W = 0.65
    M = 0
    musp500 = 42.4
    fray = 0.62
    bmie = 1.0
    gg = 0.90
    musp = musp500 * (fray * (nm / 500) ** -4 + (1 - fray) * (nm / 500) ** -bmie)
    X = [B * S, B * (1 - S), W, M]
    # tissue.append({"name": "dermis", "mua": np.dot(MU, X), "mus": musp / (1 - gg), "g": gg})
    tissue.append({"name": "dermis", "mua": 0.05, "mus": 1, "g": 0.01})

    # epidermis 5
    B = 0
    S = 0.75
    W = 0.75
    M = 0.03
    musp500 = 40
    fray = 0.0
    bmie = 1.0
    gg = 0.90
    musp = musp500 * (fray * (nm / 500) ** -4 + (1 - fray) * (nm / 500) ** -bmie)
    X = [B * S, B * (1 - S), W, M]
    tissue.append({"name": "epidermis", "mua": np.dot(MU, X), "mus": musp / (1 - gg), "g": gg})

    # skull
    B = 0.0005
    S = 0.75
    W = 0.35
    M = 0
    musp500 = 30
    fray = 0.0
    bmie = 1.0
    gg = 0.90
    musp = musp500 * (fray * (nm / 500) ** -4 + (1 - fray) * (nm / 500) ** -bmie)
    X = [B * S, B * (1 - S), W, M]
    tissue.append({"name": "skull", "mua": np.dot(MU, X), "mus": musp / (1 - gg), "g": gg})

    # gray matter
    B = 0.01
    S = 0.75
    W = 0.75
    M = 0
    musp500 = 20
    fray = 0.2
    bmie = 1.0
    gg = 0.90
    musp = musp500 * (fray * (nm / 500) ** -4 + (1 - fray) * (nm / 500) ** -bmie)
    X = [B * S, B * (1 - S), W, M]
    tissue.append({"name": "gray matter", "mua": np.dot(MU, X), "mus": musp / (1 - gg), "g": gg})

    # white matter
    B = 0.01
    S = 0.75
    W = 0.75
    M = 0
    musp500 = 20
    fray = 0.2
    bmie = 1.0
    gg = 0.90
    musp = musp500 * (fray * (nm / 500) ** -4 + (1 - fray) * (nm / 500) ** -bmie)
    X = [B * S, B * (1 - S), W, M]
    tissue.append({"name": "white matter", "mua": np.dot(MU, X), "mus": musp / (1 - gg), "g": gg})

    # standard tissue
    tissue.append({"name": "standard tissue", "mua": 1, "mus": 100, "g": 0.90})

    print("---- tissueList ------ \tmua   \tmus  \tg  \tmusp")
    for i, t in enumerate(tissue):
        print(f"{i+1}\t{t['name']}\tmua_length={np.array(t['mua']).shape}\t{t['mus']}\t{t['g']}\t{t['mus'] * (1 - t['g'])}")

    return tissue
