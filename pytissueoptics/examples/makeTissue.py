import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics.examples.makeTissueList import makeTissueList

"""
Wrapper translated from matlab files.
"""


def makecmap(Nt):
    cmap = np.zeros((64, 3))
    f1 = 0.7  # color adjustments
    f2 = 0.5  # color adjustments
    f3 = 0.3  # color adjustments
    dj = 0.05
    for i in range(64):
        j = round((i - dj) / 64 * (Nt - 1))
        if j <= 1 - dj:
            cmap[i, :] = [0, 0, 0]
        elif j <= 2 - dj:
            cmap[i, :] = [1, 1, 1]
        elif j <= 3 - dj:
            cmap[i, :] = [1, 0, 0]
        elif j <= 4 - dj:
            cmap[i, :] = [1, 0.8, 0.8]
        elif j <= 5 - dj:
            cmap[i, :] = [0.5, 0.2, 0.2]
        elif j <= 6 - dj:
            cmap[i, :] = [f1, 1, f1]  # skull
        elif j <= 7 - dj:
            cmap[i, :] = [f2, f2, f2]  # gray matter
        elif j <= 8 - dj:
            cmap[i, :] = [0.5, 1, 1]  # white matter
        elif j <= 9 - dj:
            cmap[i, :] = [1, 0.5, 0.5]  # standard tissue
    return cmap


def maketissue():
    # USER CHOICES
    SAVEON = 1  # 1 = save myname_T.bin, myname_H.mci, 0 = don't save
    myname = 'custom'
    time_min = 0.1
    nm = 532
    Nbins = 200
    binsize = 60 / Nbins

    mcflag = 0
    launchflag = 0
    boundaryflag = 2
    xs = 0  # source position
    ys = 0
    zs = 0.0101
    xfocus = 0
    yfocus = 0
    zfocus = np.inf
    radius = 0.0100   # Source stuff ?
    waist = 0.0100
    ux0 = 0.99  # ignored with launchflag = 0
    uy0 = 0.5
    uz0 = np.sqrt(1 - ux0**2 - uy0**2)

    # PREPARE MONTE CARLO
    # tissue = makeTissueList(nm)
    tissue = [
        {"name": "dermisa", "mua": 0.05, "mus": 1, "g": 0.01},
        {"name": "dermis", "mua": 0.05, "mus": 1, "g": 0.01},
    ]
    Nt = len(tissue)
    muav = np.zeros(Nt)
    musv = np.zeros(Nt)
    gv = np.zeros(Nt)
    for i in range(Nt):
        mua = tissue[i]['mua']
        if not isinstance(mua, (int, float)):
            print(np.unique(mua))
            mua = mua[0]
        muav[i] = mua
        # print(f"tissue[{i}]['mua'] = {tissue[i]['mua']}")
        # print(f"tissue[{i}]['mus'] = {tissue[i]['mus']}")
        # print(f"tissue[{i}]['g'] = {tissue[i]['g']}")
        # muav[i] = tissue[i]['mua']
        musv[i] = tissue[i]['mus']
        gv[i] = tissue[i]['g']

    # CREATE TISSUE STRUCTURE T(y,x,z)
    Nx = Ny = Nz = Nbins
    dx = dy = dz = binsize
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    z = np.arange(Nz) * dz
    zmin, zmax = np.min(z), np.max(z)
    xmin, xmax = np.min(x), np.max(x)

    if np.isinf(zfocus):
        zfocus = 1e12

    T = np.zeros((Ny, Nx, Nz))

    T += 1  # fill background with skin (dermis)

    # zsurf = 0.0100  # position of air/skin surface

    # for iz in range(Nz):
    #     if iz <= round(zsurf / dz):
    #         T[:, :, iz] = 2  # air  # or water??
    #     if round(zsurf / dz) < iz <= round((zsurf + 0.0060) / dz):
    #         T[:, :, iz] = 5  # epidermis (60 um thick)
    #     xc = 0  # [cm], center of blood vessel
    #     zc = Nz / 2 * dz  # [cm], center of blood vessel
    #     vesselradius = 0.0100  # blood vessel radius [cm]
    #     for ix in range(Nx):
    #         xd = x[ix] - xc  # vessel, x distance from vessel center
    #         zd = z[iz] - zc  # vessel, z distance from vessel center
    #         r = np.sqrt(xd**2 + zd**2)  # r from vessel center
    #         if r <= vesselradius:  # if r is within vessel
    #             T[:, ix, iz] = 3  # blood

    if SAVEON:
        v = np.uint8(T.reshape(Ny * Nx * Nz))
        filename = f"{myname}_H.mci"
        print(f"Creating {filename}")
        with open(filename, 'w') as fid:
            fid.write(f"{time_min:.2f}\n")
            fid.write(f"{Nx}\n{Ny}\n{Nz}\n")
            fid.write(f"{dx:.4f}\n{dy:.4f}\n{dz:.4f}\n")
            fid.write(f"{mcflag}\n{launchflag}\n{boundaryflag}\n")
            fid.write(f"{xs:.4f}\n{ys:.4f}\n{zs:.4f}\n")
            fid.write(f"{xfocus:.4f}\n{yfocus:.4f}\n{zfocus:.4f}\n")
            fid.write(f"{ux0:.4f}\n{uy0:.4f}\n{uz0:.4f}\n")
            fid.write(f"{radius:.4f}\n{waist:.4f}\n")
            fid.write(f"{Nt}\n")
            for i in range(Nt):
                fid.write(f"{muav[i]:.4f}\n{musv[i]:.4f}\n{gv[i]:.4f}\n")

        filename = f"{myname}_T.bin"
        print(f"Creating {filename}")
        with open(filename, 'wb') as fid:
            fid.write(v)

    # LOOK AT STRUCTURE OF Tzx AT iy = Ny / 2
    Txzy = np.transpose(T, (1, 0, 2))  # Tyxz --> Txzy
    Tzx = Txzy[:, :, Ny // 2]  # Tzx

    plt.figure()
    plt.imshow(Tzx, extent=[xmin, xmax, zmin, zmax], vmin=1, vmax=Nt)
    plt.colorbar()
    cmap = makecmap(Nt)
    # plt.set_cmap(cmap)
    plt.gca().set_xlabel('x [cm]')
    plt.gca().set_ylabel('z [cm]')
    plt.axis('equal')
    plt.title('Tissue Types')
    plt.show()

    print('Done')


if __name__ == '__main__':
    maketissue()
