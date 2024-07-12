import numpy as np
import matplotlib.pyplot as plt

# from pytissueoptics.examples.reportHcmi import reportHmci

myname = "custom"

# Load header file
filename = f"{myname}_H.mci"
print(f"loading {filename}")
with open(filename, 'r') as fid:
    A = np.array(fid.read().split(), dtype=np.float32)

print(A)
# Extract parameters
time_min = A[0]
Nx, Ny, Nz = map(int, A[1:4])
dx, dy, dz = A[4:7]
mcflag, launchflag, boundaryflag = map(int, A[7:10])
xs, ys, zs = A[10:13]
xfocus, yfocus, zfocus = A[13:16]
ux0, uy0, uz0 = A[16:19]
radius, waist, Nt = A[19:22]
print(A[22:])
# Extract optical properties. They are grouped together as (mua, mus, g) for each tissue type
muav, musv, gv = [], [], []
for i in range(int(Nt)):
    muav.append(A[22 + 3 * i])
    musv.append(A[23 + 3 * i])
    gv.append(A[24 + 3 * i])


print("mua", muav)
print("mus", musv)
print("g", gv)
# Report input parameters
# reportHmci(myname)

# Load Fluence rate F(y,x,z)
filename = f"{myname}_F.bin"
print(f"loading {filename}")
with open(filename, 'rb') as fid:
    Data = np.fromfile(fid, dtype=np.float32)
F = np.reshape(Data, (Nz, Nx, Ny))

# Load tissue structure in voxels, T(y,x,z)
filename = f"{myname}_T.bin"
print(f"loading {filename}")
with open(filename, 'rb') as fid:
    Data = np.fromfile(fid, dtype=np.uint8)
T = np.reshape(Data, (Nz, Nx, Ny))

x = (np.arange(Nx) - Nx / 2 - 0.5) * dx
y = (np.arange(Ny) - Ny / 2 - 0.5) * dy
z = (np.arange(Nz) - 0.5) * dz
ux = np.arange(1, Nx - 1)
uy = np.arange(1, Ny - 1)
uz = np.arange(1, Nz - 1)
zmin, zmax = np.min(z), np.max(z)
zdiff = zmax - zmin
xmin, xmax = np.min(x), np.max(x)
xdiff = xmax - xmin

# Look at structure, Tzx
Tzx = np.transpose(T[Ny // 2, :, :])
plt.figure(1)
plt.clf()
plt.imshow(Tzx[ux, :], extent=(np.min(x[ux]), np.max(x[ux]), np.min(z[uz]), np.max(z[uz])), aspect='auto')
plt.colorbar()
plt.xlabel('x [cm]')
plt.ylabel('z [cm]')
plt.title('Tissue')
plt.show()

# Look at Fluence Fzx @ launch point
Fzx = np.transpose(F[Ny // 2, :, :])
plt.figure(2)
plt.clf()
plt.imshow(np.log10(Fzx), extent=(np.min(x), np.max(x), np.min(z), np.max(z)), aspect='auto')
plt.colorbar()
plt.xlabel('x [cm]')
plt.ylabel('z [cm]')
plt.title('Fluence')
plt.show()

# Look at Fluence Fzy
Fzy = np.transpose(F[:, Nx // 2, :])
plt.figure(3)
plt.clf()
plt.imshow(np.log10(Fzy), extent=(np.min(y), np.max(y), np.min(z), np.max(z)), aspect='auto')
plt.colorbar()
plt.xlabel('y [cm]')
plt.ylabel('z [cm]')
plt.title('Fluence')
plt.show()

# Look at Fluence Fxy
Fxy = np.transpose(F[:, :, Nz // 2])
plt.figure(4)
plt.clf()
plt.imshow(np.log10(Fxy), extent=(np.min(x), np.max(x), np.min(y), np.max(y)), aspect='auto')
plt.colorbar()
plt.xlabel('x [cm]')
plt.ylabel('y [cm]')
plt.title('Fluence')
plt.show()

# Look at Azx
print(muav)
A = F * muav[0]  # [W/cm^3 / W delivered]  aka Relative energy deposition
plt.figure(4)
plt.clf()
# plt.imshow(np.log10(Azx), extent=(np.min(x), np.max(x), np.min(z), np.max(z)), aspect='auto')
A_zsum = np.sum(A, axis=1)
plt.imshow(A_zsum, extent=(np.min(x), np.max(x), np.min(z), np.max(z)), aspect='auto')
plt.colorbar()
plt.xlabel('x [cm]')
plt.ylabel('z [cm]')
plt.title('Deposition A')
plt.show()

print('done')

# Integrate deposited energy accross whole volume
A_total = np.sum(A)  #
print(f"Total deposited energy = {A_total:.2f}")


# TODO: refact code and automate makeTissue, mcxyz exec and analysis.
#  - Simplify makeTissue to easily generate new tissue input files from optical properties.
#  - Move input files to mcxyz folder and run mcxyz from python.
#  - Transfer back output files.
#  - Analyze output files to extract total deposited energy.
#  - Compare with PyTissueOptics
