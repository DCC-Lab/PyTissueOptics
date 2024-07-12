def reportHmci(myname):
    # Load header file
    filename = f"{myname}_H.mci"
    print(f"Loading {filename}")
    with open(filename, 'r') as fid:
        B = [float(val) for val in fid.read().split()]

    # Define parameter names
    param_names = [
        'time_min', 'Nx', 'Ny', 'Nz', 'dx', 'dy', 'dz', 'mcflag', 'launch',
        'boundary', 'xs', 'ys', 'zs', 'xfocus', 'yfocus', 'zfocus', 'ux0',
        'uy0', 'uz0', 'radius', 'waist', 'Nt'
    ]

    # Display parameter values
    for i, name in enumerate(param_names, start=1):
        print(f"{i}\t{name} = {B[i-1]:.4f}")

    # Display optical parameters
    j = 22
    for _ in range(int(B[21])):
        print('---')
        j += 1
        print(f"{j}\tmua = {B[j]:.4f}")
        j += 1
        print(f"{j}\tmus = {B[j]:.4f}")
        j += 1
        print(f"{j}\tg   = {B[j]:.4f}")
