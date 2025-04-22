import numpy as np


def MCG1Dx(b, dt, Theta, u):
    """
    Python translation of the MATLAB function MCG1Dx.
    Inputs:
        b: 2D numpy array of shape (N+2, N+2)
        dt: time step scalar
        Theta: scalar
        u: 2D numpy array of shape (N+2, N+2)
    Output:
        U: 2D numpy array of shape (N+2, N+2)
    """
    ITE = 2
    N = 128
    # allocate arrays with one-based indexing (ignore index 0)
    U = np.zeros((N+2, N+2))
    Uht = np.zeros((N+2, N+2))

    # coarse multigrid arrays
    U2h = np.zeros(N//2 + 2)
    U4h = np.zeros(N//4 + 2)
    U8h = np.zeros(N//8 + 2)
    U16h = np.zeros(N//16 + 2)
    U32h = np.zeros(N//32 + 2)
    U2ht = np.zeros(N//2 + 2)
    U4ht = np.zeros(N//4 + 2)
    U8ht = np.zeros(N//8 + 2)
    U16ht = np.zeros(N//16 + 2)

    RH = np.zeros(N+2)
    R2H = np.zeros(N//2 + 2)
    R4H = np.zeros(N//4 + 2)
    R8H = np.zeros(N//8 + 2)
    R16H = np.zeros(N//16 + 2)
    r32h = np.zeros(N//32 + 2)

    F2H = np.zeros(N//2 + 2)
    F4H = np.zeros(N//4 + 2)
    F8H = np.zeros(N//8 + 2)
    F16H = np.zeros(N//16 + 2)

    NN = N//32 - 1
    A32h = np.zeros((NN, NN))

    # grid spacings and coefficients
    p1 = dt * Theta / (1 ** 2)
    p2 = dt * Theta / (2 ** 2)
    p4 = dt * Theta / (4 ** 2)
    p8 = dt * Theta / (8 ** 2)
    p16 = dt * Theta / (16 ** 2)
    p32 = dt * Theta / (32 ** 2)

    # setup 3x3 A32h for coarse solve
    A32h[0, 0:3] = [1 + 2 * p32, -p32, 0]
    A32h[1, 0:3] = [-p32, 1 + 2 * p32, -p32]
    A32h[2, 0:3] = [0, -p32, 1 + 2 * p32]

    # main loop over interior columns
    for j in range(2, N+1):
        # two-cycle multigrid (fixed two iterations)
        for _cycle in range(2):
            # h-level relaxation
            for _ in range(ITE):
                VV = U.copy()
                for i in range(2, N+1):
                    U[i, j] = (
                        u[i, j]
                        + p1 * (u[i, j+1] - 2 * u[i, j] + u[i, j-1])
                        - dt * b[i, j]
                        + p1 * (VV[i+1, j] + VV[i-1, j])
                    ) / (1 + 2 * p1)

            # compute residual RH
            for i in range(2, N+1):
                RH[i] = (
                    u[i, j]
                    + p1 * (u[i, j+1] - 2 * u[i, j] + u[i, j-1])
                    - dt * b[i, j]
                    + p1 * (U[i+1, j] + U[i-1, j])
                    - (1 + 2 * p1) * U[i, j]
                )

            # restrict RH -> F2H
            for i in range(2, N//2 + 1):
                F2H[i] = 0.25 * (RH[2*i-1] + 2 * RH[2*i] + RH[2*i+1])

            # 2h relaxation
            VV2h = np.zeros_like(U2h)
            for _ in range(ITE):
                for i in range(2, N//2 + 1):
                    U2h[i] = (F2H[i] + p2 * (VV2h[i+1] + VV2h[i-1])) / (1 + 2 * p2)
                VV2h[:] = U2h[:]

            # residual R2H
            for i in range(2, N//2 + 1):
                R2H[i] = F2H[i] + p2 * (U2h[i+1] + U2h[i-1]) - (1 + 2 * p2) * U2h[i]

            # restrict R2H -> F4H
            for i in range(2, N//4 + 1):
                F4H[i] = 0.25 * (R2H[2*i-1] + 2 * R2H[2*i] + R2H[2*i+1])

            # 4h relaxation
            VV4h = np.zeros_like(U4h)
            for _ in range(ITE):
                for i in range(2, N//4 + 1):
                    U4h[i] = (F4H[i] + p4 * (VV4h[i+1] + VV4h[i-1])) / (1 + 2 * p4)
                VV4h[:] = U4h[:]

            # residual R4H
            for i in range(2, N//4 + 1):
                R4H[i] = F4H[i] + p4 * (U4h[i+1] + U4h[i-1]) - (1 + 2 * p4) * U4h[i]

            # restrict R4H -> F8H
            for i in range(2, N//8 + 1):
                F8H[i] = 0.25 * (R4H[2*i-1] + 2 * R4H[2*i] + R4H[2*i+1])

            # 8h relaxation
            VV8h = np.zeros_like(U8h)
            for _ in range(ITE):
                for i in range(2, N//8 + 1):
                    U8h[i] = (F8H[i] + p8 * (VV8h[i+1] + VV8h[i-1])) / (1 + 2 * p8)
                VV8h[:] = U8h[:]

            # residual R8H
            for i in range(2, N//8 + 1):
                R8H[i] = F8H[i] + p8 * (U8h[i+1] + U8h[i-1]) - (1 + 2 * p8) * U8h[i]

            # restrict R8H -> F16H
            for i in range(2, N//16 + 1):
                F16H[i] = 0.25 * (R8H[2*i-1] + 2 * R8H[2*i] + R8H[2*i+1])

            # 16h relaxation
            VV16h = np.zeros_like(U16h)
            for _ in range(ITE):
                for i in range(2, N//16 + 1):
                    U16h[i] = (F16H[i] + p16 * (VV16h[i+1] + VV16h[i-1])) / (1 + 2 * p16)
                VV16h[:] = U16h[:]

            # residual R16H
            for i in range(2, N//16 + 1):
                R16H[i] = F16H[i] + p16 * (U16h[i+1] + U16h[i-1]) - (1 + 2 * p16) * U16h[i]

            # restrict R16H -> r32h
            for i in range(2, N//32 + 1):
                r32h[i-1] = 0.25 * (R16H[2*i-1] + 2 * R16H[2*i] + R16H[2*i+1])

            # solve on 32h
            temp = np.linalg.solve(A32h, r32h[:NN])
            for i in range(2, N//32 + 1):
                U32h[i] = temp[i-2]

            # prolongation back to 16h
            for i in range(1, N//32 + 1):
                U16ht[2*i-1] = U32h[i]
                U16ht[2*i]   = 0.5 * (U32h[i] + U32h[i+1])
            for i in range(2, N//16 + 1):
                U16h[i] += U16ht[i]

            # 16h relax
            for _ in range(ITE):
                VV16h[:] = U16h[:]
                for i in range(2, N//16 + 1):
                    U16h[i] = (F16H[i] + p16 * (VV16h[i+1] + VV16h[i-1])) / (1 + 2 * p16)

            # prolongation back to 8h
            for i in range(1, N//16 + 1):
                U8ht[2*i-1] = U16h[i]
                U8ht[2*i]   = 0.5 * (U16h[i] + U16h[i+1])
            for i in range(2, N//8 + 1):
                U8h[i] += U8ht[i]

            # 8h relax
            for _ in range(ITE):
                VV8h[:] = U8h[:]
                for i in range(2, N//8 + 1):
                    U8h[i] = (F8H[i] + p8 * (VV8h[i+1] + VV8h[i-1])) / (1 + 2 * p8)

            # prolongation back to 4h
            for i in range(1, N//8 + 1):
                U4ht[2*i-1] = U8h[i]
                U4ht[2*i]   = 0.5 * (U8h[i] + U8h[i+1])
            for i in range(2, N//4 + 1):
                U4h[i] += U4ht[i]

            # 4h relax
            for _ in range(ITE):
                VV4h[:] = U4h[:]
                for i in range(2, N//4 + 1):
                    U4h[i] = (F4H[i] + p4 * (VV4h[i+1] + VV4h[i-1])) / (1 + 2 * p4)

            # prolongation back to 2h
            for i in range(1, N//4 + 1):
                U2ht[2*i-1] = U4h[i]
                U2ht[2*i]   = 0.5 * (U4h[i] + U4h[i+1])
            for i in range(2, N//2 + 1):
                U2h[i] += U2ht[i]

            # 2h relax
            for _ in range(ITE):
                VV2h[:] = U2h[:]
                for i in range(2, N//2 + 1):
                    U2h[i] = (F2H[i] + p2 * (VV2h[i+1] + VV2h[i-1])) / (1 + 2 * p2)

            # prolongation back to h
            for i in range(1, N//2 + 1):
                Uht[2*i-1, j] = U2h[i]
                Uht[2*i, j]   = 0.5 * (U2h[i] + U2h[i+1])

            # final h-level update and relax
            for i in range(2, N+1):
                U[i, j] += Uht[i, j]
            for _ in range(ITE):
                VV = U.copy()
                for i in range(2, N+1):
                    U[i, j] = (
                        u[i, j]
                        + p1 * (u[i, j+1] - 2 * u[i, j] + u[i, j-1])
                        - dt * b[i, j]
                        + p1 * (VV[i+1, j] + VV[i-1, j])
                    ) / (1 + 2 * p1)
    return U
