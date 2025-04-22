import numpy as np


def MCG1Dy(b, dt, Theta, u):
    """
    Python translation of MCG1Dy.m
    Inputs:
      b     : 2D array of shape (N+1, N+1)
      dt    : time step (scalar)
      Theta : scalar parameter
      u     : 2D array of shape (N+1, N+1)
    Output:
      U     : 2D array of shape (N+1, N+1), same as MATLAB version
    """
    ITE = 2
    N = 128

    # Initialize U and boundary to zero
    U = np.zeros((N+1, N+1))

    # Multigrid auxiliary arrays
    NN = N//32 - 1
    U2h   = np.zeros(N//2 + 1)
    U4h   = np.zeros(N//4 + 1)
    U8h   = np.zeros(N//8 + 1)
    U16h  = np.zeros(N//16 + 1)
    U32h  = np.zeros(N//32 + 1)
    U2ht  = np.zeros_like(U2h)
    U4ht  = np.zeros_like(U4h)
    U8ht  = np.zeros_like(U8h)
    U16ht = np.zeros_like(U16h)
    Uht   = np.zeros_like(U)

    A32h = np.zeros((NN, NN))
    r32h = np.zeros(NN)

    RH    = np.zeros(N+1)
    R2H   = np.zeros(N//2 + 1)
    R4H   = np.zeros(N//4 + 1)
    R8H   = np.zeros(N//8 + 1)
    R16H  = np.zeros(N//16 + 1)

    F2H   = np.zeros_like(R2H)
    F4H   = np.zeros_like(R4H)
    F8H   = np.zeros_like(R8H)
    F16H  = np.zeros_like(R16H)

    # Grid spacings
    H1, H2, H4, H8, H16, H32 = 1, 2, 4, 8, 16, 32

    # Precompute relaxation parameters
    p1  = dt*Theta/(H1**2)
    p2  = dt*Theta/(H2**2)
    p4  = dt*Theta/(H4**2)
    p8  = dt*Theta/(H8**2)
    p16 = dt*Theta/(H16**2)
    p32 = dt*Theta/(H32**2)

    # Initial three-point operator on coarsest grid
    A32h[0, :3] = [1+2*p32, -p32,       0]
    A32h[1, :3] = [-p32,     1+2*p32, -p32]
    A32h[2, :3] = [0,        -p32,     1+2*p32]

    # Main loop over rows
    for i in range(1, N):
        esp = 1.0
        kkk = 0

        # Single V-cycle (while esp > tol && kkk <=1)
        while esp > 1e-6 and kkk <= 1:
            kkk += 1
            U0 = U.copy()

            # ---- h-relax (Gauss-Seidel) ----
            for _ in range(ITE):
                VV = U.copy()
                for j in range(1, N):
                    U[i, j] = (
                        u[i, j]
                        - dt*b[i, j]
                        + p1*(u[i+1, j] - 2*u[i, j] + u[i-1, j])
                        + p1*(VV[i, j+1] + VV[i, j-1])
                    ) / (1 + 2*p1)

            # ---- compute fine-grid residual RH ----
            for j in range(0, N-1):
                idx = j + 1
                RH[idx] = (
                    u[i, idx]
                    - dt*b[i, idx]
                    + p1*(u[i+1, idx] - 2*u[i, idx] + u[i-1, idx])
                    + p1*(U[i, idx+1] + U[i, idx-1])
                    - (1 + 2*p1)*U[i, idx]
                )

            # ---- restrict to 2h grid ----
            for j in range(1, N//2):
                ii = j - 1
                F2H[j] = 0.25*(RH[2*ii] + 2*RH[2*ii+1] + RH[2*ii+2])

            # ---- 2h-relax ----
            VV2h = np.zeros_like(U2h)
            for _ in range(ITE):
                for j in range(1, N//2):
                    U2h[j] = (F2H[j] + p2*(VV2h[j+1] + VV2h[j-1]))/(1 + 2*p2)
                VV2h[:] = U2h

            # ---- compute 2h residual, restrict to 4h ----
            for j in range(0, N//2-1):
                idx = j + 1
                R2H[idx] = F2H[idx] + p2*(U2h[idx+1] + U2h[idx-1]) - (1+2*p2)*U2h[idx]
            for j in range(1, N//4):
                ii = j - 1
                F4H[j] = 0.25*(R2H[2*ii] + 2*R2H[2*ii+1] + R2H[2*ii+2])

            # ---- 4h-relax ----
            VV4h = np.zeros_like(U4h)
            for _ in range(ITE):
                for j in range(1, N//4):
                    U4h[j] = (F4H[j] + p4*(VV4h[j+1] + VV4h[j-1]))/(1 + 2*p4)
                VV4h[:] = U4h

            # ---- restrict to 8h ----
            for j in range(0, N//4-1):
                idx = j + 1
                R4H[idx] = F4H[idx] + p4*(U4h[idx+1] + U4h[idx-1]) - (1+2*p4)*U4h[idx]
            for j in range(1, N//8):
                ii = j - 1
                F8H[j] = 0.25*(R4H[2*ii] + 2*R4H[2*ii+1] + R4H[2*ii+2])

            # ---- 8h-relax ----
            VV8h = np.zeros_like(U8h)
            for _ in range(ITE):
                for j in range(1, N//8):
                    U8h[j] = (F8H[j] + p8*(VV8h[j+1] + VV8h[j-1]))/(1 + 2*p8)
                VV8h[:] = U8h

            # ---- restrict to 16h ----
            for j in range(0, N//8-1):
                idx = j + 1
                R8H[idx] = F8H[idx] + p8*(U8h[idx+1] + U8h[idx-1]) - (1+2*p8)*U8h[idx]
            for j in range(1, N//16):
                ii = j - 1
                F16H[j] = 0.25*(R8H[2*ii] + 2*R8H[2*ii+1] + R8H[2*ii+2])

            # ---- 16h-relax ----
            VV16h = np.zeros_like(U16h)
            for _ in range(ITE):
                for j in range(1, N//16):
                    U16h[j] = (F16H[j] + p16*(VV16h[j+1] + VV16h[j-1]))/(1 + 2*p16)
                VV16h[:] = U16h

            # ---- restrict to 32h and solve exactly ----
            for j in range(1, N//32):
                ii = j - 1
                r32h[ii] = 0.25*(R16H[2*ii] + 2*R16H[2*ii+1] + R16H[2*ii+2])
            temp = np.linalg.solve(A32h, r32h)
            # inject back
            for j in range(3):
                U32h[j+1] = temp[j]

            # ---- prolongation from 32h to 16h ----
            for j in range(N//32):
                ii = j
                U16ht[2*ii+1] = U32h[ii+1]
                U16ht[2*ii+2] = 0.5*(U32h[ii+1] + U32h[ii+2])
            for j in range(1, N//16):
                U16h[j] += U16ht[j]

            # ---- upward relax on 16h, 8h, 4h, 2h ----
            for _ in range(ITE):
                VV16h[:] = U16h
                for j in range(1, N//16):
                    U16h[j] = (F16H[j] + p16*(VV16h[j+1] + VV16h[j-1]))/(1 + 2*p16)

            for j in range(N//16):
                ii = j
                U8ht[2*ii+1] = U16h[ii+1]
                U8ht[2*ii+2] = 0.5*(U16h[ii+1] + U16h[ii+2])
            for j in range(1, N//8):
                U8h[j] += U8ht[j]

            for _ in range(ITE):
                VV8h[:] = U8h
                for j in range(1, N//8):
                    U8h[j] = (F8H[j] + p8*(VV8h[j+1] + VV8h[j-1]))/(1 + 2*p8)

            for j in range(N//8):
                ii = j
                U4ht[2*ii+1] = U8h[ii+1]
                U4ht[2*ii+2] = 0.5*(U8h[ii+1] + U8h[ii+2])
            for j in range(1, N//4):
                U4h[j] += U4ht[j]

            for _ in range(ITE):
                VV4h[:] = U4h
                for j in range(1, N//4):
                    U4h[j] = (F4H[j] + p4*(VV4h[j+1] + VV4h[j-1]))/(1 + 2*p4)

            for j in range(N//4):
                ii = j
                U2ht[2*ii+1] = U4h[ii+1]
                U2ht[2*ii+2] = 0.5*(U4h[ii+1] + U4h[ii+2])
            for j in range(1, N//2):
                U2h[j] += U2ht[j]

            for _ in range(ITE):
                VV2h[:] = U2h
                for j in range(1, N//2):
                    U2h[j] = (F2H[j] + p2*(VV2h[j+1] + VV2h[j-1]))/(1 + 2*p2)

            # ---- inject to fine grid and final h-relax ----
            for j in range(N//2):
                ii = j
                Uht[i, 2*ii+1] = U2h[ii+1]
                Uht[i, 2*ii+2] = 0.5*(U2h[ii+1] + U2h[ii+2])

            for j in range(1, N):
                U[i, j] = Uht[i, j] + U[i, j]

            for _ in range(ITE):
                VVh = U.copy()
                for j in range(1, N):
                    U[i, j] = (
                        u[i, j]
                        - dt*b[i, j]
                        + p1*(u[i+1, j] - 2*u[i, j] + u[i-1, j])
                        + p1*(VVh[i, j+1] + VVh[i, j-1])
                    ) / (1 + 2*p1)

            # update convergence metric (optional)
            # HH = U - U0
            # esp = np.max(np.abs(HH))

        # end while

    # end for i
    return U
