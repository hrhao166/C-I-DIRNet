import numpy as np

def MCG1Dx(b: np.ndarray, dt: float, Theta: float, u: np.ndarray) -> np.ndarray:
    """
    Python translation of the MATLAB function MCG1Dx.
    Inputs:
        b     : numpy array of shape (N+1, N+1)
        dt    : time step (float)
        Theta : parameter (float)
        u     : numpy array of shape (N+1, N+1)
    Output:
        U     : numpy array of shape (N+1, N+1)
    """
    # Parameters
    ITE = 2
    N = 128

    # Initialize solution and multigrid arrays
    U = np.zeros((N+1, N+1))
    U2h = np.zeros((N//2+1,))
    U4h = np.zeros((N//4+1,))
    U8h = np.zeros((N//8+1,))
    U16h = np.zeros((N//16+1,))
    U32h = np.zeros((N//32+1,))

    Uht = np.zeros((N+1, N+1))

    NN = N//32 - 1
    A32h = np.zeros((NN, NN))
    r32h = np.zeros((NN,))

    # Residuals and right-hand sides
    RH   = np.zeros((N+1,))
    R2H  = np.zeros((N//2+1,))
    R4H  = np.zeros((N//4+1,))
    R8H  = np.zeros((N//8+1,))
    R16H = np.zeros((N//16+1,))

    F2H  = np.zeros((N//2+1,))
    F4H  = np.zeros((N//4+1,))
    F8H  = np.zeros((N//8+1,))
    F16H = np.zeros((N//16+1,))

    # Grid spacings
    h1, h2, h4, h8, h16, h32 = 1, 2, 4, 8, 16, 32

    # Relaxation parameters
    p1  = dt*Theta/(h1**2)
    p2  = dt*Theta/(h2**2)
    p4  = dt*Theta/(h4**2)
    p8  = dt*Theta/(h8**2)
    p16 = dt*Theta/(h16**2)
    p32 = dt*Theta/(h32**2)

    # Setup coarse-grid operator for 32h
    A32h[0,0:3] = [1+2*p32, -p32, 0]
    A32h[1,0:3] = [-p32, 1+2*p32, -p32]
    A32h[2,0:3] = [0, -p32, 1+2*p32]

    # Multigrid sweep over j-direction
    for j in range(1, N):  # j = 1..N-1 corresponds to MATLAB 2..N
        esp = 1.0
        kkk = 0
        while esp > 1e-6 and kkk <= 1:
            kkk += 1
            U0 = U.copy()

            # h-relaxation (Gauss-Seidel)
            for _ in range(ITE):
                for i in range(1, N):  # i = 1..N-1
                    U[i,j] = (
                        u[i,j] + p1*(u[i,j+1] - 2*u[i,j] + u[i,j-1])
                        - dt*b[i,j]
                        + p1*(U[i+1,j] + U[i-1,j])
                    ) / (1 + 2*p1)

            # Build fine-grid residual RH
            for i in range(0, N-1):  # i = 0..N-2
                RH[i+1] = (
                    u[i+1,j] + p1*(u[i+1,j+1] - 2*u[i+1,j] + u[i+1,j-1])
                    - dt*b[i+1,j]
                    + p1*(U[i+2,j] + U[i,j])
                    - (1 + 2*p1)*U[i+1,j]
                )

            # Restrict to 2h-grid
            for i in range(2, N//2 + 1):  # i = 2..N/2
                ii = i - 1
                F2H[i] = 0.25 * (RH[2*ii] + 2*RH[2*ii+1] + RH[2*ii+2])

            # 2h-relaxation
            VV2h = np.zeros_like(U2h)
            for _ in range(ITE):
                for i in range(1, N//2):  # i = 1..N/2-1
                    U2h[i] = (F2H[i] + p2*(VV2h[i+1] + VV2h[i-1])) / (1 + 2*p2)
                VV2h[:] = U2h[:]

            # Compute 2h residual R2H
            for i in range(0, N//2-1):  # i = 0..N/2-2
                R2H[i+1] = (
                    F2H[i+1] + p2*(U2h[i+2] + U2h[i])
                    - (1 + 2*p2)*U2h[i+1]
                )

            # Restrict to 4h-grid
            for i in range(2, N//4 + 1):  # i = 2..N/4
                ii = i - 1
                F4H[i] = 0.25 * (R2H[2*ii] + 2*R2H[2*ii+1] + R2H[2*ii+2])

            # 4h-relaxation
            VV4h = np.zeros_like(U4h)
            for _ in range(ITE):
                for i in range(1, N//4):  # i = 1..N/4-1
                    U4h[i] = (F4H[i] + p4*(VV4h[i+1] + VV4h[i-1])) / (1 + 2*p4)
                VV4h[:] = U4h[:]

            # Compute 4h residual R4H
            for i in range(0, N//4-1):
                R4H[i+1] = (
                    F4H[i+1] + p4*(U4h[i+2] + U4h[i])
                    - (1 + 2*p4)*U4h[i+1]
                )

            # Restrict to 8h-grid
            for i in range(2, N//8 + 1):
                ii = i - 1
                F8H[i] = 0.25 * (R4H[2*ii] + 2*R4H[2*ii+1] + R4H[2*ii+2])

            # 8h-relaxation
            VV8h = np.zeros_like(U8h)
            for _ in range(ITE):
                for i in range(1, N//8):
                    U8h[i] = (F8H[i] + p8*(VV8h[i+1] + VV8h[i-1])) / (1 + 2*p8)
                VV8h[:] = U8h[:]

            # Compute 8h residual R8H
            for i in range(0, N//8-1):
                R8H[i+1] = (
                    F8H[i+1] + p8*(U8h[i+2] + U8h[i])
                    - (1 + 2*p8)*U8h[i+1]
                )

            # Restrict to 16h-grid
            for i in range(2, N//16 + 1):
                ii = i - 1
                F16H[i] = 0.25 * (R8H[2*ii] + 2*R8H[2*ii+1] + R8H[2*ii+2])

            # 16h-relaxation
            VV16h = np.zeros_like(U16h)
            for _ in range(ITE):
                for i in range(1, N//16):
                    U16h[i] = (F16H[i] + p16*(VV16h[i+1] + VV16h[i-1])) / (1 + 2*p16)
                VV16h[:] = U16h[:]

            # Compute 16h residual R16H
            for i in range(0, N//16-1):
                R16H[i+1] = (
                    F16H[i+1] + p16*(U16h[i+2] + U16h[i])
                    - (1 + 2*p16)*U16h[i+1]
                )

            # Restrict to 32h-grid (solve coarse)
            for i in range(2, N//32 + 1):
                ii = i - 1
                r32h[ii] = 0.25 * (R16H[2*ii] + 2*R16H[2*ii+1] + R16H[2*ii+2])

            # 32h coarse solve (3-point)
            temp = np.linalg.solve(A32h, r32h)
            for i in range(3):
                U32h[i+1] = temp[i]

            # Prolongate and correct: 16h intermediate
            for i in range(N//16):
                ii = i
                Uht[2*ii+1, j] = U32h[ii]
                Uht[2*ii+2, j] = 0.5*(U32h[ii] + U32h[ii+1])
            for i in range(1, N//16 + 1):
                U16h[i] += Uht[i*2-1, j] if i*2-1 <= N else 0

            # Post-16h-relax
            for _ in range(ITE):
                for i in range(1, N//16):
                    U16h[i] = (F16H[i] + p16*(U16h[i+1] + U16h[i-1])) / (1 + 2*p16)

            # Prolongate to 8h-grid
            for i in range(N//8):
                ii = i
                Uht[2*ii+1, j] = U16h[ii]
                Uht[2*ii+2, j] = 0.5*(U16h[ii] + U16h[ii+1])
            for i in range(1, N//8 + 1):
                U8h[i] += Uht[i*2-1, j] if i*2-1 <= N else 0

            # Post-8h-relax
            for _ in range(ITE):
                for i in range(1, N//8):
                    U8h[i] = (F8H[i] + p8*(U8h[i+1] + U8h[i-1])) / (1 + 2*p8)

            # Prolongate to 4h-grid
            for i in range(N//4):
                ii = i
                Uht[2*ii+1, j] = U8h[ii]
                Uht[2*ii+2, j] = 0.5*(U8h[ii] + U8h[ii+1])
            for i in range(1, N//4 + 1):
                U4h[i] += Uht[i*2-1, j] if i*2-1 <= N else 0

            # Post-4h-relax
            for _ in range(ITE):
                for i in range(1, N//4):
                    U4h[i] = (F4H[i] + p4*(U4h[i+1] + U4h[i-1])) / (1 + 2*p4)

            # Prolongate to 2h-grid
            for i in range(N//2):
                ii = i
                Uht[2*ii+1, j] = U4h[ii]
                Uht[2*ii+2, j] = 0.5*(U4h[ii] + U4h[ii+1])
            for i in range(1, N//2 + 1):
                U2h[i] += Uht[i*2-1, j] if i*2-1 <= N else 0

            # Post-2h-relax
            for _ in range(ITE):
                for i in range(1, N//2):
                    U2h[i] = (F2H[i] + p2*(U2h[i+1] + U2h[i-1])) / (1 + 2*p2)

            # Prolongate to fine grid and correct Uht
            for i in range(N//2):
                ii = i
                Uht[2*ii+1, j] = U2h[ii]
                Uht[2*ii+2, j] = 0.5*(U2h[ii] + U2h[ii+1])

            # Final h-correction
            for i in range(1, N):
                U[i,j] = Uht[i,j] + U[i,j]

            # Final h-relax
            for _ in range(ITE):
                for i in range(1, N):
                    U[i,j] = (
                        u[i,j] + p1*(u[i,j+1] - 2*u[i,j] + u[i,j-1])
                        - dt*b[i,j]
                        + p1*(U[i+1,j] + U[i-1,j])
                    ) / (1 + 2*p1)

            # Compute max update
            esp = np.max(np.abs(U - U0))

        # end while
    # end for j

    return U
