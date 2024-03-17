import numpy as np
import Toeplitz_System
import CG
import PCG

sizes = [50, 100, 200, 400, 800, 1600, 3200]
ps = [2, 1, 1 / 10, 1 / 100]
systems = [Toeplitz_System.generate_power_decay_system, Toeplitz_System.generate_FFT_system]
solvers = [CG.conjugate_gradient, PCG.preconditioned_conjugate_gradient]
preconditioners = [PCG.G_Strang_first_column, PCG.G_Strang_first_column]

for system in systems:
    for size in sizes:
        for p in ps:
            A = system(size, p)
            b = np.random.rand(size)
            for solver in solvers:
                if solver == PCG.preconditioned_conjugate_gradient:
                    for preconditioner in preconditioners:
                        solver(A, b, preconditioner)
                        print(f"System: {system.__name__}, Size: {size}, p: {p}, Solver: {solver.__name__}, "
                              f"Preconditioner: {preconditioner.__name__}")
                        print("--------------------------------------------------------------------------")
                else:
                    solver(A, b)
                    print(f"System: {system.__name__}, Size: {size}, p: {p}, Solver: {solver.__name__}")
                    print("--------------------------------------------------------------------------")