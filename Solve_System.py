import numpy as np
import Toeplitz_System
import CG
import PCG

sizes = [50, 100, 200, 400, 800, 1600, 3200]
ps = [2, 1, 1 / 10, 1 / 100]
solvers = [CG.conjugate_gradient, PCG.preconditioned_conjugate_gradient]
preconditioners = [PCG.G_Strang_first_column, PCG.T_Chan_first_column]


system = Toeplitz_System.generate_power_decay_system
for size in sizes:
    for p in ps:
        for solver in solvers:
            A = system(size, p)
            b = np.random.rand(size)
            if solver == CG.conjugate_gradient and system == Toeplitz_System.generate_power_decay_system:
                print(f"System: {system.__name__}, Size: {size}, p: {p}, Solver: {solver.__name__}")
                solver(A, b)
                print("--------------------------------------------------------------------------")
            else:
                for preconditioner in preconditioners:
                    print(f"System: {system.__name__}, Size: {size}, p: {p}, Solver: {solver.__name__}, "
                          f"Preconditioner: {preconditioner.__name__}")
                    solver(A, b, preconditioner)
                    print("--------------------------------------------------------------------------")


system = Toeplitz_System.generate_FFT_system
for size in sizes:
    for solver in solvers:
        A = system(size)
        b = np.random.rand(size)
        if solver == CG.conjugate_gradient and system == Toeplitz_System.generate_FFT_system:
            print(f"System: {system.__name__}, Size: {size}, Solver: {solver.__name__}")
            solver(A, b)
            print("--------------------------------------------------------------------------")
        else:
            for preconditioner in preconditioners:
                print(f"System: {system.__name__}, Size: {size}, Solver: {solver.__name__}, "
                      f"Preconditioner: {preconditioner.__name__}")
                solver(A, b, preconditioner)
                print("--------------------------------------------------------------------------")
