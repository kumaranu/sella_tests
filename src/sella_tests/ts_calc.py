import os
from ase.io import read
from sella import Sella
from newtonnet.utils.ase_interface import MLAseCalculator
from ase import Atoms
import numpy as np
from multiprocessing import Pool


def optimize_and_analyze(index):
    mult = 1
    # run_path = 'runs'
    run_path = 'runs1'
    scratch_path = '/global/home/users/kumaranu/Documents/sella_si/singlet_tests'
    check_freqs = True
    spin = 'singlets' if mult == 1 else 'doublets'
    run_dir = os.path.join(run_path, spin, index)
    os.makedirs(run_dir, exist_ok=True)
    label = f'{spin}_{index}'
    atoms = read(f'{spin}/{index}.xyz')
    ml_path = '/global/home/users/kumaranu/Documents/NewtonNet/example/' \
              'predict/training_2/models/best_model_state.tar'
    config_path = '/global/home/users/kumaranu/Documents/NewtonNet/example/' \
                  'predict/training_2/run_scripts/config0.yml'
    mlcalculator = MLAseCalculator(model_path=ml_path,
                                   settings_path=config_path)

    atoms.set_calculator(mlcalculator)
    opt = Sella(atoms,
                internal=True,
                logfile=os.path.join(run_dir, f'{label}.log'),
                trajectory=os.path.join(run_dir, f'{label}.traj'))
    opt.run(fmax=0.01)

    if check_freqs:
        traj = read(os.path.join(run_dir, f'{label}.traj'))
        mlcalculator.calculate(traj)
        H = mlcalculator.results['hessian']
        n_atoms = np.shape(H)[0]
        A = np.reshape(H, (n_atoms * 3, n_atoms * 3))
        eigvals, eigvecs = np.linalg.eig(A)
        # eigvals_file = os.path.join(run_dir, f'hessian_eigvals.txt')
        # eigvecs_file = os.path.join(run_dir, f'hessian_eigvecs.txt')
        # np.savetxt(eigvals_file, eigvals, fmt='%.5f')
        # np.savetxt(eigvecs_file, eigvecs, fmt='%.5f')
        return_vars = {'eigvals': eigvals, 'eigvecs': eigvecs}
        return return_vars


did_not_converge = [14, 16, 17, 21, 22, 23, 35, 38, 39, 40, 41, 44, 51, 52, 53, 63, 64, 65, 68, 69, 70, 71, 77, 78, 79,
                    80, 87, 88, 89, 97, 98, 101, 112, 113, 116, 123, 124, 125, 130, 131, 135, 136, 137, 138, 139, 140,
                    146, 155, 169, 170, 177, 178, 179, 186, 187, 188, 193, 194, 218, 226, 227, 231, 232, 233, 256, 257]

if __name__ == '__main__':
    with Pool(processes=30) as pool:
        # indices = [f"{i:03}" for i in range(265)]
        indices = [f"{i:03}" for i in did_not_converge]
        pool.map(optimize_and_analyze, indices)
    '''
    for index in did_not_converge:
        index = str(index).zfill(3)
        print(index)
        try:
            optimize_and_analyze(index)
        except Exception as e:
            print(f"Error: {e}")
            continue
    '''
