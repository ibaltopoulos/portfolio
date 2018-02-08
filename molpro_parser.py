import numpy as np
import glob
import pandas as pd


def molpro_parse(filename):
    with open(filename) as f:
        lines = f.readlines()
        try:
            energy = float(lines[-3])
        except:
            print("Could not read energies of %s" % filename)

        for line in lines[-3::-1]:
            if "CPU TIMES" in line:
                time = float(line.split()[3])

    return energy, time


def filename_parse(filename):
    tokens = filename.split("/")[-1].split("_")
    mol = tokens[0]
    func = tokens[1]
    basis = tokens[2].split(".")[0]
    unrestricted = len(tokens) == 4

    return mol, func, basis, unrestricted


def main():
    filenames = glob.glob("abde12/*.out")

    molecules = []
    functionals = []
    basis_sets = []
    unrestricted_flags = []
    energies = []
    timings = []
    for filename in filenames:
        mol, func, basis, unrestricted = filename_parse(filename)
        energy, time = molpro_parse(filename)
        molecules.append(mol)
        functionals.append(func)
        basis_sets.append(basis)
        unrestricted_flags.append(unrestricted)
        energies.append(energy)
        timings.append(time)

    d = {"molecule": molecules,
            "functional": functionals,
            "basis": basis_sets,
            "unrestricted": unrestricted_flags,
            "energy": energies,
            "time": timings}
    
    df = pd.DataFrame.from_dict(d)

    unique_functionals = df.functional.unique().tolist()
    unique_molecules = df.molecule.unique().tolist()
    unique_basis = df.basis.unique().tolist()

    for func in unique_functionals:
        for mol in unique_molecules:
            for basis in unique_basis:
                for unres in True, False:
                    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
                    print(func, mol, basis + unres*"_u", int(bool(sub.size)))




if __name__ == "__main__":
    main()
