import numpy as np
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt


def molpro_parse(filename):
    time, energy = None, None
    with open(filename) as f:
        lines = f.readlines()
        try:
            if 'uCCSD' in filename:
                energy = float(lines[-3].split()[0])
            else:
                energy = float(lines[-3])
        except:
            return None, None
            #print("Could not read energies of %s" % filename)

        for line in lines[-3::-1]:
            if "CPU TIMES" in line:
                time = float(line.split()[3])
                break
        else:
            return None, None
        return energy, time

def is_none(x):
    return isinstance(x, type(None))

def filename_parse(filename):
    tokens = filename.split("/")[-1].split("_")
    mol = tokens[0]
    func = tokens[1]
    basis = tokens[2].split(".")[0]
    unrestricted = len(tokens) == 4

    return mol, func, basis, unrestricted

def parse_molpro(filenames):

    molecules = []
    functionals = []
    basis_sets = []
    unrestricted_flags = []
    energies = []
    timings = []
    for filename in filenames:
        energy, time = molpro_parse(filename)
        if energy == None:
            continue
        mol, func, basis, unrestricted = filename_parse(filename)
        # redundancy in the dataset
        if mol == 'C2H6':
            mol = 'Et-H'
        # Remove trailing -
        if mol[-1] == "-":
            mol = mol[:-1]

        molecules.append(mol)
        functionals.append(func)
        basis_sets.append(basis)
        unrestricted_flags.append(unrestricted)
        energies.append(energy)
        timings.append(time)
    # hartree to kcal/mol
    energies = np.asarray(energies) * 627.509

    d = {"molecule": molecules,
            "functional": functionals,
            "basis": basis_sets,
            "unrestricted": unrestricted_flags,
            "energy": energies,
            "time": timings}
    
    df = pd.DataFrame.from_dict(d)
    # Remove duplicates
    pd.DataFrame.drop_duplicates(df, inplace=True, subset=['functional', 'molecule','basis','unrestricted'])

    unique_functionals = df.functional.unique().tolist()
    unique_molecules = df.molecule.unique().tolist()
    unique_basis = df.basis.unique().tolist()
    #print(unique_functionals)
    #print(unique_basis)
    #quit()

    missing_mols = []
    missing_functional = []
    missing_basis = []
    missing_unrestricted = []
    for mol in unique_molecules:
        for func in unique_functionals:
            if func == 'uCCSD':
                sub = df.loc[(df.molecule == mol) & (df.functional == func)]
                if sub.size == 0:
                    #print(func, mol, basis, unres)
                    missing_mols.append(mol)
                    missing_functional.append(func)
                    missing_basis.append(basis)
                    missing_unrestricted.append(unres)
                elif sub.size != 6:
                    quit("Missed something")
                continue
            for basis in unique_basis:
                for unres in True, False:
                    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
                    if sub.size == 0:
                        #print(func, mol, basis, unres)
                        missing_mols.append(mol)
                        missing_functional.append(func)
                        missing_basis.append(basis)
                        missing_unrestricted.append(unres)
                    elif sub.size != 6:
                        quit("Missed something")


    d = {"molecule": missing_mols,
            "functional": missing_functional,
            "basis": missing_basis,
            "unrestricted": missing_unrestricted,
            "energy": [None]*len(missing_mols),
            "time": [None]*len(missing_mols)
            }

    df2 = pd.DataFrame.from_dict(d)

    df = df.append(df2, ignore_index = True)

    return df

def print_missing(df):
    missing = df.loc[(df.energy.isnull())]
    if missing.size > 0:
        print(missing)

def parse_reactions(reaction_filename, df):
    # sort so the order for the reactions are correct
    df.sort_values(['functional', 'basis', 'unrestricted', 'molecule'], inplace=True)
    # make new dataframe for the reactions with the same structure as df
    dfr = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    # rename molecule to reaction
    dfr.rename(index=str, columns={'molecule':'reaction'}, inplace=True)
    mol_list = df.molecule.unique().tolist()
    with open(reaction_filename) as f:
        lines = f.readlines()
        for line in lines:
            reactants, products = line.split(";")
            reactants = reactants.split()
            products = products.split()
            for i in reactants + products:
                if i not in mol_list:
                    quit("molecule %s found in reaction list, but not in parsed molecules" % str(i))

            reaction_name = "+".join(reactants) + "->" + "+".join(products)

            df_reaction = df.loc[df.molecule == reactants[0]].copy()
            df_reaction.rename(index=str, columns={'molecule':'reaction'}, inplace=True)
            # set energy and time to 0 makes the next part a bit easier
            df_reaction.energy = 0
            df_reaction.time = 0
            df_reaction['reaction'] = pd.Series([reaction_name]*(len(df_reaction.energy.tolist())), index=df_reaction.index) #df_reaction.assign('reaction'=pd.Series([reaction_name for _ in range(len(df_reaction.energy.tolist()))])
            for reactant in reactants:
                df_reaction.energy -= df.loc[df.molecule == reactant].energy.as_matrix()
                df_reaction.time += df.loc[df.molecule == reactant].time.as_matrix()
            for product in products:
                df_reaction.energy += df.loc[df.molecule == product].energy.as_matrix()
                df_reaction.time += df.loc[df.molecule == product].time.as_matrix()

            # subtract uCCSD reference
            df_reaction2 = df_reaction.loc[df_reaction.functional != 'uCCSD']
            df_reaction2 = df_reaction2.assign(energy = lambda x: x.energy-df_reaction.loc[df_reaction.functional == 'uCCSD'].energy.as_matrix()[0])

            dfr = dfr.append(df_reaction2, ignore_index = True)
    return dfr


def main(mol_df_name = None, reac_df_name = None):
    data_set_name = "abde12"
    if is_none(mol_df_name):
        filenames = glob.glob("../portfolio_datasets/%s/*.out" % data_set_name)
        mol_df = parse_molpro(filenames)
        # SOGGA11 doesn't converge for hydrogen
        mol_df = mol_df[(mol_df.functional != "SOGGA11") & (mol_df.functional != "SOGGA11-X")]
        print_missing(mol_df)
        mol_df.to_pickle(data_set_name+'_mol.pkl')
    else:
        mol_df = pd.read_pickle(mol_df_name)
    if is_none(reac_df_name):
        reac_df = parse_reactions("%s_reactions" % data_set_name, mol_df)
        reac_df['dataset'] = data_set_name
        reac_df.to_pickle(data_set_name+'_reac.pkl')
    else:
        reac_df = pd.read_pickle(reac_df_name)

    #pd.options.display.max_rows = 9999999
    #print(reac_df.loc[(reac_df.functional == 'M06-2X') & (reac_df.basis == 'qzvp') & (reac_df.unrestricted == True)])
    #uniq_functional = reac_df.functional.unique()
    #uniq_basis = reac_df.basis.unique()
    #uniq_reaction = reac_df.reaction.unique()
    #for b in ['qzvp']:
    #    for f in uniq_functional:
    #        df = reac_df.loc[(reac_df.functional == f) & (reac_df.basis == b) & (reac_df.unrestricted == True)]
    #        df.plot(x='reaction', y='energy')
    #    plt.show()
    #    quit()

    #for c in uniq_compound:
    #    df_c = df.loc[df.compound == c]

    #df.to_pickle('lol.pkl')

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        main()

    #df.loc[(df.energy.isnull())].to_csv('missing')
    #d = {}
    #unique_functionals = df.functional.unique().tolist()
    #unique_molecules = df.molecule.unique().tolist()
    #unique_basis = df.basis.unique().tolist()
    #print(unique_functionals)
    #print(unique_basis)
    #for func in unique_functionals:
    #    for basis in unique_basis:
    #        for unres in True, False:
    #            sub = df.loc[(df.functional == func) & (df.basis == basis) & (df.unrestricted == unres) & (df.energy.isnull())]
    #            if sub.size == 0:
    #                pass
    #            else:
    #                if sub.size not in d: d[sub.size] = 0
    #                d[sub.size] += 1
    #            #if sub.size == 126:
    #            #    print(func, basis, unres)
    #            if sub.size == 6:
    #                print( sub)
    #                #for mol in unique_molecules:
    #                #    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
    #                #    if sub.size != 6:
    #                #        print(mol)

