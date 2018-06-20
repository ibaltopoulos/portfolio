import numpy as np
import glob
import pandas as pd
import sys
#import matplotlib.pyplot as plt


def data_parse(filename):
    """
    Opens a molpro output file and parses the energy and computation time.
    Could be extended to get out more information.
    """
    time, energy = None, None
    with open(filename) as f:
        lines = f.readlines()
        try:
            # uCCSD has multiple output.
            energy = float(lines[-3].split()[0])
        except:
            return None, None

        for line in lines[-3::-1]:
            if "CPU TIMES" in line:
                time = float(line.split()[3])
                if "uCCSD" not in filename:
                    break
            if "F12b" in line:
                energy = float(line.split()[-1])
                break
        else:
            return None, None
        return energy, time

def is_none(x):
    """
    Helper function since x == None doesn't work well on arrays. Returns True if x is None.
    """
    return isinstance(x, type(None))

def filename_parse(filename):
    """
    Parses filename to get information about the method used.
    It is assumed that u-pbe/avtz on hcn will be named
    hnc_pbe_avtz_u.xxx, where xxx is an arbitrary extension.
    """
    tokens = filename.split("/")[-1].split("_")
    mol = tokens[0]
    func = tokens[1]
    basis = tokens[2].split(".")[0]
    unrestricted = (len(tokens) == 4)

    return mol, func, basis, unrestricted

def parse_molpro(filenames, data_set):
    """
    Parse data from all filenames. data_set is used to create file output.
    """

    molecules = []
    functionals = []
    basis_sets = []
    unrestricted_flags = []
    energies = []
    timings = []
    for filename in filenames:
        # Parse the data files
        energy, time = data_parse(filename)
        # skip if the calculation has failed
        if energy == None:
            continue
        # Parse the filename
        mol, func, basis, unrestricted = filename_parse(filename)

        # Special case for abde12 due to redundancy in the 
        # naming of the files
        if data_set == "abde12":
            # redundancy in the dataset
            if mol == 'C2H6':
                mol = 'Et-H'
            # Remove trailing -
            if mol[-1] == "-":
                mol = mol[:-1]

        # The SOGGA11 and SOGGA11-X functionals didn't converge for hydrogen
        # And was removed from the set
        if func.lower() in ['sogga11','sogga11-x']:
            continue

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

    # Do a sanity check on everything and make
    # sure that every combination of methods exist
    # The missing ones are given an energy of None
    missing_mols = []
    missing_functional = []
    missing_basis = []
    missing_unrestricted = []
    for mol in unique_molecules:
        for func in unique_functionals:
            # The CCSD is only unrestricted and has the same basis.
            if func == 'uCCSD':
                sub = df.loc[(df.molecule == mol) & (df.functional == func)]
                if sub.size == 0:
                    missing_mols.append(mol)
                    missing_functional.append(func)
                    missing_basis.append(basis)
                    missing_unrestricted.append(False)
                elif sub.size != 6:
                    quit("Missed something")
                continue
            for basis in unique_basis:
                for unres in True, False:
                    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
                    if sub.size == 0:
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

def print_missing(df, name):
    """
    Print any method with missing energy
    """
    missing = df.loc[(df.energy.isnull())]
    if missing.size > 0:
        print("Missing elements in the %s data set" % name)
        print(missing)

def parse_reactions(reaction_filename, df):
    """
    Create the reactions described in reaction_filename, from the parsed molecules.
    """
    # Sort so the order for the reactions are correct
    df.sort_values(['functional', 'basis', 'unrestricted', 'molecule'], inplace=True)
    # make new dataframe for the reactions with the same structure as df
    dfr = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    # rename molecule to reaction
    dfr.rename(index=str, columns={'molecule':'reaction'}, inplace=True)
    mol_list = df.molecule.unique().tolist()
    # Open the reaction file
    with open(reaction_filename) as f:
        lines = f.readlines()
    for line in lines:
        # Skip comments
        if line.startswith("#"):
            continue

        # Parse each reaction
        reactants, products, main_class, sub_class = line.split(",")
        reactants = reactants.split()
        products = products.split()
        for i in reactants + products:
            if i not in mol_list:
                quit("molecule %s found in reaction list, but not in parsed molecules" % str(i))
        reaction_name = "+".join(reactants) + "->" + "+".join(products)

        # Make a dataframe for this reaction
        df_reaction = df.loc[df.molecule == reactants[0]].copy()
        df_reaction.rename(index=str, columns = {'molecule':'reaction'}, inplace=True)
        # set energy and time to 0 makes the next part a bit easier
        df_reaction.energy = 0
        df_reaction.time = 0
        # Add up the time cost and energy of the reaction for all methods
        df_reaction['reaction'] = pd.Series([reaction_name]*(len(df_reaction.energy.tolist())), index=df_reaction.index) 
        for reactant in reactants:
            df_reaction.energy -= df.loc[df.molecule == reactant].energy.as_matrix()
            df_reaction.time += df.loc[df.molecule == reactant].time.as_matrix()
        for product in products:
            df_reaction.energy += df.loc[df.molecule == product].energy.as_matrix()
            df_reaction.time += df.loc[df.molecule == product].time.as_matrix()

        # subtract uCCSD reference
        df_reaction['error'] = df_reaction.energy - df_reaction.loc[df_reaction.functional == 'uCCSD'].energy.as_matrix()[0]

        ## Make sure that reaction energy is negative
        #df_tmp = df_reaction.loc[(df_reaction.functional == 'uCCSD') & (df_reaction.energy > 0)]
        #if df_tmp.size > 0:
        #    print("Reverse reaction way of")
        #    print(df_tmp)

        # Add the reaction class
        df_reaction['main_class'] = main_class.strip()
        df_reaction['sub_class'] =  sub_class.strip()

        # Add everything but the uCCSD method to the dataframe
        dfr = dfr.append(df_reaction[df_reaction.functional != 'uCCSD'], ignore_index = True)
    return dfr

def make_pickles(data_set_name, data_set_path = "../portfolio_datasets/", pickle_path = "pickles/"):
    """
    Create the pandas dataframe pickles if the don't already exist.
    """

    path = data_set_path + "/" + data_set_name
    mol_df_name = pickle_path + data_set_name + "_mol.pkl"
    reac_df_name = pickle_path + data_set_name + "_reac.pkl"

    # Try to read the dataset pickle, else make it.
    try:
        mol_df = pd.read_pickle(mol_df_name)
    except FileNotFoundError:
        filenames = glob.glob(path + "/*.out")
        mol_df = parse_molpro(filenames, data_set_name)
        ## SOGGA11 doesn't converge for hydrogen
        #mol_df = mol_df.loc[(mol_df.functional != "SOGGA11") & (mol_df.functional != "SOGGA11-X")]
        print_missing(mol_df, data_set_name)
        mol_df.to_pickle(mol_df_name)


    # Try to read the reaction pickle, else make it.
    try:
        reac_df = pd.read_pickle(reac_df_name)
    except FileNotFoundError:
        reac_df = parse_reactions(data_set_name + "_reactions", mol_df)
        reac_df['dataset'] = data_set_name
        set_median_timings(reac_df)
        reac_df.to_pickle(reac_df_name)
    return reac_df

def set_median_timings(df):
    """
    Since we know that some methods should take the same time to
    be computed, the median time is used for all of them.
    This should be updated if more methods is added.
    """
    unique_reactions = df.reaction.unique()
    unique_basis = df.basis.unique()

    for reac in unique_reactions:
        for un in True, False:
            for bas in unique_basis:
                # First ggas
                gga_df = df.loc[(df.reaction == reac) & (df.basis == bas) & (df.unrestricted == un) & 
                        (df.isin(['B88X', 'B', 'BECKE', 'B-LYP', 'B-P', 'B-VWN', 'CS', 'D', 'HFB', 'HFS', 
                            'LDA', 'LSDAC', 'LSDC', 'LYP88', 'PBE', 'PBEREV', 'PW91', 'S', 'SLATER', 'SOGGA11', 
                            'SOGGA', 'S-VWN', 'VS99', 'VWN80', 'VWN']).functional)]
                # then hybrids
                hybrid_df = df.loc[(df.reaction == reac) & (df.basis == bas) & (df.unrestricted == un) & (df.isin(['B3LYP3','B3LYP5','B97', 'B97R', 'BH-LYP', 'PBE0', 'PBE0MOL', 'SOGGA11-X']).functional)]
                # then mega gga
                meta_df = df.loc[(df.reaction == reac) & (df.basis == bas) & (df.unrestricted == un) & (df.isin(['M06-L','M11-L','MM06-L']).functional)]
                # then meta hybrids
                hybrid_meta_df = df.loc[(df.reaction == reac) & (df.basis == bas) & (df.unrestricted == un) & (df.isin(['M05-2X','M05','M06-2X','M06','M06-HF','M08-HX','M08-SO','MM05-2X','MM05','MM06-2X','MM06','MM06-HF']).functional)]

                # Get the median time and set it.
                time = np.median(gga_df.time.as_matrix())
                df.at[gga_df.index, "time"] = time
                time = np.median(hybrid_df.time.as_matrix())
                df.at[hybrid_df.index, "time"] = time
                time = np.median(meta_df.time.as_matrix())
                df.at[meta_df.index, "time"] = time
                time = np.median(hybrid_meta_df.time.as_matrix())
                df.at[hybrid_meta_df.index, "time"] = time

def simplify_timings(base_df, reaction_name):
    """
    Set timings for all reactions to be equal to a named reaciton
    """
    df = base_df.copy()
    for func in df.functional.unique():
        for basis in df.basis.unique():
            for unres in True, False:
                sub_df = df.loc[(df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
                time = sub_df.loc[(sub_df.reaction == reaction_name)].time.as_matrix()[0]
                df.at[sub_df.index, "time"] = time
    return df

def main():
    """
    Create all the reaction pickles
    """
    abde12_reac = make_pickles("abde12")
    df = abde12_reac
    df = df.loc[(df.functional == "M06-2X") & (df.basis == "avtz") & (df.unrestricted == True)]
    print(df[["reaction", "energy"]])
    quit()
    nhtbh38_reac = make_pickles("nhtbh38")

    # combine
    df = abde12_reac.append(nhtbh38_reac, ignore_index = True)
    df.to_pickle("pickles/combined_reac.pkl")
    #print(df.head())
    df2 = df.loc[(df.functional == "PBE0") & (df.basis == "qzvp") & (df.unrestricted == True)].time
    slow_name = df.at[df2.idxmax(), "reaction"]
    slow_df = simplify_timings(df, slow_name)
    #print(slow_name)
    slow_df.to_pickle("combined_reac_slow.pkl")

    # Set cost to match the most expensive reaction
    df = set_expensive_timings(df)
    df.to_pickle("pickles/combined_high_cost.pkl")


if __name__ == "__main__":
    main()
