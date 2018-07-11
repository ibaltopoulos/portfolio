import numpy as np
import glob
import pandas as pd
import sys

def data_parse(filename, real_filename = None):
    """
    Opens a molpro output file and parses the energy and computation time.
    """

    try:
        with open(filename) as f:
            lines = f.readlines()
        if "DCSD" in filename:
            # ccsd tends to crash with H since there's only one electron
            if filename.split("/")[-1].startswith("H-_") \
                or filename.split("/")[-1].startswith("H_") \
                or filename.split("/")[-1].startswith("F_")   and "sto-3g" in filename \
                or filename.split("/")[-1].startswith("F-_")  and "sto-3g" in filename \
                or filename.split("/")[-1].startswith("O_")   and "sto-3g" in filename \
                or filename.split("/")[-1].startswith("O-_")  and "sto-3g" in filename \
                or filename.split("/")[-1].startswith("Cl_")  and "sto-3g" in filename \
                or filename.split("/")[-1].startswith("Cl-_") and "sto-3g" in filename:
                data = parse_dcsd(lines, True)
            else:
                data =  parse_dcsd(lines)
        #elif "luCCSD" in filename:
        #    if filename.split("/")[-1].startswith("H-_") \
        #        or filename.split("/")[-1].startswith("H_"):
        #        data = parse_luccsd(lines, True)
        #    else:
        #        data =  parse_luccsd(lines)
        elif "_uCCSD" in filename:
            data =  parse_uccsd(lines)

        elif "lrmp2" in filename:
            data = parse_lrmp2(lines)
        elif "KS" in lines[-4]:
            data = parse_dft(lines)
        elif filename.split("/")[-1].startswith("H_D3") and "sto-3g" in filename:
            # special case where molpro fails to run dispersion on hydrogen
            return data_parse(filename.replace("D3",""), filename)
        elif filename.split("/")[-1].startswith("H_dc") and "sto-3g" in filename:
            # special case where molpro fails to run dc on hydrogen
            return data_parse(filename.replace("dc",""), filename)

        else:
            print("parsing error for", filename)
            quit()
    except:
        print("parsing error for", filename)
        return

    if not is_none(real_filename):
        filename = real_filename


    filename_data = filename_parse(filename)
    return (*data, *filename_data)

def parse_lrmp2(lines):
    e1 = None
    e2 = None
    ce = None
    time = None
    for i, line in enumerate(lines[-3::-1]):
        if "DF-LRMP2" in line or "HF-SCF" in line:
            energy = float(lines[-3-i+1].split()[0])
        elif "CPU TIMES" in line:
            time = float(line.split()[3])
        elif "LRMP2 correlation energy" in line:
            ce = float(line.split()[3])
        elif "Two-electron energy" in line:
            e2 = float(line.split()[2])
        elif "One-electron energy" in line:
            e1 = float(line.split()[2])
            # Since this is the last to be read
            break

    return energy, time, e1, e2, ce

def parse_uccsd(lines):
    # We only care about the energy of the reference
    e1 = None
    e2 = None
    ce = None
    time = None
    energy = None
    for line in lines[-3::-1]:
        if "!RHF-UCCSD(T)-F12b energy" in line:
            energy = float(line.split()[2])
            break
    return energy, time, e1, e2, ce

#def parse_luccsd(lines, single_atom = False):
#    e1 = None
#    e2 = None
#    ce = None
#    time = None
#    for i, line in enumerate(lines[-3::-1]):
#        if "LUCCSD(T)-F12" in line:
#            energy = float(lines[-3-i+1].split()[0])
#        elif "CPU TIMES" in line:
#            time = float(line.split()[3])
#        elif "!RHF STATE 1.1 Energy" in line:
#            if single_atom:
#                energy = float(line.split()[4])
#            ce = energy - float(line.split()[4])
#            # Since this is the last to be read
#            break
#        elif "Two-electron energy" in line:
#            e2 = float(line.split()[2])
#        elif "One-electron energy" in line:
#            e1 = float(line.split()[2])
#
#    return energy, time, e1, e2, ce

def parse_dcsd(lines, single_atom = False):
    """
    Many ugly fixes to make sure single atom
    calculations gets read correctly.
    """
    e1 = None
    e2 = None
    ce = None
    time = None
    if not single_atom:
        energy = float(lines[-3].split()[0])
    if single_atom:
        ce = 0

    for line in lines[-3::-1]:
        if "!RHF STATE 1.1 Energy" in line and single_atom:
            energy = float(line.split()[4])
            break
        elif "CPU TIMES" in line:
            time = float(line.split()[3])
        elif "DCSD correlation energy" in line:
            ce = float(line.split()[3])
        elif "Two-electron energy" in line:
            e2 = float(line.split()[2])
        elif "One-electron energy" in line:
            e1 = float(line.split()[2])
            if not single_atom:
                break


    return energy, time, e1, e2, ce

def parse_dft(lines, single_atom = False):
    e1 = None
    e2 = None
    ce = None
    time = None
    if not single_atom:
        energy = float(lines[-3].split()[0])
    if single_atom:
        ce = 0

    for line in lines[-3::-1]:
        if "!RHF STATE 1.1 Energy" in line and single_atom:
            energy = float(line.split()[4])
            break
        elif "CPU TIMES" in line:
            time = float(line.split()[3])
        elif "Density functional" in line:
            ce = float(line.split()[2])
        elif "Two-electron energy" in line:
            e2 = float(line.split()[2])
        elif "One-electron energy" in line:
            e1 = float(line.split()[2])
            # Since this is the last to be read
            if not single_atom:
                break
    return energy, time, e1, e2, ce

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

    # Since SV-P is misspelled as SV-P_, catch that
    filename = filename.replace("SV-P_", "SV-P")

    tokens = filename.split("/")[-1].split("_")
    mol = tokens[0]
    func = tokens[1]
    basis = tokens[2].split(".")[0]
    name = func + "/" + basis
    if func in ['rDCSD', 'df-lrmp2']:
        unrestricted = False
    #elif func in ['luCCSD', 'uCCSD', 'uDCSD']:
    elif func in ['uCCSD', 'uDCSD']:
        unrestricted = True
    else:
        unrestricted = (len(tokens) == 4)
        name = "u-" * unrestricted + name

    if func in ['rDCSD', 'uDCSD']:
        func = 'DCSD'

    return mol, func, basis, unrestricted, name

def parse_molpro(filenames, data_set):
    """
    Parse data from all filenames. data_set is used to create file output.
    """

    names = []
    molecules = []
    functionals = []
    basis_sets = []
    unrestricted_flags = []
    energies = []
    timings = []
    oneelectron = []
    twoelectron = []
    correlation = []
    for filename in filenames:
        # Parse the data files
        args = data_parse(filename)
        # Continue if parsing fails
        if is_none(args):
            continue
        energy, time, e1, e2, ce, mol, func, basis, unrestricted, name = args
        # skip if the calculation has failed
        if None in [energy, time, e1, e2, ce] and '_uCCSD' not in filename \
            or is_none(energy):
            print(energy, time, e1, e2, ce, filename)
            continue
        if func == "DCSD" and unrestricted == False and basis == 'sto-3g':
            print("sto-3g have been removed for rDCSD temporarily")
            continue

        # Special case for abde12 due to redundancy in the 
        # naming of the files
        if data_set == "abde12":
            # redundancy in the dataset
            if mol == 'Et-H':
                mol = 'C2H6'
            elif mol == "OCH3-":
                mol = "CH3O"
            elif mol == "tBu-H":
                mol = "C4H10"

            # Remove trailing -
            if mol[-1] == "-":
                mol = mol[:-1]

        names.append(name)
        molecules.append(mol)
        functionals.append(func)
        basis_sets.append(basis)
        unrestricted_flags.append(unrestricted)
        energies.append(energy)
        timings.append(time)
        oneelectron.append(e1)
        twoelectron.append(e2)
        correlation.append(ce)

    unrestricted = np.asarray(unrestricted, dtype = bool)

    # hartree to kcal/mol
    energies = np.asarray(energies) * 627.509

    d = {"name": names,
         "molecule": molecules,
         "functional": functionals,
         "basis": basis_sets,
         "unrestricted": unrestricted_flags,
         "energy": energies,
         "time": timings,
         "one_electron_energy": oneelectron,
         "two_electron_energy": twoelectron,
         "correlation_energy": correlation}

    df = pd.DataFrame.from_dict(d)
    # Convert to bool since pandas assume bool are int

    # Remove duplicates
    pd.DataFrame.drop_duplicates(df, inplace=True, subset=['functional', 'molecule','basis','unrestricted'])

    ## check for duplicated functionals
    #df2 = pd.DataFrame.duplicated(df, keep = False, subset=['molecule','basis','unrestricted', 'energy'])
    #print(df[df2])
    #quit()

    unique_functionals = df.functional.unique().tolist()
    unique_molecules = df.molecule.unique().tolist()
    unique_basis = df.basis.unique().tolist()

    # Do a sanity check on everything and make
    # sure that every combination of methods exist
    # The missing ones are given an energy of None
    missing_names = []
    missing_mols = []
    missing_functional = []
    missing_basis = []
    missing_unrestricted = []
    for mol in unique_molecules:
        for func in unique_functionals:
            # The CCSD is only unrestricted and has the same basis.
            if func == 'uCCSD':
                sub = df.loc[(df.molecule == mol) & (df.functional == func)]
                if sub.shape[0] == 0:
                    missing_names.append('uCCSD/avtz')
                    missing_mols.append(mol)
                    missing_functional.append(func)
                    missing_basis.append(basis)
                    missing_unrestricted.append(False)
                elif sub.shape[0] != 1:
                    quit(("Missed something", func, mol))
                continue
            for basis in unique_basis:
                # The lCCSD is only unrestricted and doesn't have all basis sets
                #if func == 'luCCSD':
                #    if basis in ['sto-3g', 'SV-P']:
                #        continue
                #    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis)]
                #    if sub.shape[0] == 0:
                #        missing_names.append('luCCSD/' + basis)
                #        missing_mols.append(mol)
                #        missing_functional.append(func)
                #        missing_basis.append(basis)
                #        missing_unrestricted.append(False)
                #    elif sub.shape[0] != 1:
                #        quit(("Missed something", func, mol, basis))
                #    continue
                # The df-lrmp2 is only restricted.
                if func == 'df-lrmp2':
                    if basis in ['sto-3g', 'SV-P']:
                        continue
                    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis)]
                    if sub.shape[0] == 0:
                        missing_names.append('df-lrmp2/' + basis)
                        missing_mols.append(mol)
                        missing_functional.append(func)
                        missing_basis.append(basis)
                        missing_unrestricted.append(True)
                    elif sub.shape[0] != 1:
                        quit(("Missed something", func, mol, basis))
                    continue
                for unres in True, False:
                    # The DCSD only have select basis sets.
                    if func == 'DCSD':
                        if basis in ['qzvp', 'avtz', 'tzvp', 'avdz']:
                            continue
                        if basis == 'sto-3g' and unres == False:
                            print("sto-3g have been removed for rDCSD temporarily")
                            continue
                        sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
                        if sub.shape[0] == 0:
                            missing_names.append('u-'*unres + func + '/' + basis)
                            missing_mols.append(mol)
                            missing_functional.append(func)
                            missing_basis.append(basis)
                            missing_unrestricted.append(True)
                        elif sub.shape[0] != 1:
                            quit(("Missed something", func, mol, basis, unres))
                        continue
                    sub = df.loc[(df.molecule == mol) & (df.functional == func) & (df.basis == basis) & (df.unrestricted == unres)]
                    if sub.shape[0] == 0:
                        missing_names.append('u-'*unres + func + '/' + basis)
                        missing_mols.append(mol)
                        missing_functional.append(func)
                        missing_basis.append(basis)
                        missing_unrestricted.append(unres)
                    elif sub.shape[0] != 1:
                        quit(("Missed something", func, mol, basis, unres, sub.size))

    d = {"name": missing_names,
         "molecule": missing_mols,
         "functional": missing_functional,
         "basis": missing_basis,
         "unrestricted": missing_unrestricted,
         "energy": [None]*len(missing_mols),
         "time": [None]*len(missing_mols), 
         "one_electron_energy": [None]*len(missing_mols),
         "two_electron_energy": [None]*len(missing_mols),
         "correlation_energy": [None]*len(missing_mols)
         }

    df2 = pd.DataFrame.from_dict(d)

    df = df.append(df2, ignore_index = True, sort = False)

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
    """.
    Cre.ate the reactions described in reaction_filename, from the parsed molecules.
    """
    # Sort so the order for the reactions are correct
    df.sort_values(['functional', 'basis', 'unrestricted', 'molecule'], inplace=True)
    # make new dataframe for the reactions with the same structure as df
    dfr = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    # rename molecule to reaction
    dfr.rename(index=str, columns={'molecule':'reaction'}, inplace=True)
    mol_list = df.molecule.unique().tolist()

    # Store reaction energies from reaction files to make sure our reference is mostly correct.
    reference_energies = []

    # Open the reaction file
    with open(reaction_filename) as f:
        lines = f.readlines()
    for line in lines:
        # Skip comments
        if line.startswith("#"):
            continue

        # Parse each reaction
        reactants, products, rtype, charge, spin, reference = line.split(",")
        reference_energies.append(float(reference))
        reactants = reactants.split()
        products = products.split()
        for i in reactants + products:
            if i not in mol_list:
                quit("molecule %s found in reaction list, but not in parsed molecules" % str(i))
        reaction_name = "+".join(reactants) + "->" + "+".join(products)

        # Make a dataframe for this reaction
        df_reaction = df.loc[df.molecule == reactants[0]].copy()
        df_reaction.rename(index=str, columns = {'molecule':'reaction'}, inplace=True)
        # set energies and time to 0 makes the next part a bit easier
        df_reaction.energy = 0
        df_reaction.one_electron_energy = 0
        df_reaction.two_electron_energy = 0
        df_reaction.correlation_energy = 0
        df_reaction.time = 0
        # Add up the time cost and energy of the reaction for all methods
        df_reaction['reaction'] = pd.Series([reaction_name]*(len(df_reaction.energy.tolist())), index=df_reaction.index) 
        for reactant in reactants:
            df_reaction.energy -= df.loc[df.molecule == reactant].energy.values
            df_reaction.one_electron_energy -= df.loc[df.molecule == reactant].one_electron_energy.values
            df_reaction.two_electron_energy -= df.loc[df.molecule == reactant].two_electron_energy.values
            df_reaction.correlation_energy -= df.loc[df.molecule == reactant].correlation_energy.values
            df_reaction.time += df.loc[df.molecule == reactant].time.values
        for product in products:
            df_reaction.energy += df.loc[df.molecule == product].energy.values
            df_reaction.one_electron_energy -= df.loc[df.molecule == product].one_electron_energy.values
            df_reaction.two_electron_energy -= df.loc[df.molecule == product].two_electron_energy.values
            df_reaction.correlation_energy -= df.loc[df.molecule == product].correlation_energy.values
            df_reaction.time += df.loc[df.molecule == product].time.values

        # subtract uCCSD reference
        #df_reaction['error'] = df_reaction.energy - df_reaction.loc[df_reaction.functional == 'uCCSD'].energy.values[0]

        # Add the reaction class, charge and spin
        df_reaction['reaction_class'] = rtype.strip()
        df_reaction['charge'] = charge.strip()
        df_reaction['spin'] = spin.strip()

        ## Add everything but the uCCSD method to the dataframe
        #dfr = dfr.append(df_reaction[df_reaction.functional != 'uCCSD'], ignore_index = True)
        # Add everything to the dataframe
        dfr = dfr.append(df_reaction, ignore_index = True, sort = True)
        print(reaction_name, df_reaction[df_reaction.functional == 'uCCSD'].energy.values[0], float(reference), df_reaction[df_reaction.functional == 'uCCSD'].energy.values[0] - float(reference))
    print(dfr.head())
    print(dfr[dfr.functional == 'uCCSD'].energy.values - np.asarray(reference_energies))
    quit()

    return dfr

def make_pickles(data_set_name, data_set_path = "../../portfolio_datasets/", pickle_path = "../pickles/"):
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
        print_missing(mol_df, data_set_name)
        mol_df.to_pickle(mol_df_name)
    return

    # Try to read the reaction pickle, else make it.
    try:
        reac_df = pd.read_pickle(reac_df_name)
    except FileNotFoundError:
        reac_df = parse_reactions(data_set_name + "_reactions", mol_df)
        quit()
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
                time = np.median(gga_df.time.values)
                df.at[gga_df.index, "time"] = time
                time = np.median(hybrid_df.time.values)
                df.at[hybrid_df.index, "time"] = time
                time = np.median(meta_df.time.values)
                df.at[meta_df.index, "time"] = time
                time = np.median(hybrid_meta_df.time.values)
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
                time = sub_df.loc[(sub_df.reaction == reaction_name)].time.values[0]
                df.at[sub_df.index, "time"] = time
    return df

def main():
    """
    Create all the reaction pickles
    """
    abde12_reac = make_pickles("abde12")
    nhtbh38_reac = make_pickles("nhtbh38")
    htbh38_reac = make_pickles("htbh38")
    quit()

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
