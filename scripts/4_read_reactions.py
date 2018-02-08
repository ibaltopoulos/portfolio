# imports
import glob
import sys
import os
from os.path import basename
from pprint import pprint
import numpy

# Script to read in a text file of reactions, then calculate the reaction energy. Requires e.g. HTBH_Molpro_tabulated absolute energy data
# , and creates HTBH_Molpro_tabulated_reactionenergies with rows of the form "H,N2O->TS1 6-31g LSDC -0.023526".
# The new tabulated data can be read by the analysis programs in the same way as the absolute energy file.

#################################################
def tryFloat(testFloat):
    """Return float if can, otherwise None"""
    try:
        return float(testFloat)
    except:
        return None

# Subroutine to get energy from reactant, basis and method
def getEnergy(rctant, bsis, method):
    index = identifiers.index([(rctant + ' ' + bsis), method])
    return energies[index]
#################################################

# Read in reactions from file (second command line argument)
reactions_filename = sys.argv[2]
reactions_file = open(reactions_filename)
reactions_whole = reactions_file.readlines()
reactions_split =  [line.split() for line in reactions_whole]
nreactions = (len(reactions_split)+1)/3 # Number of reactions

# Create and occupy reactants and products lists. These are lists of lists of reactants/products
reactants = []
products = []
for i in range(nreactions):
    reactants.append(reactions_split[3*i])
    products.append(reactions_split[3*i+1])

# Read in database of energies from file
try:
    data_name = sys.argv[1]
except:
    print('Please pass datafile name.')
    exit()

try:
    datafile = open(data_name)
except:
    print('Data cannot be opened.')
    exit()

# Read raw data
total_data = []
for line in datafile.readlines():
    total_data.append(line.split())

# Set up arrays, with molecule/basis set together
identifiers = [[(datapoint[0]+' '+datapoint[1]),datapoint[2]] for datapoint in total_data]
energies = [tryFloat(datapoint[3]) for datapoint in total_data]
mols_bases = list(set([identifiers[i][0] for i in range(len(identifiers))]))
methods = list(set([identifiers[i][1] for i in range(len(identifiers))]))
mols_bases = sorted(mols_bases)

# Each basis
bases = list(set([mols_bases[i].split()[1] for i in range(len(mols_bases))]))

myfile = open('%s_reactionenergies'%data_name,'w') # New file for reaction energies
# Loop through reactions specified in the file
for reaction in range(nreactions):
    # Loop through basis sets
    for basis in bases:
        # Loop through methods (DFT functionals etc.)
        for method_in in range(len(methods)):
            # Try to read energies of all reactants/products and write line to new file
            try:
                # tempenergy is the reaction energy
                tempenergy = 0.0
                for reactant in reactants[reaction]:
                   tempenergy = tempenergy - getEnergy(reactant,basis,methods[method_in])
                for product in products[reaction]:
                    tempenergy = tempenergy + getEnergy(product,basis,methods[method_in])
                myfile.write('%s->%s %s %s %f\n' %(','.join(reactants[reaction]), ','.join(products[reaction]), basis, methods[method_in], tempenergy))
            # If doesn't work, write nan (e.g. if a value isn't specified/there has been an error)
            except:
                myfile.write('%s->%s %s %s nan\n' %(','.join(reactants[reaction]), ','.join(products[reaction]), basis, methods[method_in]))
