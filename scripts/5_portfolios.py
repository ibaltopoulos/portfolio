# imports
import glob
import sys
import time
import os
from os.path import basename
from pprint import pprint
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
import portfolioopt as pfopt
import pandas as pd

# User should run the script from command line as python mining.py my_tabulated_data (my_input_portfolio).
# The optimal portfolio is determined using Markowitz portfolio theory if one is not provided.

#################################################

# Parameters
kcalMol_units = True # Otherwise gives results output in Hartree
kcalMol_per_Hartree = 627.503
benchmark_method = 'rccsdt avtz'

# Parameters ONLY relevant if a portfolio is not user-supplied
basis_sets = {'svp','6-31gXX','sto-3g'}
functional_zoo = {'M06-HF','M08-SO','M11-L','UHF','VS99'}
remove_high_error_functionals = False # Removes method from portfolio if error over some limit for ANY reaction
remove_high_error_functionals_limit = 0.5 # Error over which removed, in Hartree
shorting_allowed = False # Shorting allowed in portfolio construction?

# Set up appropriate strings for units, and conversion factor from Molpro energies
if kcalMol_units == True:
    unit = 'kcal mol-1'
    unit_conversion_factor = kcalMol_per_Hartree
else:
    unit = 'Hartree'
    unit_conversion_factor = 1.0

#################################################

def optimal_portfolio(returns,desired):
    """Returns optimal portfolio for given return. Takes in array of returns and desired portfolio mean."""
    # Matrix of means and covariance
    means = np.asmatrix(np.mean(returns,axis=1))
    cov_mat = np.asmatrix(np.cov(returns))
    # Convert to pandas data frames for use by PortfolioOpt library
    means_pd = pd.Series(np.mean(returns,axis=1))
    cov_mat_pd = pd.DataFrame(cov_mat)
    # Deduce optimal portfolio, with or without shorting allowed
    opt_w = pfopt.markowitz_portfolio(cov_mat_pd, means_pd, desired, allow_short = shorting_allowed)
    opt_mean = np.matrix(opt_w) * means.T
    # Return optimal portfolio and mean of this portfolio (should be approximately zero)
    # Portfolio in numpy matrix form, converted back from pandas
    return np.asmatrix(opt_w), opt_mean[(0,0)]

#################################################

def getEnergy(mol,method_bas):
    """Subroutine to get energy from molecule and method_bas."""
    index = identifiers.index([mol, method_bas])
    return energies[index]

#################################################

def tryFloat(testFloat):
    """Return float if can, otherwise None."""
    try:
        return float(testFloat)
    except:
        return None

#################################################

# Determine whether portfolio is provided by user, based on number of command line arguments
if (len(sys.argv) == 2):
    print ('\nPortfolio will be created...\n')
    createPortfolio = True
elif (len(sys.argv) == 3):
    print ('\nPortfolio will be read from file...\n')
    createPortfolio = False

# Read in tabulated data
try:
    data_name = sys.argv[1]
    datafile = open(data_name)
    total_data = []
    for line in datafile.readlines():
        total_data.append(line.split())
except:
    print('Tabulated data cannot be opened.')
    exit()

# Set up arrays of the available data
identifiers = [[datapoint[0],datapoint[2]+' '+datapoint[1]] for datapoint in total_data] #['molecule', 'method_basis']
energies = [tryFloat(datapoint[3]) for datapoint in total_data]
mols = list(set([identifiers[i][0] for i in range(len(identifiers))]))
# Now to get a certain energy, getEnergy('molecule','method basis')
# NOTE that here 'molecule' can be a reaction, depending on what is input to program

# Read in portfolio if provided
portfolio_name =  data_name + '_portfolio' # Initially set portfolio name to this
if createPortfolio:
    # Create all possible combinations of basis set and functional from the lists provided
    # in script parameters, for optimization.
    portfolio_methods_bases = sorted([(functional+' '+basis) for basis in basis_sets for functional in functional_zoo])
else:
    # When a portfolio is provided, we take the method/basis set combinations from it instead.
    try:
        portfolio_name = sys.argv[2] # If portfolio provided, change portfolio name to this
        portfolio_file = open(portfolio_name)
        portfolio_data = []
        for line in portfolio_file.readlines():
            portfolio_data.append(line.split())
        portfolio_methods_bases = [(i[0] + ' ' + i[1]) for i in portfolio_data]
        portfolio_weights = np.asmatrix([float(i[2]) for i in portfolio_data])
    except:
        print('Problem opening or reading portfolio file.')
        exit()

# Possible reference methods added to methods
portfolio_methods_bases.append('rccsdt avtz')
portfolio_methods_bases.append('uCCSDT avtz')

#################################################

# Put data into a 2D numpy array with molecule (or rxn) as one dimension and method (method & basis) as the other
tabulated_data = np.zeros([len(mols),len(portfolio_methods_bases)])
for mol_in in range(len(mols)):
    for method_basis_in in range(len(portfolio_methods_bases)):
        tabulated_data[mol_in,method_basis_in] = getEnergy(mols[mol_in],portfolio_methods_bases[method_basis_in])

# Create tabulated data for differences to rccsdt avtz results.
ccsd_index = portfolio_methods_bases.index('rccsdt avtz')
differences = np.zeros([len(mols),len(portfolio_methods_bases)])
for mol_in in range(len(mols)):
    for method_basis_in in range(len(portfolio_methods_bases)):
        differences[mol_in,method_basis_in] = (tabulated_data[mol_in,method_basis_in] - tabulated_data[mol_in,ccsd_index])

# Remove rccsdt and uccsdt columns from differences array.
mymask = np.ones(len(portfolio_methods_bases),dtype=bool)
for method_basis_in in range(len(portfolio_methods_bases)):
    if (portfolio_methods_bases[method_basis_in].split()[0] == 'rccsdt') or (portfolio_methods_bases[method_basis_in].split()[0] == 'uCCSDT'):
        mymask[method_basis_in] = False
portfolio_methods_bases = list(compress(portfolio_methods_bases, mymask))
differences = differences[:,mymask]

# Remove molecules if any functional gives a NaN difference to ccsdt
mask = [~np.isnan(np.min(differences[dif,:])) for dif in range(len(differences[:,1]))]
differences = differences[mask]
mols = list(compress(mols,mask))

if createPortfolio:
    # Check for duplicate rows and columns and if so remove, so portfolio optimization can be performed
    # Molecules/basis sets
    mask1 = np.ones(len(differences[:,1]),dtype=bool)
    for i in range(len(differences[:,1])-1):
        for j in range(i+1,len(differences[:,1])):
           if np.all(differences[i,:] == differences[j,:]):
               mask1[j] = False
    # Methods
    mask2 = np.ones(len(differences[1,:]), dtype=bool)
    for i in range(len(differences[1, :]) - 1):
      for j in range(i + 1, len(differences[1, :])):
           if np.all(differences[:,i] == differences[:,j]):
             mask2[j] = False
    # Remove functionals from optimization if any error is over a limit for ANY molecule
    if remove_high_error_functionals:
        for i in range(len(differences[1,:])):
            if np.any(np.abs(differences[:,i]) > remove_high_error_functionals_limit):
                mask2[i] = False
                print ('Removed for error: %s' %methods[i])
    differences = differences[mask1]
    differences = differences[:,mask2]
    mols = list(compress(mols,mask1))
    portfolio_methods_bases = list(compress(portfolio_methods_bases,mask2))

#################################################

nfunctionals = len(differences[1,:])
if createPortfolio:
    # If portfolio optimization is going to happen, and there are more methods than data
    # points to fit to, then reduce number of functionals and warn the user.
    if (nfunctionals > len(differences[:,1])):
        nfunctionals = len(differences[:,1])
        print ("WARNING: Number of functionals reduced to number of data points.")
    returns = differences[:,:nfunctionals].T #returns = differences[:,:10].T
    portfolio_weights, opt_mean = optimal_portfolio(returns,0.0)

# Error of portfolio calculated using portfolio weights and differences matrix
portfolio_outcomes = differences[:,:nfunctionals]*portfolio_weights.T #portfolio_outcomes = differences[:,:10]*portfolio.T

# Unit conversion
portfolio_outcomes = portfolio_outcomes * unit_conversion_factor
differences = differences * unit_conversion_factor

#################################################
#
# Now we have an array 'differences', the number of molecules by the number of methods
# (which are functional-basis set combinations). It is a matrix of errors. We also have an array
# 'portfolio_weights' which is either the user-input portfolio or one constructed using
# Markowitz portfolio theory. The array 'portfolio_outcomes' gives the error of the portfolio
# for the different molecules.
#
# The rest of the script just plots and prints info from these.
#
#################################################

# Plot each individual method errors and portfolio errors
for i in range(len(differences[1,:nfunctionals])): #differences[1,:10])
    plt.plot(range(len(mols)),differences[:,i],label=portfolio_methods_bases[i].replace('X','*'))
plt.plot(range(len(mols)),portfolio_outcomes,'b--',label='Portfolio',linewidth=2.0)
plt.legend(prop={'size':6},ncol=4,loc='lower left')
#plt.title(('Data from %s with portfolio %s, %d data points.'%(data_name,portfolio_name,len(portfolio_outcomes))).replace('_Molpro_tabulated',''))
plt.xlabel('Barrier')
plt.ylabel('Error / %s'%(unit))

# Write portfolio to file
portfolio_file = open(portfolio_name,'w')
for i in range(len(portfolio_methods_bases[:nfunctionals])): #methods[:10]
    portfolio_file.write('%s %f\n' %(portfolio_methods_bases[i],portfolio_weights[(0,i)]))
portfolio_file.close()

# Write mean absolute errors to file
means_file = open((data_name+'_means'),'w')
for i in range(len(portfolio_methods_bases[:nfunctionals])): #methods[:10]
    means_file.write('%s %f\n' %(portfolio_methods_bases[i],np.mean([np.abs(j) for j in differences[:,i]])))
means_file.write('%s %f\n' % (portfolio_name,np.mean([np.abs(j) for j in portfolio_outcomes])))
means_file.close()

# Write RMS error to file
methods_RMSs = []
RMSs_file = open((data_name+'_RMSs'),'w')
for i in range(len(portfolio_methods_bases[:nfunctionals])): #methods[:10]
    methods_RMSs.append(np.sqrt(np.sum([j**2 for j in differences[:,i]])/len(differences[:,1])))
    RMSs_file.write('%s %f\n' %(portfolio_methods_bases[i],methods_RMSs[i]))
portfolio_RMS = np.sqrt(np.sum([j**2 for j in portfolio_outcomes])/len(differences[:,1]))
RMSs_file.write('%s %f\n' % (portfolio_name,portfolio_RMS))
RMSs_file.close()

# Work out minimum RMS individual method and write out
minError = min(methods_RMSs)
minIndex = methods_RMSs.index(minError)
#plt.text(12,18,'Portfolio RMS error:\n%f\nBest individual method (%s):\n%f'%(portfolio_RMS,portfolio_methods_bases[minIndex].replace('X','*'),minError),backgroundcolor = 'w',fontsize=6)
#plt.text(12,30,'Training Data Set',weight='bold',backgroundcolor = 'w',fontsize=10)
print ('Portfolio RMS error: %f %s\nLowest individual method RMS error (%s): %f %s\n'%(portfolio_RMS,unit,portfolio_methods_bases[minIndex].replace('X','*'),minError,unit))

# Plotting
plt.xticks(range(len(mols)),mols,rotation='vertical',fontsize=7)
plt.savefig((data_name+'.png'),dpi=200,bbox_inches='tight')
plt.show()

