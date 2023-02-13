# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:34:41 2022

@author: Rotman A. Criollo Manajarrez
"""


########################################################################################
##
##              CORRELATION MATRIX 
##
## Author: Rotman A. Criollo Manajarrez
########################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

### reading from the file
filename = 'RES_Sist_Geot_2023.02.09.xlsx'
path = 'F:/RCM/Divulgacion/Articulos/2020 RIGS artiiculo geotermia/'
data = pd.read_excel(path + filename, sheet_name='res')

#########  CORRELATION
C = data.corr()

#Filtered High Values > 0.5
filteredC_05_1 = C[((C >= 0.5) | (C <= -0.5))]
#Filtered Low Values < 0.5
filteredC_0_05 = C[((C > 0.25) & (C > -0.25))]

# Figure
fig = plt.figure(dpi=380)
ax = fig.add_subplot(111)

# Good color distribution: hsv, PuOr
# Diverging colors: 
    # ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    # 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
# Cyclic colors: ['twilight', 'twilight_shifted', 'hsv']

#Color Selection
cmapSel = 'twilight_shifted'
# Low matrix
CLowVal= ax.matshow(filteredC_0_05,cmap=cmapSel, vmin=-1, vmax=1, alpha =0.3,
                  aspect='equal', interpolation='none')
# High matrix
CHighVal= ax.matshow(filteredC_05_1,cmap=cmapSel, vmin=-1, vmax=1, alpha =0.9,
                  aspect='equal', interpolation='none')
#Figures characteristics
cbar = fig.colorbar(CHighVal, orientation='vertical', pad=0.015)
cbar.ax.tick_params(labelsize=6)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns,fontsize=6)
ax.set_yticklabels(data.columns,fontsize=6)
plt.show()

# Other Option
filteredDf = C[((C >= .5) | (C <= -.5)) & (C !=1.000)]
plt.figure(figsize=(30,10))
sn.heatmap(filteredDf, annot=True, cmap = cmapSel)
plt.show()

# ###### DESCRIPTION STATISTICAL VALUES
# desc = data.describe()
# desc.to_csv(path+'2023.02.10_statistics.dat')
# corr_matrix = data.corr()
# corr_matrix.to_csv(path+'2023.02.10_corr_matrix_ALL.dat')
## Crrelation list. No duplicates
### first element of sol series is the pair with the biggest correlation

# corr_matrix_ABS = dfres.corr().abs()
# #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
# sol = (corr_matrix_ABS.where(np.triu(np.ones(corr_matrix_ABS.shape), k=1).astype(bool))
#                   .stack()
#                   .sort_values(ascending=False))
# sol.to_csv(path+'2023.02.13_correlation_pairs.dat')

# Correlation with COP
# a = corr_matrix['COP'].sort_values(ascending=False)
# print(a)
