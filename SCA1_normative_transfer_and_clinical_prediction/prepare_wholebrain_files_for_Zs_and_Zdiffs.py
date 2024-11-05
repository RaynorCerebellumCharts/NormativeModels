#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:24:03 2024

@author: alicha
"""

import os
import pandas as pd
import numpy as np
from pcntoolkit.util.utils import create_design_matrix
from pcntoolkit import dataio
import glob

root_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia'
transfer_dir = 'transfer_model_char'
data_dir = os.path.join(root_dir,'raynor_data/')
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
nii_suffix = 'nonlin_jac.nii.gz'
proc_dir = os.path.join(data_dir,'subjects_data')

extension = '' #session name, e.g. 'ses-01'
option = ''# leave empty to prepare files for normal transfer, or use '_cheat' to prepare files for Z-diffs extraction with no HC site adaptation set

#%% Loading

#loading nifiti filepaths 
jac_paths = sorted(glob.glob(os.path.join(proc_dir,'*',extension,'anat','*'+'.anat', '*'+ nii_suffix))) 

#add exclusions if needed
excluded_subjects = []
    
    
# Filter the list to exclude paths containing any of the specified strings
jac_paths = [path for path in jac_paths if not any(exclude in path for exclude in excluded_subjects)]

#loading demographics
df_dem = pd.read_csv(os.path.join(data_dir,'SCA1_50s_covariates.csv'))

# delete rows with bad quality subjects
df_dem = df_dem[~df_dem['subject'].isin(excluded_subjects)]
df_dem = df_dem.reset_index(drop=True)

# split between sca set and adaptation set           
df_sca = df_dem.loc[df_dem['SCA1'] == 1]
df_HC =  df_dem.loc[df_dem['SCA1'] == 0]          

df_dem.to_csv(os.path.join(data_dir,'dem_all_'+extension+'.csv'), index = False)           
df_sca.to_csv(os.path.join(data_dir,'dem_sca_'+extension+'.csv'), index = False)
df_HC.to_csv(os.path.join(data_dir,'dem_HC_'+extension+'.csv'), index = False)

#need to load covariates of the training set of the model we'll use after
#because test and adapt covariate matrices need to have same number of columns as train
df_tr = pd.read_csv(os.path.join(root_dir, transfer_dir,'dem_tr_01.csv'))
site_ids_tr = df_tr['site'].to_list()
site_ids_tr =  sorted(set(df_tr['site'].to_list()))

#%% design matrix parameters

xmin = 5 # boundaries for ages of participants +/- 5
xmax = 100

if extension =='ses-01':
    cols_cov = ['age_baseline','sex']
elif extension =='ses-02':
    cols_cov = ['age_Y1','sex']
elif extension == 'ses-03':
    cols_cov = ['age_Y2','sex']
    
    
site_ids =  sorted(set(df_dem['site'].to_list()))
print('configuring covariates ...')

#cheating for site because we don't have enough controls for adaptation set

if option == '_cheat':
    df_sca.loc[:,'site'] = 'ukb-11025.0'
    df_sca.loc[:,'sitenum'] = 4
    df_HC.loc[:,'site'] = 'ukb-11025.0'
    df_HC.loc[:,'sitenum'] = 4
    
X_sca = create_design_matrix(df_sca[cols_cov], site_ids = df_sca['site'], all_sites = site_ids_tr,
                            basis = 'bspline', xmin = xmin, xmax = xmax)
X_HC = create_design_matrix(df_HC[cols_cov], site_ids = df_HC['site'], all_sites=site_ids_tr,
                            basis = 'bspline', xmin = xmin, xmax = xmax)

cov_file_sca = os.path.join(data_dir, 'cov_bspline_cheat_sca_'+extension+'.txt')
cov_file_HC = os.path.join(data_dir, 'cov_bspline_cheat_HC_'+extension+'.txt')
dataio.fileio.save(X_sca, cov_file_sca)
dataio.fileio.save(X_HC, cov_file_HC)

# save the site ids for the test data 
site_file_sca = os.path.join(data_dir, 'site_sca_cheat_'+ extension+'.txt')
site_sca = df_sca['site'].to_numpy()
np.savetxt(site_file_sca, site_sca,fmt='%s')

sitenum_file_sca = os.path.join(data_dir, 'sitenum_sca_cheat_'+ extension+'.txt')
site_num_sca = df_sca['sitenum'].to_numpy(dtype=int)
np.savetxt(sitenum_file_sca, site_num_sca)

# save the site ids for the adaptation data
site_file_HC = os.path.join(data_dir, 'site_HC_cheat_'+ extension+'.txt')
site_HC = df_HC['site'].to_numpy()
np.savetxt(site_file_HC, site_HC,fmt='%s')


sitenum_file_HC = os.path.join(data_dir, 'sitenum_HC_cheat_'+ extension+'.txt') 
site_num_HC = df_HC['sitenum'].to_numpy(dtype=int)
np.savetxt(sitenum_file_HC, site_num_HC)


#%%  extract voxelwise data from wholebrain jacobian niftis preprocessed with standard fsl
print('loading wholebrain response data ...') 
    
x = []

for sub in range(len(jac_paths)):
    flat_sub = dataio.fileio.load(jac_paths[sub], mask=mask_nii, vol=False).T
    x.append(flat_sub)

# Ensure that the lengths are the same
if len(x) != len(df_dem):
    raise ValueError("Length of x and df_dem do not match!")
                     
# Convert x into a DataFrame with same index as df_dem
df_x = pd.DataFrame(x, index=df_dem.index)
df_x_sca = df_x.loc[df_sca.index]
df_x_HC = df_x.loc[df_HC.index]

# and write out as pkl
resp_file_sca = os.path.join(data_dir,'resp_sca_cheat_'+extension + option +'.pkl')
resp_file_HC = os.path.join(data_dir,'resp_HC_cheat_'+extension+ option +'.pkl')
dataio.fileio.save(df_x_sca.values, resp_file_sca)
dataio.fileio.save(df_x_HC.values, resp_file_HC)

