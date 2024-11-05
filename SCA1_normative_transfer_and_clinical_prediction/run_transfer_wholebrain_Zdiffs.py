#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:44:35 2024

@author: alicha
"""

import os
import numpy as np
import pandas as pd
import itertools
import xarray as xr
import pcntoolkit as ptk


os.chdir('/project/4290000.01/alicha/Cerebellar_Ataxia/transfer_model_char/')
root_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia'
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
data_dir = os.path.join(root_dir,'raynor_data/')
proc_dir = os.path.join(root_dir,'transfer_model_char/vox_powell_01/')
transfer_dir = 'transfer_model_char'
interactive = 'auto' # put false if debugging or if we do not want to collect results

## define stuff to prepare for batch prediction
python_path = '/home/preclineu/alicha/.conda/envs/venv11/bin/python'
log_path = os.path.join(proc_dir,'logs/')
job_name = 'blr_predict'
memory = '30gb'
batch_size = 300
duration = '01:00:00'
alg = 'blr'
normative_path = '/home/preclineu/alicha/.conda/envs/venv11/lib/python3.11/site-packages/pcntoolkit/normative.py'
count_jobsdone = "True"
outputsuffix = '_predict'
inputsuffix = 'estimate'
output_path = os.path.join(proc_dir, 'Predict_cheat') 

 # Set default configuration
inscaler = 'None' 
outscaler = 'None'
method = 'linear' #'bspline' # 'linear' 'polynomial' 
likelihood = 'Normal'        
random_intercept = 'True'
random_slope = 'True'
random_noise = 'True'
random_slope_sigma = 'False'      
hetero_noise = 'False'

sessions = ['ses-01','ses-02','ses-03']

#%% Cross-sectional transfer/predict



for extension in sessions : # transfer for all three sessions separately

    #take dem data
    df_sca = pd.read_csv(os.path.join(data_dir,'dem_sca_'+extension+'.csv'))
    df_HC = pd.read_csv(os.path.join(data_dir,'dem_HC_'+extension+'.csv'))
    
    #dummy sites
    site_ids_sca = pd.read_csv(os.path.join(data_dir,'sitenum_sca_cheat_'+extension+'.txt'))
    site_ids_HC = pd.read_csv(os.path.join(data_dir,'sitenum_HC_cheat_'+extension+'.txt'))
    
    #load resp and cov
    resp_file_sca = os.path.join(data_dir,'resp_sca_cheat_'+extension+'.pkl')
    resp_file_HC = os.path.join(data_dir,'resp_HC_cheat_'+extension+'.pkl')
    

    cov_file_sca = os.path.join(data_dir, 'cov_bspline_cheat_sca_'+extension+'.txt')
    cov_file_HC = os.path.join(data_dir, 'cov_bspline_cheat_HC_'+extension+'.txt')
    
    sitenum_file_sca = os.path.join(data_dir, 'sitenum_sca_cheat_'+ extension+'.txt') 
    sitenum_file_HC = os.path.join(data_dir, 'sitenum_HC_cheat_'+ extension+'.txt') 
    
    
    #transfer model for that session on patients - it's actually a predict as there is no adaptation data
    ptk.normative_parallel.execute_nm(processing_dir = proc_dir, 
                                      python_path = python_path, 
                                      job_name='Predict_'+extension + '_' + job_name, 
                                      covfile_path = cov_file_sca,
                                      respfile_path = resp_file_sca, 
                                      batch_size = batch_size, 
                                      memory =  memory, 
                                      duration =duration, 
                                      func='predict', 
                                      alg=alg, 
                                      binary=True, 
                                      model_type=method, 
                                      random_intercept=random_intercept, 
                                      random_slope=random_slope, 
                                      random_noise=random_noise, 
                                      hetero_noise=hetero_noise, 
                                      savemodel='True', 
                                      outputsuffix = outputsuffix + 'cheatSCA' + extension, #no underscore because then collect_nm doesn't find it
                                      inputsuffix=inputsuffix, 
                                      n_samples='1000', 
                                      inscaler=inscaler, 
                                      outscaler=outscaler, 
                                      output_path=output_path+ '_cheat_SCA_' + extension + '/',
                                      log_path=log_path,
                                      count_jobsdone=count_jobsdone,
                                      likelihood=likelihood, 
                                      random_slope_sigma=random_slope_sigma,
                                      return_y = True,
                                      interactive = interactive)

    
    #same on controls, that we will use to estimate variance for Z_diffs
    ptk.normative_parallel.execute_nm(processing_dir = proc_dir, 
                                      python_path = python_path, 
                                      job_name='Predict_'+extension + '_' + job_name, 
                                      covfile_path = cov_file_HC,
                                      respfile_path = resp_file_HC, 
                                      batch_size = batch_size, 
                                      memory =  memory, 
                                      duration =duration, 
                                      func='predict', 
                                      alg=alg, 
                                      binary=True, 
                                      model_type=method, 
                                      random_intercept=random_intercept, 
                                      random_slope=random_slope, 
                                      random_noise=random_noise, 
                                      hetero_noise=hetero_noise, 
                                      savemodel='True', 
                                      outputsuffix = outputsuffix + 'cheatHC' + extension, #no underscore because then collect_nm doesn't find it
                                      inputsuffix=inputsuffix, 
                                      n_samples='1000', 
                                      inscaler=inscaler, 
                                      outscaler=outscaler, 
                                      output_path=output_path+ '_cheat_HC_' + extension + '/',
                                      log_path=log_path,
                                      count_jobsdone=count_jobsdone,
                                      likelihood=likelihood, 
                                      random_slope_sigma=random_slope_sigma,
                                      return_y = True,
                                      interactive = interactive)
    
#%% Z-diffs for all pairs of sessions

session_pairs = list(itertools.combinations(sessions, 2))


for pair in session_pairs : 
    
    v1 = pair[0]
    v2 = pair[1]
    
    
    df_sca_v1 = pd.read_csv(os.path.join(data_dir,'dem_sca_'+v1+'.csv'))
    df_HC_v1 = pd.read_csv(os.path.join(data_dir,'dem_HC_'+v1+'.csv'))

    df_sca_v2 = pd.read_csv(os.path.join(data_dir,'dem_sca_'+v2+'.csv'))
    df_HC_v2 = pd.read_csv(os.path.join(data_dir,'dem_HC_'+v2+'.csv'))
    
    #take overlapping subject indices
    common_df_sca_v1 = df_sca_v1[df_sca_v1['subject'].isin(df_sca_v2['subject'])].index
    common_df_HC_v1= df_HC_v1[df_HC_v1['subject'].isin(df_HC_v2['subject'])].index
    
    common_df_sca_v2 = df_sca_v2[df_sca_v2['subject'].isin(df_sca_v1['subject'])].index
    common_df_HC_v2 = df_HC_v2[df_HC_v2['subject'].isin(df_HC_v1['subject'])].index  
    
    # take measures of interest at both timepoints for overlapping subjects    
    df_sca_v1 = df_sca_v1.iloc[common_df_sca_v1].reset_index(drop=True)
    df_HC_v1 = df_HC_v1.iloc[common_df_HC_v1].reset_index(drop=True) 
    df_sca_v2 = df_sca_v2.iloc[common_df_sca_v2].reset_index(drop=True)
    df_HC_v2 = df_sca_v2.iloc[common_df_HC_v2].reset_index(drop=True)
    
    

    yhat_sca_v1 = pd.read_pickle(os.path.join(proc_dir, 'yhat'+ outputsuffix + 'cheatSCA' +v1+'.pkl'))
    yhat_sca_v1 = yhat_sca_v1.iloc[common_df_sca_v1].reset_index(drop=True)
    
    yhat_HC_v1 = pd.read_pickle(os.path.join(proc_dir, 'yhat'+outputsuffix + 'cheatHC' +v1+'.pkl'))
    yhat_HC_v1 = yhat_HC_v1.iloc[common_df_HC_v1].reset_index(drop=True)
    
    yhat_sca_v2 = pd.read_pickle(os.path.join(proc_dir, 'yhat'+outputsuffix + 'cheatSCA' +v2+'.pkl'))
    yhat_sca_v2 = yhat_sca_v2.iloc[common_df_sca_v2].reset_index(drop=True)
    
    yhat_HC_v2= pd.read_pickle(os.path.join(proc_dir, 'yhat'+outputsuffix + 'cheatHC' +v2+'.pkl'))
    yhat_HC_v2 = yhat_HC_v2.iloc[common_df_HC_v2].reset_index(drop=True)

    
    Y_sca_v1 = pd.read_pickle(os.path.join(proc_dir, 'Y'+ outputsuffix + 'cheatSCA' +v1+'.pkl'))
    Y_sca_v1 = Y_sca_v1.iloc[common_df_sca_v1].reset_index(drop=True)
    
    Y_HC_v1 = pd.read_pickle(os.path.join(proc_dir, 'Y'+ outputsuffix + 'cheatHC' +v1+'.pkl'))
    Y_HC_v1 = Y_HC_v1.iloc[common_df_HC_v1].reset_index(drop=True)
    
    Y_sca_v2 = pd.read_pickle(os.path.join(proc_dir, 'Y'+ outputsuffix + 'cheatSCA' +v2+'.pkl'))
    Y_sca_v2 = Y_sca_v2.iloc[common_df_sca_v2].reset_index(drop=True)
    
    Y_HC_v2= pd.read_pickle(os.path.join(proc_dir, 'Y'+ outputsuffix + 'cheatHC' +v2+'.pkl'))
    Y_HC_v2 = Y_HC_v2.iloc[common_df_HC_v2].reset_index(drop=True)
    
    
    #wrap neuroimaging measures
    xr_sca_v1 = xr.concat([
                xr.DataArray(yhat_sca_v1, dims=['subject', 'voxels'], name='yhat'),
                xr.DataArray(Y_sca_v1, dims=['subject', 'voxels'], name='Y'),
                ],
                pd.Index(['yhat','Y'], name = 'features'))
    
    xr_sca_v2 = xr.concat([
                xr.DataArray(yhat_sca_v2, dims=['subject', 'voxels'], name='yhat'),
                xr.DataArray(Y_sca_v2, dims=['subject', 'voxels'], name='Y'),
                ],
                pd.Index(['yhat','Y'], name = 'features'))
    
    
    xr_HC_v1 = xr.concat([
                xr.DataArray(yhat_HC_v1, dims=['subject', 'voxels'], name='yhat'),
                xr.DataArray(Y_HC_v1, dims=['subject', 'voxels'], name='Y'),
                ],
                pd.Index(['yhat','Y'], name = 'features'))
    
    xr_HC_v2 = xr.concat([
                xr.DataArray(yhat_HC_v2, dims=['subject', 'voxels'], name='yhat'),
                xr.DataArray(Y_HC_v2, dims=['subject', 'voxels'], name='Y'),
                ],
                pd.Index(['yhat','Y'], name = 'features'))
    
    
    xr_sca = xr.concat([xr_sca_v1,xr_sca_v2],  pd.Index(['V1', 'V2'], name = 'visit'))
    
    xr_HC = xr.concat([xr_HC_v1,xr_HC_v2],  pd.Index(['V1', 'V2'], name = 'visit'))
    
    
    # 1) Estimate the variance in healthy controls
    HC_sqrt = np.sqrt(
        pd.DataFrame(
    (
            (xr_HC.sel(visit='V2', features='Y')- xr_HC.sel(visit='V2', features='yhat'))
            -
            (xr_HC.sel(visit='V1', features='Y') - xr_HC.sel(visit='V1', features='yhat'))
    
    ).var(axis=0).to_pandas()
    ).T
    )
    
    
    # 2) Substract the two patient visits
    sca_diff =((xr_sca.sel(visit='V2', features='Y') - xr_sca.sel(visit='V2', features='yhat'))
                    -
                (xr_sca.sel(visit='V1', features='Y') - xr_sca.sel(visit='V1', features='yhat'))
                ).to_pandas()
    
    # 3) Compute the zdiff score
    sca_zdiff = sca_diff.div(HC_sqrt.squeeze(), axis='columns')
    
    pd.to_pickle(sca_zdiff, os.path.join(proc_dir, 'Z_diffs_SCA_cheat_'+ v1 + '_' + v2 +'.pkl'))


   


