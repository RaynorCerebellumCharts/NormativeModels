#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:44:35 2024

@author: alicha
"""

import os
import pandas as pd
import pcntoolkit as ptk

os.chdir('/project/4290000.01/alicha/Cerebellar_Ataxia/transfer_model_char/')

root_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia'
mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
data_dir = os.path.join(root_dir,'raynor_data/')
proc_dir = os.path.join(root_dir,'transfer_model_char/vox_powell_01/')
extension = ''
interactive = 'auto' # put false if debugging or not want to collect results

# define stuff to prepare for batch prediction
python_path = '/home/preclineu/alicha/.conda/envs/venv11/bin/python'
log_path = os.path.join(proc_dir,'logs/')
job_name = 'blr_transfer'
memory = '30gb'
batch_size = 300
duration = '01:00:00'
alg = 'blr'
cluster = 'torque'
model_path = os.path.join(proc_dir,'Models/')
count_jobsdone = "True"
outputsuffix = '_transfer'
inputsuffix = 'estimate'
output_path = os.path.join(proc_dir, 'Transfer/') 

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

    
#take patient data
df_sca = pd.read_csv(os.path.join(data_dir,'dem_sca'+extension+'.csv'))
site_ids_sca =  sorted(set(df_sca['site'].to_list()))

#take healthy data for adaptation set
df_ad = pd.read_csv(os.path.join(data_dir,'dem_ad'+extension+'.csv'))
site_ids_ad =  sorted(set(df_ad['site'].to_list()))

#load resp and cov 
resp_file_sca = os.path.join(data_dir,'resp_sca'+extension+'.pkl')
resp_file_ad = os.path.join(data_dir,'resp_ad'+extension+'.pkl')

cov_file_sca = os.path.join(data_dir, 'cov_bspline_sca'+extension+'.txt')
cov_file_ad = os.path.join(data_dir, 'cov_bspline_ad'+extension+'.txt')

sitenum_file_sca = os.path.join(data_dir, 'sitenum_sca.txt') 
sitenum_file_ad = os.path.join(data_dir, 'sitenum_ad.txt') 


ptk.normative_parallel.execute_nm(proc_dir, python_path, 
        'Transfer_' + job_name, cov_file_ad, resp_file_ad, batch_size, memory, duration, func='transfer', 
        alg=alg, binary=True, trbefile=sitenum_file_ad, 
        model_type=method, random_intercept=random_intercept, 
        random_slope=random_slope, random_noise=random_noise, 
        hetero_noise=hetero_noise, savemodel='True', outputsuffix=outputsuffix,
        inputsuffix=inputsuffix, 
        n_samples='1000', inscaler=inscaler, outscaler=outscaler, 
        testcovfile_path=cov_file_sca, testrespfile_path=resp_file_sca,
        tsbefile = sitenum_file_sca, output_path=output_path,
        model_path=model_path, log_path=log_path,count_jobsdone=count_jobsdone,
        likelihood=likelihood, random_slope_sigma=random_slope_sigma,
        interactive = interactive)

