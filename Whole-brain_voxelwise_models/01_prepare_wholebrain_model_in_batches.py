#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:39:47 2023 edited 2025

@author: alicha
"""
from pcntoolkit import NormData, dataio

import numpy as np
import pandas as pd
import os
import dill
import joblib
from joblib import Parallel, delayed
import nibabel as nib
from nibabel import processing
import nilearn as nil
from nilearn import image

#%% Global settings
modality = 'log_jacs' #e.g. 'log_jacs', 'mod_gmv', 'mod_wmv'
extension = '_wholebrain_2mm_01'#naming purposes
root_dir = '/path/to/root/dir/'

#where the new models will live
base_dir =os.path.join(root_dir,'models/') 
proc_dir = os.path.join(base_dir,'BLR_aff_nonlin_' + modality + extension +'/')
w_dir = os.path.join(proc_dir,'models/')
os.makedirs(proc_dir, exist_ok=True)
os.makedirs(w_dir, exist_ok=True)

if modality == 'log_jacs':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
    smooth = False
if modality == 'mod_gmv':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg.nii.gz')
    smooth = True
if modality == 'mod_wmv':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg.nii.gz')
    smooth = True

cohorts = ['dataset1',
           'dataset2',
           'dataset3']

data_dir = '/path/to/data/dir/'


#details for brain data extraction from niftis
resample = True #True if not 1mmm
vox_size = [2, 2, 2] #2mm
fwhm = 6 #used if smooth is true
batch_size = 150 #150 voxels per batch was a good balance to run with 4GB RAM on the Donders cluster
chunk_size = 500 #optional - chunks of subjects to load the niftis of, for option 1 (limited RAM) below

#%% Load covariates - These have to be in the same order as the cohorts variable
# Example snippets of code to pull demographic data from each dataset folder

print('loading covariate data ...')
dfs = []
ii = 0

dfs.append(pd.read_csv(os.path.join(data_dir,cohorts[ii],'participants.csv'))) 
#example cleanup
dfs[ii].rename(columns={'participant_id': 'sub_id','gender': 'sex', 'site_id':'site'}, inplace=True)
dfs[ii]['site'] = f'{cohorts[ii]}-'+dfs[ii]['site'].astype(str)#optional - rename as str rather than int or float to avoid potential downstream problems
dfs[ii] = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,cohorts[ii],'participants.txt'),sep="	"))
#example cleanup
dfs[ii]['sex'] = (dfs[ii]['sex'] == 'M').astype(int) #makes males into 1s and female 0, because True or False
dfs[ii] = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
dfs[ii].sort_index(inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,cohorts[ii],'participants.csv')))
#example cleanup
dfs[ii].rename(columns={'Unnamed: 0': 'sub_id'}, inplace=True)
dfs[ii]['site'] = cohorts[ii].astype(str)
dfs[ii].rename(columns={'Subject': 'sub_id','Sex': 'sex', 'Age':'age'}, inplace=True)
dfs[ii] = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1


#remove sites with less than 60 subjects (30 for train) to allow decent site effect modeling
for ii in range(len(dfs)):
    counts = dfs[ii]['site'].value_counts() 
    valid = counts[counts >= 60].index
    dfs[ii]= dfs[ii][dfs[ii]['site'].isin(valid)]
    print(dfs[ii]['site'].value_counts())
    dfs[ii].sort_index(inplace = True) #just in case some are not ordered  


#%% Filter covariates 
#excludes the QCed-out subjects based on T1 Warped files from ants (and, if relevant, the warped segmentation from fsl) 
# also exclude subjects that have no relevant (jacobian, GMV...) images
#optionally, also exclude those with previous anatomical QC exclusions, and those ones with a lot ofoutlier Zscores voxel values
 

print('filtering covariate data ... ')

if modality == 'log_jacs': # gmv and wmv probably have fully overlapping exclusions for absent scans but this is just in case
    list_name = 'have_jacs.txt'
elif modality == 'mod_gmv':
    list_name = 'have_gmv.txt'
elif modality == 'mod_wmv':
    list_name = 'have_wmv.txt'

for ii, cohort in enumerate(cohorts):
    print(cohort)
    
    excluded = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants','sub_exclude_ants_reg.txt'), header = None)
    
    #optional - if datasets already have anatomical exclusions based on raw T1 artefacts etc. 
    excluded_anat = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1','sub_exclude_anat.txt'), header = None)#add the anatomical exclusions
    
    #optional - only after a first estimation of the normative models have been run, if some subjects were not initally flagged by ants QC 
    #but still have a very large number of voxels with outlier Z-scores, can flag them with the normative model and exclude for next iterations
    #this is done in the 03_evaluate_wholebrain_model_in_batches.py script
    excluded_outlier = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants',f'sub_exclude_outliers_5PercentofVoxels_{modality}.txt'), header = None)
    

    excluded = pd.concat([excluded, excluded_anat, excluded_outlier])
    excluded_seg = pd.DataFrame()
        
    if modality in ['mod_gmv', 'mod_wmv']: #for these modalities exclude those from segmentation QC as well
        excluded_seg = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants','sub_exclude_seg.txt'), header = None)
        excluded = pd.concat([excluded, excluded_seg])
    
    #don't forget to make these into str in case some subjects ids are numbers
    excluded = excluded.astype('string')
    
    #exclude those with no relevant nifti files
    have_scans = pd.read_csv(os.path.join(data_dir,cohort,'qc', 'T1_ants', list_name), header = None)[0]
    have_scans = have_scans.astype(str).to_list()
    excluded_no_scans = pd.DataFrame(dfs[ii][~dfs[ii].index.isin(have_scans)].index)
    excluded_no_scans.columns=[0] #make it concatenable with the other one 
     
    excluded = pd.concat([excluded, excluded_no_scans],ignore_index = True)
    excluded = excluded.values.flatten().tolist()
    
    #just  to be sure
    excluded_seg = excluded_seg.astype('string')
    excluded_seg = excluded_seg.values.flatten().tolist()
    
    print('from', len(dfs[ii]), 'subjects, exclude', len(dfs[ii].loc[dfs[ii].index.isin(excluded)]))
    print('specifically, excluded for seg reasons', len(dfs[ii].loc[dfs[ii].index.isin(excluded_seg)]))
    print('retaining', len(dfs[ii])-len(dfs[ii].loc[dfs[ii].index.isin(excluded)]), '\n')

    dfs[ii] = dfs[ii].drop(excluded, errors='ignore')
    
# join all datasets
df_dem = pd.concat(dfs)
df_dem = df_dem.dropna() #remove rows that have Nan for anything

#make sure that sex is the same datatype everywhere otherwise that'll double the batch effect e.g. ints vs floats
df_dem.loc[:,'sex'] = df_dem.loc[:,'sex'].astype(int)

print(df_dem.shape)


#%%#grab nifti filepaths for sujects we have covariate data for

sub_paths = []

if modality == 'log_jacs':
    file_name = 'T1_BrainNorm_Jacobian.nii.gz'
elif modality == 'mod_gmv':
    file_name = 'T1_BrainNorm_GMV.nii.gz'
elif modality == 'mod_wmv':
    file_name = 'T1_BrainNorm_WMV.nii.gz'

for ii, cohort in enumerate(cohorts):
    f_pattern =os.path.join(data_dir, cohort,'subjects','{}','T1_ants_MNI152NLin2009cSym', file_name)
    dataset_subpaths = [f_pattern.format(sub) for sub in sorted(dfs[ii].index.tolist())]
    sub_paths.extend(dataset_subpaths)
    
# Ensure that the lengths are the same
if len(sub_paths) != len(df_dem):
    raise ValueError("Length of filepaths and df_dem do not match!")   
    
#%% Load the niftis to collect the response data

#grab total number of voxels from mask  
mask = dataio.fileio.load(mask_nii, mask=mask_nii, vol=False).T
n_voxels = len(mask)
n_batches = n_voxels // batch_size + int(n_voxels % batch_size != 0)
voxel_indices = list(range(n_voxels))

def load_file(subpath, mask_nii, resample, smooth, vox_size):
    
    img = nib.load(subpath)
    if resample : #2mm
        img = nib.processing.resample_to_output(img, vox_size)
    if smooth:
        img = nil.image.smooth_img(img,fwhm)
    data = img.get_fdata()
    data = dataio.fileio.vol2vec(data, nib.load(mask_nii).get_fdata())
    
    return data

#these will go into the model
covariates = ["age"]
batch_effects = ["sex", "site"]

#%%# option 1 - if not enough RAM, we have to load and save the nifti data for chunks of subjects before aggregating them in batches of voxels

# Initialize directories and empty files to store batch data
for b in range(n_batches):
    batch = f'batch_{b}'
    if not os.path.exists(os.path.join(w_dir,batch)):
        os.makedirs(os.path.join(w_dir,batch))
        os.makedirs(os.path.join(w_dir,batch,'subjects_chunks'))


# #load niftis from subject chunks,
for chunk_start in range(0, len(sub_paths), chunk_size):
    chunk_paths = sub_paths[chunk_start:chunk_start+chunk_size] 
    current_chunk_size = len(chunk_paths)#this accounts for the last chunk, which will have a smaller number of subjects
    print(f"Loading chunk: subjects {chunk_start} to {chunk_start + current_chunk_size}")

    chunk_data = np.zeros((current_chunk_size, n_voxels), dtype=np.float64)
    data = Parallel(n_jobs=-1,verbose =5)(delayed(load_file)(subpath, mask_nii,resample, smooth, vox_size) for subpath in chunk_paths) #long step here
    chunk_data = np.stack(data)
    del data #saves some RAM

    #slice each chunk into voxel batches and save those
    print("Saving chunk data to voxel batches...")
    for b in range(n_batches):
        batch = f'batch_{b}'
        start = b * batch_size
        end = min(start + batch_size, n_voxels)
        batch_data = chunk_data[:, start:end]
        
        chunk_name = str(chunk_start)+ '_'+str(chunk_start + current_chunk_size)
        data_path = os.path.join(w_dir,batch,'subjects_chunks', f'subjects_{chunk_name}.pkl')
        with open(data_path, 'wb') as f:
            joblib.dump(batch_data, f)
            
###now concatenate chunks inside each batch to get the full voxel data per batch and then build the norm_data with it
for b in range(n_batches):
    full_batch_data = []
    batch = 'batch_' + str(b+1)
    print(f"Concatenating chunk data in {batch}...")
    
    #concatenate all chunks in the batch
    for chunk_start in range(0, len(sub_paths), chunk_size):
        chunk_paths = sub_paths[chunk_start:chunk_start+chunk_size]
        current_chunk_size = len(chunk_paths)
        chunk_name = str(chunk_start)+ '_'+str(chunk_start + current_chunk_size)
        data_path = os.path.join(w_dir,batch,'subjects_chunks', f'subjects_{chunk_name}.pkl')
        with open(data_path, 'rb') as f:
            chunk_data = joblib.load(f)
            full_batch_data.extend(chunk_data)
            
    #build the norm_data
    print(f"Building norm object for {batch}")
    norm_data_path = os.path.join(w_dir, batch, 'norm_data.pkl')

    #important : this should output 2 batch effects
    norm_data = NormData.from_ndarrays(
        name=str(len(df_dem)) + "_subjects", #must not have spaces other job submission might fail
        X = np.array(df_dem.loc[:, covariates]).squeeze(), 
        Y = np.vstack(full_batch_data), 
        batch_effects=np.array(df_dem.loc[:, batch_effects]), 
        subject_ids = np.array(df_dem.index)
        )
    #set variable names
    norm_data = norm_data.assign_coords(covariates=[covariates[0] if x == "covariate_0" else x for x in norm_data.covariates.values])
    norm_data = norm_data.assign_coords(batch_effect_dims=[batch_effects[0] if x == "batch_effect_0" else batch_effects[1] if x == "batch_effect_1" else x for x in norm_data.batch_effect_dims.values])
    norm_data.register_batch_effects() #updates the cached stuff dependent on batch effects names
    norm_data = norm_data.assign_coords(response_vars=[f'voxel_{x}' for x in voxel_indices[start:start+batch_size]])
  
    with open(norm_data_path, 'wb') as f:
        dill.dump(norm_data,f)

###optional cleanup : remove saved chunks to save diskspace - do it when you're sure the batch/norm_data building proceeded correctly otherwise you'll have to load all niftis again
# import shutil
# for b in range(n_batches):
#     batch = f'batch_{b}'
#     if os.path.exists(os.path.join(out_dir,batch,'subjects_chunks')):
#         shutil.rmtree(os.path.join(out_dir,batch,'subjects_chunks'))
     
#%%##option 2 - if we have more than 100GB RAM, we can just load all niftis and keep them all as one python variable       
resp_data = Parallel(n_jobs=-1,verbose =5)(delayed(load_file)(sub_path, mask,resample, smooth, vox_size)for sub_path in sub_paths)

    
#Build norm_data objects and save them
for b in range(n_batches):
    batch = f'batch_{b}'
    print(batch)
    
    #initialize if not there
    if not os.path.exists(w_dir + batch):
        os.makedirs(w_dir + batch)
    
    start = b*batch_size
    data= [arr[start:start+batch_size] for arr in resp_data]
    
    normdata_path = os.path.join(w_dir + batch, 'norm_data.pkl')
    norm_data = NormData.from_ndarrays(
        name=str(len(df_dem)) + "_subjects", #must not have spaces otherwise job submission with the runner will fail
        X = np.array(df_dem.loc[:, covariates]).squeeze(), 
        Y = np.vstack(data), 
        batch_effects=np.array(df_dem.loc[:, batch_effects]), 
        subject_ids = np.array(df_dem.index)
        )
    
    #set variable names
    norm_data = norm_data.assign_coords(covariates=[covariates[0] if x == "covariate_0" else x for x in norm_data.covariates.values])
    norm_data = norm_data.assign_coords(batch_effect_dims=[batch_effects[0] if x == "batch_effect_0" else batch_effects[1] if x == "batch_effect_1" else x for x in norm_data.batch_effect_dims.values])
    norm_data.register_batch_effects() #updates the cached stuff dependent on batch effects names
    norm_data = norm_data.assign_coords(response_vars=[f'voxel_{x}' for x in voxel_indices[start:start+batch_size]])
    
    
    with open(normdata_path, 'wb') as f:
        dill.dump(norm_data,f)