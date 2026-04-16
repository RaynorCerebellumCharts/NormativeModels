#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:39:47 2024

@author: Alice Chavanne, with some adaptations from Charlotte Fraza

This script collects demographic data (age, sex, site) and outputs from ANTs preprocessing (log jacobians, GMV or WMV nifti files)
for the aggreagated reference sample, slices voxelwise values into batches of voxels to accomodate RAM and runtime requirements later on, 
and saves both covariates and batched voxel values into norm_data objects.
"""
from pcntoolkit import NormData, dataio

import numpy as np
import pandas as pd
import os
import glob
import dill
import joblib
from joblib import Parallel, delayed
import nibabel as nib
from nibabel import processing
import nilearn as nil
from nilearn import image

#%% Global settings

modality = 'log_jacs'#choose brain measure ('log_jacs', 'mod_gmv', 'mod_wmv')
prefix = 'BLR_aff_nonlin_'#prefix for model name/directory
suffix = '_wholebrain_2mm_14' #suffix same 

root_dir = '/path/to/root_dir/'
tpl_dir = os.path.join(root_dir,'tpl-MNI152NLin2009cSym')
proc_dir = os.path.join(root_dir, 'models', prefix + modality + suffix)
w_dir = os.path.join(proc_dir,'models')

os.makedirs(proc_dir, exist_ok=True)
os.makedirs(w_dir, exist_ok=True)

data_dir = '/path/to/one/data/storage/partition/'
data_dir_alt = '/path/to/another/data/storage/partition/'
ukb_dir = '/path/to/ukb/data/'

if modality == 'log_jacs':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
if modality == 'mod_gmv':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg0.5.nii.gz')
if modality == 'mod_wmv':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg0.5.nii.gz')

cohorts = ['ABCD',
           'CMI-HBN',
           'HCP_Aging',
           'HCP_Dev',
           'HCP_S1200_processed',
           'PNC',
           'CAMCAN',
           'PING',
           'IXI',
           'UKB']


#details for brain data extraction from niftis
if modality in ['mod_gmv', 'mod_wmv']:
    smooth = True
elif modality == 'log_jacs' : 
    smooth = False #jacobian determinants are already a smooth measure

resample = True #True unless we want to keep isometric 1mmm
vox_size = [2, 2, 2] # 2mm - used if resample 
fwhm = 6 #used if smooth is true

vox_batch_size = 150 #150 voxels is good balance for 1core, 4GB RAM model estimation jobs
#chunk_size = 500 #optional - chunks of subjects to load the niftis, if limited RAM


#%% Load covariates - These have to be in the same order as the cohorts variable !

print('Loading covariate data ...')
dfs = []
ii = 0

dfs.append(pd.read_csv(os.path.join(data_dir, 'ABCD/phenotypes/ABCDstudyNDA_FastTrack_202404/image03.txt'), sep = '\t',  dtype  = 'str'))
dfs[ii] = dfs[ii][dfs[ii]['image_description'] == 'ABCD-T1']
dfs[ii].loc[:,'sex'] = (dfs[ii]['sex'] == 'M').astype(int) #makes males into 1s and female 0, because True or False
dfs[ii].loc[:,'age'] = dfs[ii]['interview_age'].astype(int)/12
dfs[ii] = dfs[ii][dfs[ii]['age'] >= 7] #there is one deficient row with the wrong age
dfs[ii].rename(columns={'subjectkey': 'sub_id'}, inplace=True)
dfs[ii]['sub_id'] = dfs[ii]['sub_id'].str.replace('_', '', regex=False) #correspondence to sub foldernames
dfs[ii].loc[dfs[ii]['visit'] == 'baseline_year_1_arm_1', 'sub_id'] = dfs[ii]['sub_id'] + '_BL'#add timepoint to subject names to differentiate the same-subject rows
dfs[ii].loc[dfs[ii]['visit'] == '2_year_follow_up_y_arm_1', 'sub_id'] = dfs[ii]['sub_id'] + '_2Y'
dfs[ii].loc[dfs[ii]['visit'] == '4_year_follow_up_y_arm_1', 'sub_id'] = dfs[ii]['sub_id'] + '_4Y'
dfs[ii]  = dfs[ii][['sub_id','age','sex']]
#as this is missing site info for BL subjects, we're grabbing site from another csv
baseline = pd.read_csv(os.path.join(data_dir, 'ABCD/phenotypes/ABCD_AnnualRelease5.0/core/imaging/mri_y_adm_info.csv'))
baseline.rename(columns={'src_subject_id': 'sub_id','eventname':'visit', 'mri_info_deviceserialnumber':'site'}, inplace=True)
baseline['sub_id'] = baseline['sub_id'].str.replace('_', '', regex=False) #correspondence to sub foldernames
baseline.loc[baseline['visit'] == 'baseline_year_1_arm_1', 'sub_id'] = baseline['sub_id'] + '_BL'
baseline.loc[baseline['visit'] == '2_year_follow_up_y_arm_1', 'sub_id'] = baseline['sub_id'] + '_2Y'
baseline.loc[baseline['visit'] == '4_year_follow_up_y_arm_1', 'sub_id'] = baseline['sub_id'] + '_4Y'
baseline = baseline[['sub_id','site']]
dfs[ii] = pd.merge(dfs[ii], baseline, on='sub_id', how = 'inner') #merge the two
dfs[ii]['site'] = 'abcd-'+dfs[ii]['site'].astype(str)
dfs[ii]['sub_id'] = 'sub-'+ dfs[ii]['sub_id'].astype(str)
dfs[ii].set_index('sub_id',inplace=True)
dfs[ii] = dfs[ii][~dfs[ii].index.duplicated(keep='first')]
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'CMI-HBN/participants_3882.csv'))) #sex is already formatted 0 female 1 male
dfs[ii].rename(columns={'participant': 'sub_id','Sex': 'sex', 'Age':'age', 'deviceserialnumber':'site'}, inplace=True)
dfs[ii]['site'] = 'hbn-'+dfs[ii]['site'].astype(str)
dfs[ii]['sub_id'] = 'sub-'+ dfs[ii]['sub_id'].astype(str)
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
dfs[ii].sort_index(inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'HCP_Aging/docs/participants.csv'))) #sex is already formatted 0 female 1 male
dfs[ii].rename(columns={'participant_id': 'sub_id','gender': 'sex', 'site_id':'site'}, inplace=True)
dfs[ii]['site'] = 'hcpa-'+dfs[ii]['site'].astype(str)
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'HCP_Dev/downloads/anat_pheno_transfer2/ndar_subject01.txt'),sep="	"))
dfs[ii].drop([0], inplace = True)
dfs[ii]['sex'] = (dfs[ii]['sex'] == 'M').astype(int) #makes males into 1s and female 0, because True or False
dfs[ii]['age'] = dfs[ii]['interview_age'].astype(int)/12
dfs[ii].rename(columns={'src_subject_id': 'sub_id',}, inplace=True)
dfs[ii]['sub_id'] = 'sub-'+ dfs[ii]['sub_id'].astype(str) + '_V1_MR'#compatibility with foldernames
dfs[ii]['site'] =  'hcpd-' +dfs[ii]['site'].astype(str)
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
dfs[ii].sort_index(inplace=True)
ii += 1

dfs.append(pd.read_pickle(os.path.join(data_dir,'HCP_S1200_processed/docs/subcortical_seg_gender_age.pickle'))['covariates']) 
dfs[ii] = pd.DataFrame(dfs[ii])
dfs[ii].rename(columns={0:'age', 1:'sex'}, inplace=True)
dfs[ii]['sex'] = dfs[ii]['sex'].astype(int)
dfs[ii]['site'] = 'hcp1200'
dfs[ii]['sub_id'] = pd.read_pickle(os.path.join(data_dir,'HCP_S1200_processed/docs/subcortical_seg_gender_age.pickle'))['subjects_list'] #it's a struct so load it again
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'PNC/phenotypes/PNC_fullsample_basic_demo.csv')))
dfs[ii].rename(columns={'Subject': 'sub_id','Sex': 'sex', 'Age':'age'}, inplace=True)
dfs[ii]['sex'] = (dfs[ii]['sex'] == 'M').astype(int) #makes males into 1s and female 0, because True or False
dfs[ii]['site'] = 'pnc'
dfs[ii]['sub_id'] = 'sub-' + dfs[ii]['sub_id'].astype(str)
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'CAMCAN/docs/standard_data.csv')))
dfs[ii].rename(columns={'Age': 'age','CCID' : 'sub_id'}, inplace=True)
dfs[ii]['sex'] = (dfs[ii]['Sex'] == 'MALE').astype(int) #makes males into 1s and female 0, because True or False
dfs[ii]['site'] = 'cam'
dfs[ii]['sub_id'] = 'sub-' + dfs[ii]['sub_id'] #to match T1 foldernames
dfs[ii] = dfs[ii].dropna() #remove rows that have Nan for coils because they have no MRI data
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'PING/phenotypes/ping.csv'))) #sex is already formatted 0 female 1 male
dfs[ii].rename(columns={'Unnamed: 0': 'sub_id'}, inplace=True)
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

dfs.append(pd.read_csv(os.path.join(data_dir,'IXI/participants.csv'))) #sex is already formatted 0 female 1 male
dfs[ii].rename(columns={'participant_id': 'sub_id', 'gender':'sex'}, inplace=True)
dfs[ii]['site'] = 'ixi'
dfs[ii]  = dfs[ii][['sub_id','age','sex','site']]
dfs[ii].set_index('sub_id',inplace=True)
dfs[ii] = dfs[ii][~dfs[ii].index.duplicated(keep='first')] #some IXI subjects have duplicate rows, maybe were scanned several time so keep only the first
ii += 1

dfs.append(pd.read_csv(os.path.join(ukb_dir,'phenotypes/age_sex_site_ukb42006.csv')))
dfs[ii] = dfs[ii].rename(columns={'21003-2.0':'age', '54-2.0':'site', '31-0.0':'sex'})
dfs[ii]['sub_id'] = dfs[ii]['eid'].astype(str)
dfs[ii] = dfs[ii][['sub_id','age','sex','site']]
dfs[ii] = dfs[ii].dropna() #remove unused sub-id rows
dfs[ii]['site'] = 'ukb-' + dfs[ii]['site'].astype(str)
dfs[ii].set_index('sub_id',inplace=True)
ii += 1

#%% Filter covariates for excluded subjects
#excludes the QCed-out subjects based on T1 Warped files from ants (and, if relevant, the warped segmentation from fsl) 
# the ones that have no relevant (jacobian, GMV...) image + the UKB ones with more than 5% outlier voxel values
 
print('Filtering covariate data ... ')

if modality == 'log_jacs': 
    list_name = 'have_jacs.txt'
elif modality == 'mod_gmv':# double up just in case gmv and wmv don't have identical exclusions
    list_name = 'have_gmv.txt'
elif modality == 'mod_wmv':
    list_name = 'have_wmv.txt'

#now, we collect lists of excluded subjects
for ii, cohort in enumerate(cohorts):
    print(cohort)
    
    if cohort == 'UKB':#lives in another partition
        excluded = pd.read_csv(os.path.join(ukb_dir,'qc','T1_ants','sub_exclude_ants_reg.txt'), header = None)
        excluded_anat = pd.read_csv(os.path.join(ukb_dir,'qc','T1','sub_exclude.txt'), header = None)#add the anatomical exclusions
        excluded_outlier = pd.read_csv(os.path.join(ukb_dir,'qc','T1_ants','sub_exclude_outliers_5PercentofVoxels_jacs.txt'), header = None)
        excluded = pd.concat([excluded, excluded_anat, excluded_outlier])
        excluded_seg = pd.DataFrame()
        
        if modality in ['mod_gmv', 'mod_wmv']: #then we exclude those from segmentation QC as well
            excluded_seg = pd.read_csv(os.path.join(ukb_dir,'qc','T1_ants','sub_exclude_seg.txt'), header = None)
            excluded = pd.concat([excluded, excluded_seg])
        
        #the have_scans are just generated from a wildcard listing of files in terminal with find
        have_scans = pd.read_csv(os.path.join(ukb_dir,'qc', 'T1_ants', list_name), header = None)[0]
        
    elif cohort in ['HCP_Aging','HCP_Dev','HCP_S1200_processed','PNC']: #those live elsewhere
    
        excluded = pd.read_csv(os.path.join(data_dir_alt,cohort,'qc','T1_ants','sub_exclude_ants_reg.txt'), header = None)
        excluded_seg = pd.DataFrame()
        
        if modality in ['mod_gmv', 'mod_wmv']:
            excluded_seg = pd.read_csv(os.path.join(data_dir_alt,cohort,'qc','T1_ants','sub_exclude_seg.txt'), header = None)
            excluded = pd.concat([excluded, excluded_seg])

        have_scans = pd.read_csv(os.path.join(data_dir_alt,cohort,'qc','T1_ants',list_name), header = None)[0]

    elif cohort in ['PING','CAMCAN','IXI','CMI-HBN'] : #other partition
        excluded = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants','sub_exclude_ants_reg.txt'), header = None)
        excluded_seg = pd.DataFrame()
        
        if modality in ['mod_gmv', 'mod_wmv']:
            excluded_seg = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants','sub_exclude_seg.txt'), header = None)
            excluded = pd.concat([excluded, excluded_seg])
        
        have_scans = pd.read_csv(os.path.join(data_dir,cohort, 'qc','T1_ants',list_name), header = None)[0]

    elif cohort == 'ABCD': #different folder architecture because longitudinal dataset
        sessions =  ['ses-baselineYear1Arm1', 'ses-2YearFollowUpYArm1', 'ses-4YearFollowUpYArm1']
        ses_suffixes = ['_BL', '_2Y', '_4Y']
        excluded = pd.DataFrame()
        excluded_seg = pd.DataFrame()
        have_scans = pd.DataFrame()
    
        for j, ses in enumerate(sessions) :
            ses_suffix = ses_suffixes[j]
            excluded_timepoint = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants', ses ,'sub_exclude_ants_reg.txt'), header = None)
            excluded_bet = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants', ses,'sub_exclude_ants_bet.txt'), header = None) #those could only be seen in seg QC but were due to ants brain extraction failure so also apply here
            excluded_timepoint = pd.concat([excluded_timepoint,excluded_bet])
            excluded_seg_timepoint = pd.DataFrame()
            
            if modality in ['mod_gmv', 'mod_wmv']:
                excluded_seg_timepoint = pd.read_csv(os.path.join(data_dir,cohort,'qc','T1_ants', ses ,'sub_exclude_seg.txt'), header = None)
                excluded_timepoint = pd.concat([excluded_timepoint, excluded_seg_timepoint])
                excluded_seg_timepoint = excluded_seg_timepoint + ses_suffix #compatibility with sub_id in dfs
            
            excluded_timepoint = excluded_timepoint + ses_suffix #same
            
            have_scans_timepoint = pd.read_csv(os.path.join(data_dir,cohort, 'qc', 'T1_ants', ses, list_name), header = None)[0]
            have_scans_timepoint = have_scans_timepoint + ses_suffix

            excluded = pd.concat([excluded, excluded_timepoint], axis = 0)
            excluded_seg = pd.concat([excluded_seg, excluded_seg_timepoint])
            have_scans= pd.concat([have_scans, have_scans_timepoint], axis = 0)

        have_scans = have_scans[0]
        
        
    excluded = excluded.astype('string')# in case some subjects ids are numbers
    
    #exclude those with no output jacobian nifti files
    have_scans = have_scans.astype(str).to_list()
    excluded_no_scans = pd.DataFrame(dfs[ii][~dfs[ii].index.isin(have_scans)].index)
    excluded_no_scans.columns=[0] #make it concatenable with the other one 
     
    excluded = pd.concat([excluded, excluded_no_scans],ignore_index = True)
    excluded = excluded.values.flatten().tolist()
    
    excluded_seg = excluded_seg.astype('string')
    excluded_seg = excluded_seg.values.flatten().tolist()
    
    print('from', len(dfs[ii]), 'subjects, exclude', len(dfs[ii].loc[dfs[ii].index.isin(excluded)]))
    print('specifically, excluded for seg reasons', len(dfs[ii].loc[dfs[ii].index.isin(excluded_seg)]))
    print('retaining', len(dfs[ii])-len(dfs[ii].loc[dfs[ii].index.isin(excluded)]), '\n')

    dfs[ii] = dfs[ii].drop(excluded, errors='ignore')

#drop sites that have less than 60 subjects (i.e. ~30 for each sex) to allow for comfortable model estimation 
for ii in range(len(dfs)):
    counts = dfs[ii]['site'].value_counts() 
    valid = counts[counts >= 60].index
    dfs[ii]= dfs[ii][dfs[ii]['site'].isin(valid)]
    print(dfs[ii]['site'].value_counts())
    dfs[ii].sort_index(inplace = True) #just in case some csv were not ordered


# now join 
df_dem = pd.concat(dfs)
df_dem.loc[:,'sex'] = df_dem.loc[:,'sex'].astype(int)#sex must be same datatype everywhere otherwise that'll double the batch effect
print(df_dem.shape)

#%%#grab nifti file paths for all subjects

filepaths_all = []

#name the nifti to extract
if modality == 'log_jacs':
    file_name = 'T1_BrainNorm_Jacobian.nii.gz'
elif modality == 'mod_gmv':
    file_name = 'T1_BrainNorm_GMV.nii.gz'
elif modality == 'mod_wmv':
    file_name = 'T1_BrainNorm_WMV.nii.gz'
    
#loop through every cohort
for ii, cohort in enumerate(cohorts):
    if cohort == 'UKB':
        f_pattern =os.path.join(ukb_dir,'subjects','{}','T1', 'T1_ants_MNI152NLin2009cSym', file_name)

    elif cohort in ['HCP_Aging','HCP_Dev','HCP_S1200_processed','PNC',]: #those live elsewher
        f_pattern =os.path.join(data_dir_alt, cohort,'subjects','{}','T1_ants_MNI152NLin2009cSym', file_name)
        
    elif cohort in ['PING','CAMCAN','IXI','CMI-HBN'] :#those live elsewhere
        f_pattern =os.path.join(data_dir, cohort,'subjects','{}','T1_ants_MNI152NLin2009cSym', file_name)
        
    if cohort == 'ABCD':#longitudinal cohort
        print('Tracking down file paths...')
        filepaths = []
        sessions =  ['ses-baselineYear1Arm1', 'ses-2YearFollowUpYArm1', 'ses-4YearFollowUpYArm1']
        ses_suffixes = ['_BL', '_2Y', '_4Y']
        
        for sub in sorted(dfs[ii].index.tolist()):
            for j, ses in enumerate(sessions) :
                ses_suffix = ses_suffixes[j] 
                if ses_suffix in sub: #grab the right timepoint filepaths
                    sub = sub.replace(ses_suffix,'')
                    f_pattern =os.path.join(data_dir, cohort,'subjects','{}',ses, 'anat','ABCD-T1-min-preproc*', 'T1_ants_MNI152NLin2009cSym', file_name)  
                    file = glob.glob(f_pattern.format(sub))
                    filepaths.extend(file)
    else : #all non-longitudinal cohorts
        filepaths = [f_pattern.format(sub) for sub in sorted(dfs[ii].index.tolist())]
    filepaths_all.extend(filepaths)
    
# Ensure that the lengths of demographics and nifti paths are the same
if len(filepaths_all) != len(df_dem):
    raise ValueError("Length of filepaths and df_dem do not match!")   
    
#%% Load the response data as nifti in chunks of subjects, populate batch pkl resp files without creating a whole respfile to avoid memory errors
#Still a greedy step, maybe set spyder to 128gb at least 

#grab total number of voxels from mask  
mask = dataio.fileio.load(mask_nii, mask=mask_nii, vol=False).T
n_voxels = len(mask)
n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)
voxel_indices = list(range(n_voxels))

#grab voxel values from nifti file
def load_file(subpath, mask_nii, resample, smooth, vox_size):

    img = nib.load(subpath)
    if resample :
        img = nib.processing.resample_to_output(img, vox_size)
    if smooth:
        img = nil.image.smooth_img(img,fwhm)
    data = img.get_fdata()
    data = dataio.fileio.vol2vec(data, nib.load(mask_nii).get_fdata())

    return data

#%%###if not enough RAM, we have to load and save the nifti data in chunks

# # Initialize directories and empty files to store chunk data
# for b in range(n_batches):
#     os.chdir(w_dir)
#     batch = f'batch_{b}'
#     if not os.path.exists(os.path.join(w_dir, batch)):
#         os.makedirs(os.path.join(w_dir, batch)
#         os.makedirs(os.path.join(w_dir, batch, 'respfile_chunks')

# # we load niftis from chunks of subjects, and from this chunk we extract voxel values to populate batches of voxels
# for chunk_start in range(0, len(filepaths_all), chunk_size):
#     chunk_paths = filepaths_all[chunk_start:chunk_start+chunk_size]
#     current_chunk_size = len(chunk_paths) #covers the last chunk which will have less subjects than normal chunk_size
#     chunk_data = np.zeros((current_chunk_size, n_voxels), dtype=np.float64)
    
#     print(f"Loading chunk: subjects {chunk_start} to {chunk_start + current_chunk_size}")

#     data = Parallel(n_jobs=-1,verbose =5)(delayed(load_file)(subpath, mask, resample, smooth, vox_size) for subpath in chunk_paths)
#     chunk_data = np.stack(data)
#     del data #saves RAM 
    
#     print("Saving chunk data to voxel batches...")
#     for b in range(n_batches):
#         start = b * vox_batch_size
#         end = min(start + vox_batch_size, n_voxels)
#         batch_data = chunk_data[:, start:end]
#         batch = f'batch_{b}'
#         chunkname = f'{chunk_start}_{chunk_start + current_chunk_size}'
#         batch_path = os.path.join(w_dir, batch, 'respfile_chunks', f'respfile_{batch}_{chunkname}.pkl')
#         with open(batch_path, 'wb') as f:
#             joblib.dump(batch_data, f)

#optional - we can now concatenate subjects chunks inside each voxel batch to get a the full values per batch -this can take some diskspace though
# for b in range(n_batches):
#     fullbatch_data = []
#     batch = f'batch_{b}'
#     print(f"Concatenating subject chunks data in {batch}...")
#     for chunk_start in range(0, len(filepaths_all), chunk_size):
#         chunk_paths = filepaths_all[chunk_start:chunk_start+chunk_size]
#         current_chunk_size = len(chunk_paths)
#         chunkname = f'{chunk_start}_{chunk_start + current_chunk_size}'
#         chunk_path = os.path.join(w_dir, batch, 'respfile_chunks', f'respfile_{batch}_{chunkname}.pkl')
#         with open(chunk_path, 'rb') as f:
#             chunk_data = joblib.load(f)
#             fullbatch_data.extend(chunk_data)
#     batch_path = os.path.join(w_dir, batch, 'respfile.pkl')
#     with open(batch_path, 'wb') as f:   
#         pickle.dump(pd.DataFrame(np.vstack(fullbatch_data),index=df_dem.index), f) 

     
#%%##if we have a lot of RAM, we can just load all niftis at once    
resp_data = Parallel(n_jobs=-1,verbose =5)(delayed(load_file)(sub_path, mask_nii,resample, smooth, vox_size) for sub_path in filepaths_all)

#specify the covariate(s) and batch effect(s) we want to go in the model, these will be stored the norm_data object
covariates = ["age"]
batch_effects = ["sex", "site"]

#Build norm_data objects and save them
for b in range(n_batches):
    batch = f'batch_{b}'
    print(batch)
    
    #initialize directories just in case
    if not os.path.exists(os.path.join(w_dir, batch)):
        os.makedirs(os.path.join(w_dir, batch))
    
    start = b*vox_batch_size
    data= [arr[start:start+vox_batch_size] for arr in resp_data]

    normdata_path = os.path.join(w_dir + batch, 'norm_data.pkl')
    norm_data = NormData.from_ndarrays(
        name=str(len(df_dem)) + "_subjects", #name must not have spaces other job submission with the runner will fail
        X = np.array(df_dem.loc[:, covariates]), 
        Y = np.vstack(data), 
        batch_effects=np.array(df_dem.loc[:, batch_effects]), 
        subject_ids = np.array(df_dem.index)
        )
    
    #rename covariates and batch effects for clarity - but this has to be kept straight in downstream analyses 
    norm_data = norm_data.assign_coords(covariates=[covariates[0] if x == "covariate_0" else x for x in norm_data.covariates.values])
    norm_data = norm_data.assign_coords(batch_effect_dims=[batch_effects[0] if x == "batch_effect_0" else batch_effects[1] if x == "batch_effect_1" else x for x in norm_data.batch_effect_dims.values])
    norm_data.register_batch_effects() #updates the cached stuff dependent on batch effects names
    norm_data = norm_data.assign_coords(response_vars=[f'voxel_{x}' for x in voxel_indices[start:start+vox_batch_size]])
    
    with open(normdata_path, 'wb') as f:
        dill.dump(norm_data,f) #normdata objects can contain lambda functions, which normal pickle doesn't like
        
        
#%% Optional - calculate demographics stats per cohort
ii = 0
for ii, cohort in enumerate(cohorts):
    if cohort == 'ABCD':
        dfs[ii]["session"] = dfs[ii].index.str.split("_").str[1]
        for ses in sorted(dfs[ii]["session"].unique()):
            print(cohort)
            print(ses)
            df_ses = dfs[ii][dfs[ii]["session"] == ses]
            print(f"subjects total {len(df_ses.index.tolist())}")
            print(f"unique scanners {len(np.unique(df_ses["site"]))}")
            print(f"female {100*df_ses["sex"].isin([0]).sum()/len(df_ses["sex"]):.1f} percent")
            print(f"male {100*df_ses["sex"].isin([1]).sum()/len(df_ses["sex"]):.1f} percent")
            print(f"age mean {np.mean(df_ses["age"]):.1f}")
            print(f"age sd {np.std(df_ses["age"]):.1f}")
            
        dfs[ii].drop(columns = ["session"], inplace = True)
        dfs[ii]["sub-id"] = dfs[ii].index.str.split("_").str[0]
        df_unique = dfs[ii].groupby("sub-id").head(1)
        print(cohort)
        print("Unique subjects")
        print(f"subjects total {len(df_unique.index.tolist())}")
        print(f"unique scanners {len(np.unique(df_unique["site"]))}")
        print(f"female {100*df_unique["sex"].isin([0]).sum()/len(df_unique["sex"]):.1f} percent")
        print(f"male {100*df_unique["sex"].isin([1]).sum()/len(df_unique["sex"]):.1f} percent")
        dfs[ii].drop(columns = ["sub-id"], inplace = True)
    else:
        print(cohort)
        print(f"subjects total {len(dfs[ii].index.tolist())}")
        print(f"unique scanners {len(np.unique(dfs[ii]["site"]))}")
        print(f"female {100*dfs[ii]["sex"].isin([0]).sum()/len(dfs[ii]["sex"]):.1f} percent")
        print(f"male {100*dfs[ii]["sex"].isin([1]).sum()/len(dfs[ii]["sex"]):.1f} percent")
        print(f"age mean {np.mean(dfs[ii]["age"]):.1f}")
        print(f"age sd {np.std(dfs[ii]["age"]):.1f}")

#%% Optional - Ridge plots to visualize age distribution in all cohorts

import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import gaussian_kde


sns.set_style("white")

for i, df in enumerate(dfs) :
    df['cohort'] = cohorts[i]
    
df_dem = pd.concat(dfs)
df_dem.loc[:,'sex'] = df_dem.loc[:,'sex'].astype(int)

df_dem["cohort"] = df_dem["cohort"].replace({"HCP_S1200_processed": "HCP_1200"})

variable = 'age'
response_var= 'cohort'
sex_var = 'sex'
df = df_dem
save_dir = os.path.join(root_dir,'for_figures')
# save_dir = False

def ridge_plot_overlap(df, variable, response_var, sex_var='sex', save_dir = False):
    cohorts_sorted = sorted(df[response_var].unique())
    sexes = sorted(df[sex_var].unique())

    sex_palette = {0: "chocolate", 1: "lightseagreen"} 
    sex_label_map = {0: "Female", 1: "Male"}

    fig, ax = plt.subplots(figsize=(3.5,0.6 * len(cohorts_sorted)))
    global_xmin = df[variable].min()
    global_xmax = df[variable].max()
    xs = np.linspace(global_xmin, global_xmax, 400)

    y_offset = 0
    y_gap = 0.5  
    
    for i, cohort in enumerate(cohorts_sorted[::-1]):
        sub_df = df[df[response_var] == cohort]
        n_cohort = len(sub_df)

        for sex in sexes:
            sex_df = sub_df[sub_df[sex_var] == sex]
            label = sex_label_map[sex] if i==0 else None
            x = sex_df[variable].values
            if len(x) == 0:
                continue
            kde = gaussian_kde(x)
            kde_values = kde(xs)

            ax.fill_between(xs, kde_values*3 + y_offset, y_offset, color=sex_palette[sex], alpha=0.3, label = label)
            ax.plot(xs, kde_values*3 + y_offset, color=sex_palette[sex], lw=0.1)
            
        ax.text(-14, y_offset + 0.12, f"{cohort}\nN={n_cohort:,}", ha="center", va="center", fontsize=8)
        y_offset +=y_gap

    sns.despine(ax=ax,left=True, bottom=True)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Age", fontsize = 9)
    ax.grid(True, axis="x", color="lightgrey", linewidth=0.5, linestyle=":")
    plt.xticks(fontsize=9)
    plt.ylim(0,5.4)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0.75, 0.97), loc="upper left",fontsize=7)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "Ridge_plots_all_datasets.png"), dpi=300)
    plt.show()
    
ridge_plot_overlap(df_dem, variable="age", response_var="cohort", save_dir = save_dir)  
