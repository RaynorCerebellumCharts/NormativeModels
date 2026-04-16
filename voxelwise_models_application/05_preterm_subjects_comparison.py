#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:03:57 2025

@author: Alice Chavanne

This scripts collects the Z scores from ABCD subjects born preterm (in the test set of estimated normative models).
Z-scores from sujects born preterm are then compared with those from controls born at-term. 
The proportion of extreme Z-scores (|Z|>2) are compared between the two groups, both visually with voxelwise nifti maps and 
also with a Mann-Whiteny U-test of whole-brain burden of extreme Z-scores. Mean TFCE differences in z-scores are also tested.
Also contains voxel-specific plotting of preterm vs. control centile plots.

"""

import warnings
import pandas as pd
from pcntoolkit import (
    NormativeModel,
    NormData,
    dataio,
)
from pcntoolkit.dataio.fileio import save as ptksave
import pcntoolkit.util.output
import logging
import os
import pickle
import sys
import nibabel as nib
import joblib
from joblib import Parallel, delayed
import numpy as np 
import copy
from sklearn.model_selection import train_test_split as skl_split
import re    
import collections
from nilearn.mass_univariate import permuted_ols
from nilearn.input_data import NiftiMasker
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu


# Suppress some annoying warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(False)

#%% Global settings
modality = 'log_jacs' #'log_jacs', 'mod_gmv_and_wmv
prefix = 'BLR_aff_nonlin_'#prefix for model name/directory
suffix = '_wholebrain_2mm_14' #suffix same #_labelmask_2mm_smoothed_fwhm6_07 #_wholebrain_2mm_14
venv_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable))) #path to toolkit
root_dir = '/path_to_root_dir/'
base_dir =os.path.join(root_dir,'models/')

preterm_thr =  32 #gestational age threshold for preterm

#detail from model estimation
vox_batch_size = 150
splits = [0.5,0.5]

#nifti details
if modality == 'log_jacs':
    name = '58539_subjects'
    proc_dir = os.path.join(base_dir,prefix + modality + suffix)
    w_dir = os.path.join(proc_dir,'models/')
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
    ex_nii = mask_nii
    smooth = False
    
    mask = dataio.fileio.load(mask_nii, mask=mask_nii, vol=False).T #load all nonzero voxels of the mask
    n_voxels = len(mask)
    n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)
    voxel_indices = list(range(n_voxels))

elif modality == 'mod_gmv_and_wmv':
    name = '53593_subjects'
    mask_nii_gm = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg0.5.nii.gz')
    mask_gm = dataio.fileio.load(mask_nii_gm, mask=mask_nii_gm, vol=False).T #load all nonzero voxels of the mask
    n_voxels_gm = len(mask_gm)
    n_batches_gm = n_voxels_gm // vox_batch_size + int(n_voxels_gm % vox_batch_size != 0)
    voxel_indices_gm = list(range(n_voxels_gm))
    
    mask_nii_wm = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg0.5.nii.gz')
    mask_wm = dataio.fileio.load(mask_nii_wm, mask=mask_nii_wm, vol=False).T #load all nonzero voxels of the mask
    n_voxels_wm = len(mask_wm)
    n_batches_wm = n_voxels_wm // vox_batch_size + int(n_voxels_wm % vox_batch_size != 0)
    voxel_indices_wm = list(range(n_voxels_wm))
    
    #now combine the two (non-overlapping) masks in a whole-brain approach
    mask_nii_combined = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_combined_label-GM_WM_mask.nii.gz')
    w_dir_gm= os.path.join(base_dir,prefix + 'mod_gmv'+ suffix, 'models')
    w_dir_wm= os.path.join(base_dir,prefix + 'mod_wmv'+ suffix, 'models') 
    
    #just for plotting back onto the combined mask 
    ex_nii = mask_nii_combined
    mask_nii = mask_nii_combined
    smooth = True    
    
    #output files will be saved in the gm model folder 
    w_dir = w_dir_gm
    
if not os.path.exists(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison')):
    os.makedirs(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison'))
    os.makedirs(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals'))

#details for brain data extraction from niftis
resample = True
vox_size = [2, 2, 2] #2mm

#%% prepare to stratify ABCD so that the same subject doesn't end up in both train and test

#First, extract all ABCD site names from any normdata object
# we need them to set aside ABCD subjects later
normdata_path = os.path.join(w_dir, 'batch_0', 'norm_data.pkl')

with open(normdata_path, 'rb') as f:
    norm_data = pickle.load(f)
abcd_sites = {np.str_('site'):[v for values in norm_data.unique_batch_effects.values() for v in values if "abcd" in str(v)]}



def subject_level_split(data, split, random_state: int = 42,): #uses a slightly different split to balance for clinical subjects forced into test set

    orig_ids = data.subject_ids.values

    # strip trailing timepoint name
    def extract_subject_fn(x):
        parts = re.split(r"[_\-]", x)
        return "_".join(parts[:-1]) if len(parts) > 1 else x

    subjects_base = np.array([extract_subject_fn(s) for s in orig_ids])

    #subject to indices map
    subject_to_indices = collections.defaultdict(list)
    for idx, sub in enumerate(subjects_base):
        subject_to_indices[sub].append(idx)

    unique_subjects = list(subject_to_indices.keys())
    
    #stratification labels per subject
    batch_effects_stringified = data.concatenate_string_arrays(
        *[data.batch_effects[:, i].astype(str) for i in range(data.batch_effects.shape[1])]
    )

    #one label per unique subject - take first one if several
    subject_labels = [
        batch_effects_stringified[subject_to_indices[sub][-1]].item()
        for sub in unique_subjects
    ]

    train_subjects, test_subjects = skl_split(
        unique_subjects,
        test_size= 1-split,
        random_state=random_state,
        stratify=subject_labels
    )
    
    #subject lists back to row indices
    train_idx = np.concatenate([subject_to_indices[sub] for sub in train_subjects])
    test_idx = np.concatenate([subject_to_indices[sub] for sub in test_subjects])

    #Build new NormData objects
    train = data.isel(observations=train_idx)
    test = data.isel(observations=test_idx)

    train.attrs = copy.deepcopy(data.attrs)
    test.attrs = copy.deepcopy(data.attrs)

    train.attrs["name"] = f"{data.attrs.get('name', 'data')}_train"
    test.attrs["name"] = f"{data.attrs.get('name', 'data')}_test"

    return train, test


# %%  get the preterm subjects and controls subjects Z data

#round up preterm subject ids
g_age=pd.read_csv('/project_cephfs/3022017.06/ABCD/phenotypes/ABCDstudyNDA_AnnualRelease4.0/Package_1199073/abcd_devhxss01.txt', sep = '\t')
g_age = g_age.iloc[1:]
g_age.rename(columns={'subjectkey': 'sub_id','devhx_ss_12_p':'gestational_age'}, inplace=True)
g_age = g_age[['sub_id','gestational_age']]
g_age = g_age.dropna()
g_age['sub_id'] = g_age['sub_id'].str.replace('_', '', regex=False) #correspondence to sub foldernames
g_age['sub_id'] = 'sub-'+ g_age['sub_id']
g_age['gestational_age'] = 40-g_age['gestational_age'].astype(int) # convert to actual GA
g_age = g_age[g_age.gestational_age > 0] # drop the 'uncertain' 999 values
preterm = g_age[g_age['gestational_age']<=preterm_thr]
preterm.reset_index(drop=True, inplace = True)
clin_subs = preterm.iloc[:,0].values
ses_suffixes = ['_BL', '_2Y', '_4Y'] #add all possible observations for these subjects to cover longitudinal options
all_clin_subs =np.char.add(np.repeat(clin_subs, len(ses_suffixes)), np.tile(ses_suffixes, len(clin_subs)))

#round up subject ids of subjects with mild/doubtful prematurity to push them out of control group
preterm = g_age[g_age['gestational_age']<=37] 
preterm = preterm[g_age['gestational_age']>preterm_thr]
preterm.reset_index(drop=True, inplace = True)
exclude_mild = preterm.iloc[:,0].values
ses_suffixes = ['_BL', '_2Y', '_4Y'] #add all possible observations for these subjects because longitudinal
all_exclude_mild =np.char.add(np.repeat(exclude_mild, len(ses_suffixes)), np.tile(ses_suffixes, len(exclude_mild)))


def split_normdata(all_clin_subs, norm_data, splits, abcd_sites):

    #separate the clinical subjects from norm_data 
    clin_subs_existing = [s for s in all_clin_subs if s in norm_data.subject_ids.values]
    clin_normdata = norm_data.where(norm_data.subject_ids.isin(clin_subs_existing),drop = True)
    remaining = norm_data.where(~norm_data.subject_ids.isin(clin_subs_existing), drop=True)
    
    #re-make subsets into proper norm_data objects to get subclass methods
    clin_normdata = NormData(data_vars=clin_normdata.data_vars,coords=clin_normdata.coords,
        attrs=clin_normdata.attrs,name=clin_normdata.name) #here we define it only to get proper test set, it doesn't have zscores yet so no use

    remaining = NormData(data_vars=remaining.data_vars,coords=remaining.coords,
        attrs=remaining.attrs,name = remaining.name)
    
    #now take out abcd (non-preterm) to split it independently
    abcd_subset = remaining.select_batch_effects(name = 'abcd_subset', batch_effects = abcd_sites)
    abcd_subs = abcd_subset.subject_ids.values
    abcd_subset_train, abcd_subset_test= subject_level_split(abcd_subset, split = 0.6)
    
    remaining = norm_data.where(~remaining.subject_ids.isin(abcd_subs), drop=True)
    remaining = NormData(data_vars=remaining.data_vars,coords=remaining.coords,
        attrs=remaining.attrs,name = remaining.name)
    remaining.register_batch_effects()
    
    #split and put subjects back into the test set
    train, test = remaining.train_test_split(splits = splits)
    train = train.merge(abcd_subset_train)
    test = test.merge(abcd_subset_test)
    test = test.merge(clin_normdata)
    
    return train, test, clin_subs_existing


#we split those two functions because we need to load the zscores on the test normdata before splitting by group
def normdata_per_group(clin_subs_existing, test, ses_suffixes, all_exclude_mild):

    control_subs = [s for s in test.subject_ids.values if any(tag in s for tag in ses_suffixes)]
    
    #push out preterms and mild preterms
    control_subs = [s for s in control_subs if s not in clin_subs_existing]
    control_subs = [s for s in control_subs if s not in all_exclude_mild]
    
    #grab all abcd controls except the mild preterms
    ctrl_normdata = test.where(test.subject_ids.isin(control_subs), drop=True)
    ctrl_normdata = NormData(data_vars=ctrl_normdata.data_vars,coords=ctrl_normdata.coords,attrs=ctrl_normdata.attrs,name=ctrl_normdata.name)
    
    clin_normdata = test.where(test.subject_ids.isin(clin_subs_existing), drop = True) #here we extract it because it now has zscores
    clin_normdata = NormData(data_vars=clin_normdata.data_vars,coords=clin_normdata.coords,attrs=clin_normdata.attrs,name=clin_normdata.name)
    return clin_normdata, ctrl_normdata


def process_batch (b, w_dir, all_clin_subs, splits, ses_suffix, abcd_sites, all_exclude_mild) :
    batch = f'batch_{b}'
    print(batch, flush = True)
    normdata_path = os.path.join(w_dir, batch, 'norm_data.pkl')
    
    with open(normdata_path, 'rb') as f:
        norm_data = pickle.load(f)
        
    train, test, clin_subs_existing = split_normdata(all_clin_subs, norm_data, splits, abcd_sites)
    test.load_zscores(save_dir = os.path.join(w_dir, batch, 'results'))
    clin_normdata, ctrl_normdata = normdata_per_group(clin_subs_existing, test, ses_suffixes, all_exclude_mild)

    Z_clin = pd.DataFrame(clin_normdata.Z.values, index = clin_normdata.subject_ids.values, columns = clin_normdata.Z.response_vars.values)
    Z_ctrl = pd.DataFrame(ctrl_normdata.Z.values, index = ctrl_normdata.subject_ids.values, columns = ctrl_normdata.Z.response_vars.values)
    return Z_clin, Z_ctrl


if modality == 'log_jacs':
    results = Parallel(n_jobs=-1, verbose = 5)(delayed(process_batch)(b, w_dir, all_clin_subs, splits, ses_suffixes, abcd_sites,all_exclude_mild)for b in range(n_batches)[:1])
    all_Z_clin, all_Z_ctrl = zip(*results)
    Z_preterm = pd.concat(all_Z_clin, axis = 1)
    Z_controls = pd.concat(all_Z_ctrl, axis = 1)

elif modality == 'mod_gmv_and_wmv':
    results_gm = Parallel(n_jobs=-1, verbose = 5)(delayed(process_batch)(b, w_dir_gm, all_clin_subs, splits, ses_suffixes,abcd_sites, all_exclude_mild)for b in range(n_batches_gm)[:])
    results_wm = Parallel(n_jobs=-1, verbose = 5)(delayed(process_batch)(b, w_dir_wm, all_clin_subs, splits, ses_suffixes, abcd_sites, all_exclude_mild)for b in range(n_batches_wm)[:])
    all_Z_clin_gm, all_Z_ctrl_gm = zip(*results_gm)
    all_Z_clin_wm, all_Z_ctrl_wm = zip(*results_wm)


    Z_preterm_gm = pd.concat(all_Z_clin_gm, axis = 1) 
    Z_controls_gm = pd.concat(all_Z_ctrl_gm, axis = 1) 
    Z_preterm_wm = pd.concat(all_Z_clin_wm, axis = 1) 
    Z_controls_wm = pd.concat(all_Z_ctrl_wm, axis = 1)

    #combine GM and WM Zs with mask handling
    mask_combined = nib.load(mask_nii_combined).get_fdata().astype(bool)
    mask_gm = nib.load(mask_nii_gm).get_fdata().astype(bool)
    mask_wm = nib.load(mask_nii_wm).get_fdata().astype(bool)
    
    #Get indices of GM/WM voxels within combined mask
    mask_combined_idx = np.where(mask_combined.ravel())[0]
    gm_in_combined = mask_gm.ravel()[mask_combined_idx]
    wm_in_combined = mask_wm.ravel()[mask_combined_idx]
    gm_idx = np.where(gm_in_combined)[0]
    wm_idx = np.where(wm_in_combined)[0]
    
    #manage voxel names for final matrix
    gm_names = (Z_preterm_gm.columns + "_gm").tolist() #same for preterms and controls
    wm_names = (Z_preterm_wm.columns + "_wm").tolist()
    colnames = [None] * mask_combined.sum()
    
    for i,j in enumerate(gm_idx):
        colnames[j] = gm_names[i]
    
    for i,j in enumerate(wm_idx):
        colnames[j] = wm_names[i]
    
    #Combine GM and WM voxels in order
    N_preterm = len(Z_preterm_gm)
    Z_preterm = np.zeros((N_preterm, mask_combined.sum()), dtype=float)
    Z_preterm[:, gm_idx] = Z_preterm_gm.to_numpy()
    Z_preterm[:, wm_idx] = Z_preterm_wm.to_numpy()
    Z_preterm = pd.DataFrame(Z_preterm, index = Z_preterm_gm.index, columns=colnames)

    N_controls = len(Z_controls_gm)
    Z_controls = np.zeros((N_controls, mask_combined.sum()), dtype=float)
    Z_controls[:, gm_idx] = Z_controls_gm.to_numpy()
    Z_controls[:, wm_idx] = Z_controls_wm.to_numpy()
    Z_controls = pd.DataFrame(Z_controls, index = Z_controls_gm.index, columns=colnames)
    


nan_cols = Z_preterm.columns[Z_preterm.isna().any()].tolist() #check if failed vars or batches
print ('Failed for some reason', nan_cols)

with open(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', f'Z_preterm_{preterm_thr}.pkl'), 'wb') as f:
    pickle.dump(Z_preterm, f)
        
with open(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', 'Z_controls.pkl'), 'wb') as f:
    pickle.dump(Z_controls, f)

#%% Optional - get demographics of preterms

#take random batch and rebuild the exact test norm_data
batch = 'batch_0'
normdata_path = os.path.join(w_dir, batch, 'norm_data.pkl')
with open(normdata_path, 'rb') as f:
    norm_data = pickle.load(f)
    
train, test, clin_subs_existing = split_normdata(all_clin_subs, norm_data, splits, abcd_sites)
test.load_zscores(save_dir = os.path.join(w_dir, batch, 'results'))
clin_normdata, ctrl_normdata = normdata_per_group(clin_subs_existing, test, ses_suffixes,all_exclude_mild)

df_preterm = df = pd.DataFrame({
    "sub_id": clin_normdata.subjects.values,
    "age": clin_normdata.X.values.squeeze(),
    "sex": clin_normdata.batch_effects.isel(batch_effect_dims=0).values,
    "site": clin_normdata.batch_effects.isel(batch_effect_dims=1).values,
})
df_preterm["sex"] = df_preterm["sex"].astype(int)
df_preterm.set_index('sub_id',inplace=True)
df_preterm["session"] = df_preterm.index.str.split("_").str[1]

for ses in sorted(df_preterm["session"].unique()):
    print(ses)
    df_ses = df_preterm[df_preterm["session"] == ses]
    print(f"subjects total {len(df_ses.index.tolist())}")
    print(f"unique scanners {len(np.unique(df_ses["site"]))}")
    print(f"female {100*df_ses["sex"].isin([0]).sum()/len(df_ses["sex"]):.1f} percent")
    print(f"male {100*df_ses["sex"].isin([1]).sum()/len(df_ses["sex"]):.1f} percent")
    print(f"age mean {np.mean(df_ses["age"]):.1f}")
    print(f"age sd {np.std(df_ses["age"]):.1f}")
    
df_preterm.drop(columns = ["session"], inplace = True)
df_preterm["sub-id"] = df_preterm.index.str.split("_").str[0]
df_unique = df_preterm.groupby("sub-id").head(1)
print("Unique subjects")
print(f"subjects total {len(df_unique.index.tolist())}")
print(f"unique scanners {len(np.unique(df_unique["site"]))}")
print(f"female {100*df_unique["sex"].isin([0]).sum()/len(df_unique["sex"]):.1f} percent")
print(f"male {100*df_unique["sex"].isin([1]).sum()/len(df_unique["sex"]):.1f} percent")
df_preterm.drop(columns = ["sub-id"], inplace = True)

#%% Optional - only look at unique subjects
with open(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', f'Z_preterm_{preterm_thr}.pkl'), 'rb') as f:
    Z_preterm = pickle.load(f)
with open(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', 'Z_controls.pkl'), 'rb') as f:
    Z_controls = pickle.load(f)

Z_controls = Z_controls.sort_index()

Z_preterm["sub-id"] = Z_preterm.index.str.split("_").str[0]
Z_preterm_unique = Z_preterm.groupby("sub-id").head(1)

Z_controls["sub-id"] = Z_controls.index.str.split("_").str[0]
Z_controls["session"] = Z_controls.index.str.split("_").str[1]
Z_controls_unique = Z_controls.groupby("sub-id").head(1)

preterm_timepoints = Z_preterm_unique.index.str.split("_").str[1].value_counts()
preterm_all_timepoints = Z_preterm.index.str.split("_").str[1].value_counts()
#balance the unique controls for timepoints with the preterm
Z_controls_unique = (
    Z_controls_unique.groupby("session", group_keys = False)
    .apply( lambda g: g.sample(n=min(len(g), preterm_timepoints.get(g.name, 0)), random_state = 42))
    )
Z_controls_unique_alltime = (
    Z_controls.groupby("session", group_keys = False)
    .apply( lambda g: g.sample(n=min(len(g), preterm_all_timepoints.get(g.name, 0)), random_state = 42))
    )

Z_preterm.drop(columns = ["sub-id"], inplace = True)
Z_controls.drop(columns = ["sub-id", "session"], inplace = True)
Z_preterm_unique.drop(columns = ["sub-id"], inplace = True)
Z_controls_unique.drop(columns = ["sub-id", "session"], inplace = True)
Z_controls_unique_alltime.drop(columns = ["sub-id", "session"], inplace = True)

#check distribution
controls_timepoints = Z_controls_unique.index.str.split("_").str[1].value_counts()
print(preterm_timepoints)
print(controls_timepoints)
controls_timepoints = Z_controls_unique_alltime.index.str.split("_").str[1].value_counts()

#this si used later on for optional plotting of centiles showing preterms and controls
Z_controls_unique_alltime.index.to_series().to_csv(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', 'controls_subs.txt'), index = False, header = False)

#%% Optional - build comparison niftis across all timepoints for unique subjects
      
threshold = 2 
threshold_str = '2'

#abs Z
Z_preterm_unique_average = Z_preterm_unique.abs().astype(int).mean()
print(np.array(Z_preterm_unique_average).shape)
ptksave(np.array(Z_preterm_unique_average), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'Z_preterm_unique_abs_average.nii.gz'), example = ex_nii, mask = mask_nii)

Z_controls_unique_average = Z_controls_unique.abs().astype(int).mean()
print(np.array(Z_controls_unique_average).shape)
ptksave(np.array(Z_controls_unique_average), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'Z_controls_unique_abs_average.nii.gz'), example= ex_nii, mask = mask_nii)
    
Z_preterm_thresholded = (Z_preterm_unique.abs() >threshold).astype(int)
Z_preterm_percentages = Z_preterm_thresholded.sum()/len(Z_preterm_thresholded.index)
print(np.array(Z_preterm_percentages).shape)
ptksave(np.array(Z_preterm_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_unique_abs_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)

Z_controls_thresholded = (Z_controls_unique.abs() >threshold).astype(int)
Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
print(np.array(Z_controls_percentages).shape)
ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_unique_abs_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)

#positive and negative Z
       
Z_preterm_thresholded = (Z_preterm_unique >threshold).astype(int)
Z_preterm_percentages = Z_preterm_thresholded.sum()/len(Z_preterm_thresholded.index)
print(np.array(Z_preterm_percentages).shape)
ptksave(np.array(Z_preterm_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_unique_pos_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)

Z_controls_thresholded = (Z_controls_unique >threshold).astype(int)
Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
print(np.array(Z_controls_percentages).shape)
ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_unique_pos_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)


Z_preterm_thresholded = (Z_preterm_unique < -(threshold)).astype(int)
Z_preterm_percentages = Z_preterm_thresholded.sum()/len(Z_preterm_thresholded.index)
print(np.array(Z_preterm_percentages).shape)
ptksave(np.array(Z_preterm_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_unique_neg_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)

Z_controls_thresholded = (Z_controls_unique < -(threshold)).astype(int)
Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
print(np.array(Z_controls_percentages).shape)
ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_unique_neg_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)

#%% Generate voxelwise maps of extreme Zscore proportion in preterms and controls, across each separate timepoints 

Z_preterm["session"] = Z_preterm.index.str.split("_").str[1]
Z_controls["session"] = Z_controls.index.str.split("_").str[1]

for ses in sorted (Z_preterm["session"].unique()):
    Z_preterm_ses = Z_preterm[Z_preterm["session"] == ses]
            
    threshold = 2 
    threshold_str = '2'

    Z_controls_ses = Z_controls[Z_controls["session"] == ses]
    Z_controls_ses = Z_controls_ses.iloc[0:len(Z_preterm_ses.index.values),:] #match number of preterms for meaningful comparison
    
    Z_preterm_ses.drop(columns = [ "session"], inplace = True)
    Z_controls_ses.drop(columns = [ "session"], inplace = True)
    print(Z_preterm_ses.shape)
    print(Z_controls_ses.shape)
    
    #abs Z
    Z_preterm_average = Z_preterm_ses.abs().astype(int).mean()
    print(np.array(Z_preterm_average).shape)
    ptksave(np.array(Z_preterm_average), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_{ses}_abs_average.nii.gz'), example = ex_nii, mask = mask_nii)
    
    Z_controls_average = Z_controls_ses.abs().astype(int).mean()
    print(np.array(Z_controls_average).shape)
    ptksave(np.array(Z_controls_average), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_{ses}_abs_average.nii.gz'), example = ex_nii, mask = mask_nii)
          
    Z_preterm_thresholded = (Z_preterm_ses.abs() >threshold).astype(int)
    Z_preterm_percentages = Z_preterm_thresholded.sum()/len(Z_preterm_thresholded.index)
    print(np.array(Z_preterm_percentages).shape)
    ptksave(np.array(Z_preterm_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_{ses}_abs_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
    
    Z_controls_thresholded = (Z_controls_ses.abs() >threshold).astype(int)
    Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
    print(np.array(Z_controls_percentages).shape)
    ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_{ses}_abs_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
    
    #positive and negative Z
    
    Z_preterm_thresholded = (Z_preterm_ses >threshold).astype(int)
    Z_preterm_percentages = Z_preterm_thresholded.sum()/len(Z_preterm_thresholded.index)
    print(np.array(Z_preterm_percentages).shape)
    ptksave(np.array(Z_preterm_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_{ses}_pos_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
    
    Z_controls_thresholded = (Z_controls_ses >threshold).astype(int)
    Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
    print(np.array(Z_controls_percentages).shape)
    ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_{ses}_pos_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
    
    
    Z_preterm_thresholded = (Z_preterm_ses < -(threshold)).astype(int)
    Z_preterm_percentages = Z_preterm_thresholded.sum()/len(Z_preterm_thresholded.index)
    print(np.array(Z_preterm_percentages).shape)
    ptksave(np.array(Z_preterm_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_preterm_{ses}_neg_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
    
    Z_controls_thresholded = (Z_controls_ses < -(threshold)).astype(int)
    Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
    print(np.array(Z_controls_percentages).shape)
    ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', f'Z_controls_{ses}_neg_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)

#random sampling of controls five times to ensure stable results
if not os.path.exists(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals','more_control_Zmaps')):
    os.makedirs(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals','more_control_Zmaps'))
    
for i in range(1,6):
    for ses in sorted (Z_preterm["session"].unique()):
                
        threshold = 2 
        threshold_str = '2'
        
        Z_preterm_ses = Z_preterm[Z_preterm["session"] == ses]
        
        Z_controls_ses = Z_controls[Z_controls["session"] == ses]
        Z_controls_ses = Z_controls_ses.iloc[i*len(Z_preterm_ses.index.values):(i+1)*len(Z_preterm_ses.index.values),:] #match number of preterms for meaningful comparison
        
        Z_controls_ses.drop(columns = [ "session"], inplace = True)
        print(Z_controls_ses.shape)
        
        #abs Z per ses
        Z_controls_average = Z_controls_ses.abs().astype(int).mean()
        print(np.array(Z_controls_average).shape)
        ptksave(np.array(Z_controls_average), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals','more_control_Zmaps', f'Z_controls_{ses}_iter{i}_abs_average.nii.gz'), example = ex_nii, mask = mask_nii)
              
        Z_controls_thresholded = (Z_controls_ses.abs() >threshold).astype(int)
        Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
        print(np.array(Z_controls_percentages).shape)
        ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'more_control_Zmaps',f'Z_controls_{ses}_iter{i}_abs_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
        
        #directed Z
    
        Z_controls_thresholded = (Z_controls_ses >threshold).astype(int)
        Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
        print(np.array(Z_controls_percentages).shape)
        ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals','more_control_Zmaps', f'Z_controls_{ses}_iter{i}_pos_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
        
        Z_controls_thresholded = (Z_controls_ses < -(threshold)).astype(int)
        Z_controls_percentages = Z_controls_thresholded.sum()/len(Z_controls_thresholded.index)
        print(np.array(Z_controls_percentages).shape)
        ptksave(np.array(Z_controls_percentages), os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals','more_control_Zmaps', f'Z_controls_{ses}_iter{i}_neg_thresholded{threshold_str}_percentages.nii.gz'), example= ex_nii, mask = mask_nii)
    

Z_preterm.drop(columns = [ "session"], inplace = True)
Z_controls.drop(columns = [ "session"], inplace = True)

#%%whole-brain burden of extreme zscores group test

n_voxels = len(Z_preterm.columns)#get total voxelsagain just in case
Z_preterm["session"] = Z_preterm.index.str.split("_").str[1]
Z_controls["session"] = Z_controls.index.str.split("_").str[1]

raw_pvals=[]

threshold = 2
threshold_str = '2'

for ses in sorted (Z_preterm["session"].unique()):

    Z_preterm_ses = Z_preterm[Z_preterm["session"] == ses]
    Z_controls_ses = Z_controls[Z_controls["session"] == ses]
    Z_preterm_ses.drop(columns = [ "session"], inplace = True)
    Z_controls_ses.drop(columns = [ "session"], inplace = True)
    print(Z_preterm_ses.shape)
    print(Z_controls_ses.shape)
    
    extreme_neg_preterm = (Z_preterm_ses.to_numpy() < -threshold).sum(axis=1)
    extreme_pos_preterm = (Z_preterm_ses.to_numpy() > threshold).sum(axis=1)
    extreme_neg_controls = (Z_controls_ses.to_numpy() < -threshold).sum(axis=1)
    extreme_pos_controls = (Z_controls_ses.to_numpy() > threshold).sum(axis=1)
    
    df_preterm = pd.DataFrame({
        "group": 1,
        "extreme_neg": extreme_neg_preterm,
        "extreme_pos": extreme_pos_preterm,
        "n_voxels": n_voxels
    })
    
    df_controls = pd.DataFrame({
        "group": 0,
        "extreme_neg": extreme_neg_controls,
        "extreme_pos": extreme_pos_controls,
        "n_voxels": n_voxels
    })
    

    df = pd.concat([df_preterm, df_controls], ignore_index=True)
    df["percent_neg"] = 100 * df["extreme_neg"] / df["n_voxels"]
    df["percent_pos"] = 100 * df["extreme_pos"] / df["n_voxels"]
    
    print(df.groupby("group")[["percent_neg", "percent_pos"]].mean())
    
    #test
    neg_patients = df[df["group"] == 1]["percent_neg"]
    neg_controls = df[df["group"] == 0]["percent_neg"]
    
    _, p_neg = mannwhitneyu(neg_patients,neg_controls,alternative="two-sided")
    
    pos_patients = df[df["group"] == 1]["percent_pos"]
    pos_controls = df[df["group"] == 0]["percent_pos"]
    
    _, p_pos = mannwhitneyu(pos_patients,pos_controls,alternative="two-sided")

    raw_pvals.append(p_neg)
    raw_pvals.append(p_pos)

# Apply Holm correction
reject, pvals_corr, _, _ = multipletests(raw_pvals, alpha=0.05, method="holm")

print("\nHolm-corrected p-values (6-test correction):")
for i, ses in enumerate(sorted (Z_preterm["session"].unique())):
    p_neg_corr = pvals_corr[i*2]     # negative extremes
    p_pos_corr = pvals_corr[i*2 + 1] # positive extremes
    print(f"Session {ses} — Negative extremes Holm-corrected p = {p_neg_corr:.6f}")
    print(f"Session {ses} — Positive extremes Holm-corrected p = {p_pos_corr:.6f}")
    
Z_preterm.drop(columns = [ "session"], inplace = True)
Z_controls.drop(columns = [ "session"], inplace = True)

#%% mean effect TFCE test between preterm and controls - this does not need many cores but can be greedy in RAM and time

if not os.path.exists(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test')):
    os.makedirs(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test'))

with open(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', f'Z_preterm_{preterm_thr}.pkl'), 'rb') as f:
    Z_preterm = pickle.load(f)
with open(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', 'Z_controls.pkl'), 'rb') as f:
    Z_controls = pickle.load(f)

Z_preterm["session"] = Z_preterm.index.str.split("_").str[1]
Z_controls["session"] = Z_controls.index.str.split("_").str[1]

masker = NiftiMasker(mask_img=mask_nii).fit()
mask_img = nib.load(mask_nii)
mask_idx = mask_img.get_fdata().astype(bool)

for ses in sorted(Z_preterm["session"].unique()):
    Z_preterm_ses = Z_preterm[Z_preterm["session"] == ses]
    Z_controls_ses = Z_controls[Z_controls["session"] == ses]
    Z_controls_ses = Z_controls_ses.iloc[:10*len(Z_preterm_ses.index.values),:] #large RAM requirement here
    
    Z_preterm_ses.drop(columns = [ "session"], inplace = True)
    Z_controls_ses.drop(columns = [ "session"], inplace = True)
    print(Z_preterm_ses.shape)
    print(Z_controls_ses.shape)
    
    n_pre = len(Z_preterm_ses)
    n_con = len(Z_controls_ses)
    Y = np.vstack([Z_preterm_ses.values, Z_controls_ses.values]).astype(float)
    
    #Preterm > controls
    group = np.r_[np.ones(n_pre), np.zeros(n_con)]
    group= group.reshape(len(group), 1)
    
    tfce_dict = permuted_ols(
        tested_vars=group,
        target_vars=Y,
        masker = masker,
        confounding_vars = None, #we've already accounted for these with normative modeling
        model_intercept = True,
        n_perm=5000,
        two_sided_test=False,
        tfce=True,
        verbose = 5,
        random_state = 42,
        n_jobs=3, #not many allowed given large sample size - easily crashes - thus the need to reduce memory pressure of temp files
    )
    
    tfce_stat = tfce_dict["tfce"]  # raw TFCE-enhanced t map
    tfce_logp = tfce_dict["logp_max_tfce"] #this is FWE corrected by default
    tfce_pvals = 10 ** (-tfce_dict["logp_max_tfce"])
    
    #save outputs 
    with open(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_stat_{ses}_preterm>controls_zscores_test.pkl'), 'wb') as f:
        pickle.dump(tfce_stat, f)
    with open(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_-logpval_{ses}_preterm>controls_zscores_test.pkl'), 'wb') as f:
        pickle.dump(tfce_logp, f)
    with open(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_{ses}_preterm>controls_zscores_test.pkl'), 'wb') as f:
        pickle.dump(tfce_pvals, f)
    
    ptksave(tfce_stat, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_stat_{ses}_preterm>controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    ptksave(tfce_logp, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_-logpval_{ses}_preterm>controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    ptksave(tfce_pvals, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_{ses}_preterm>controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    tfce_sig_05 = (tfce_pvals < 0.05).astype(int)
    ptksave(tfce_sig_05, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_thresh_5e-3_{ses}_preterm>controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    tfce_sig_01 = (tfce_pvals < 0.01).astype(int)
    ptksave(tfce_sig_01, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_thresh_1e-3_{ses}_preterm>controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
   
    #Preterm < controls
    group = np.r_[np.zeros(n_pre), np.ones(n_con)]
    group= group.reshape(len(group), 1)
    
    tfce_dict = permuted_ols(
        tested_vars=group,
        target_vars=Y,
        masker = masker,
        confounding_vars = None,
        model_intercept = True,
        n_perm=5000,
        two_sided_test=False,
        tfce=True,
        verbose = 5,
        random_state = 42,
        n_jobs=3,
    )
    
    tfce_stat = tfce_dict["tfce"]
    tfce_logp = tfce_dict["logp_max_tfce"] 
    tfce_pvals = 10 ** (-tfce_dict["logp_max_tfce"])
    
    #wsave outputs
    with open(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_stat_{ses}_preterm<controls_zscores_test.pkl'), 'wb') as f:
        pickle.dump(tfce_stat, f)
    with open(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_-logpval_{ses}_preterm<controls_zscores_test.pkl'), 'wb') as f:
        pickle.dump(tfce_logp, f)
    with open(os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_{ses}_preterm<controls_zscores_test.pkl'), 'wb') as f:
        pickle.dump(tfce_pvals, f)
    
    ptksave(tfce_stat, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_stat_{ses}_preterm<controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    ptksave(tfce_logp, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_-logpval_{ses}_preterm<controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    ptksave(tfce_pvals, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_{ses}_preterm<controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    tfce_sig_05 = (tfce_pvals < 0.05).astype(int)
    ptksave(tfce_sig_05, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_thresh_5e-3_{ses}_preterm<controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)
    tfce_sig_01 = (tfce_pvals < 0.01).astype(int)
    ptksave(tfce_sig_01, os.path.join(w_dir,f'preterm_{preterm_thr}_comparison', 'visuals', 'mean_test', f'TFCE_pval_thresh_1e-3_{ses}_preterm<controls_zscores.nii.gz'), example= ex_nii, mask = mask_nii)

Z_preterm.drop(columns = [ "session"], inplace = True)
Z_controls.drop(columns = [ "session"], inplace = True)

#%% Optional - prepare data for centile plots with preterms
if not os.path.exists(os.path.join(w_dir,'visuals','for_figures')):
    os.makedirs(os.path.join(w_dir,'visuals','for_figures'))
save_dir = (os.path.join(w_dir,'visuals','for_figures'))

timepoint = '_BL' #ABCD timepoint we want to plot - '_BL', '_2Y' or '_4Y'
true_modality = 'mod_gmv' #used only if modality == 'mod_gmv_and_wmv'- need to manually specify here because they live in different folders

if modality == 'mod_gmv_and_wmv' and true_modality == 'mod_gmv':
    w_dir = w_dir_gm
if modality == 'mod_gmv_and_wmv' and true_modality == 'mod_wmv':
    w_dir = w_dir_wm

idxvol = dataio.fileio.load(os.path.join(w_dir,'visuals','idx.nii.gz'), mask=mask_nii, vol=True)

#Choose voxel of interest - specify 2mm coords in fsleyes
# vox_coord = (55,41,23,0)
# vox_coord = (36,41,27,0)
# vox_coord = (35,57,32,0)
# vox_coord = (49,55,22,0)
# vox_coord = (35,59,46,0)
# vox_coord = (41,39,30,0)
# vox_coord =(46,51,48,0)
# vox_coord = (44,55,32,0)
# vox_coord = (77,59,32,0)
# vox_coord = (17,55,33,0)
# vox_coord = (27,59,25,0)
vox_coord = (52,44,32,0)
vox_id = int(idxvol[vox_coord])

# find batch id for the voxel
batch_num, mod_num = divmod(vox_id, vox_batch_size)
print (f'batch_{batch_num}_response_var_{vox_id}')

#regenerate the test set
g_age=pd.read_csv('/project_cephfs/3022017.06/ABCD/phenotypes/ABCDstudyNDA_AnnualRelease4.0/Package_1199073/abcd_devhxss01.txt', sep = '\t')
g_age = g_age.iloc[1:]
g_age.rename(columns={'subjectkey': 'sub_id','devhx_ss_12_p':'gestational_age'}, inplace=True)
g_age = g_age[['sub_id','gestational_age']]
g_age = g_age.dropna()
g_age['sub_id'] = g_age['sub_id'].str.replace('_', '', regex=False) #correspondence to sub foldernames
g_age['sub_id'] = 'sub-'+ g_age['sub_id']
g_age['gestational_age'] = 40-g_age['gestational_age'].astype(int) # convert to actual GA
g_age = g_age[g_age.gestational_age > 0] # drop the 'uncertain' 999 values
preterm = g_age[g_age['gestational_age']<=preterm_thr]
preterm.reset_index(drop=True, inplace = True)
clin_subs = preterm.iloc[:,0].values
ses_suffixes = ['_BL', '_2Y', '_4Y'] #add all possible observations for these subjects because longitudinal
clin_subs =np.char.add(np.repeat(clin_subs, len(ses_suffixes)), np.tile(ses_suffixes, len(clin_subs)))

#we stored sub-ids of unique controls earlier
control_subs = pd.read_csv(os.path.join(w_dir, f'preterm_{preterm_thr}_comparison', 'controls_subs.txt'), header = None)

with open(os.path.join(w_dir, f'batch_{batch_num}', 'norm_data.pkl'), 'rb') as f:
    norm_data= pickle.load(f)

#only take clin_subs of one timepoint for plotting
clin_subs = clin_subs[np.char.endswith(np.array(clin_subs).astype(str) , timepoint)]    
control_subs= control_subs[np.char.endswith(np.array(control_subs).astype(str) , timepoint)]    

#get the normdata subsets to haev voxel values
def extract_normdata_subsets(all_clin_subs, control_subs, norm_data):

    clin_subs_existing = [s for s in clin_subs if s in norm_data.subject_ids.values]
    clin_subset = norm_data.where(norm_data.subject_ids.isin(clin_subs_existing),drop = True)
    # remaining = norm_data.where(~norm_data.subject_ids.isin(clin_subs_existing), drop=True)
    control_subset = norm_data.where(norm_data.subject_ids.isin(control_subs),drop = True)
    
    #re-make subsets into proper norm_data objects to get subclass methods
    clin_subset = NormData(data_vars=clin_subset.data_vars,coords=clin_subset.coords,
        attrs=clin_subset.attrs,name=clin_subset.name)
    control_subset = NormData(data_vars=control_subset.data_vars,coords=control_subset.coords,
        attrs=control_subset.attrs,name=control_subset.name)

    return clin_subset, control_subset

clin_subset, control_subset = extract_normdata_subsets(clin_subs, control_subs, norm_data)

#we also need the model for the plot
model= NormativeModel.load(os.path.join(w_dir, f'batch_{batch_num}'))

#select a single voxel to avoid plotting the whole batch
response_var = f'voxel_{vox_id}'
def select_for_response_var(model: NormativeModel, data: NormData, response_var: str):
    if response_var not in model.regression_models:
        raise ValueError(f"Response variable '{response_var}' not in model.")
    new_model = copy.deepcopy(model)
    new_model.response_vars = [response_var]
    new_model.regression_models = {response_var: new_model.regression_models[response_var]}
    new_model.outscalers = {response_var: new_model.outscalers[response_var]} if new_model.outscalers else {}
    
    new_data = data.sel({"response_vars": response_var}, drop=False)
    new_data["Y"] = new_data.Y.expand_dims("response_vars", axis=-1) #preserves 1d of norm_data.Y because single voxel, otherwise plot will error
    new_data = new_data.assign_coords(response_vars=[response_var])
    
    return new_model, new_data

modelvox, clinvox = select_for_response_var(model, clin_subset, response_var)
_, controlvox = select_for_response_var(model, control_subset, response_var)

#%% Custom plotting function slightly adapted from plot_centiles_advanced() of the pcntoolkit 
#age range is hardcoded to be able to show abcd subjects in more detail

from typing import Any, Dict, List, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import seaborn as sns  
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches


def plot_centiles_custom(
    model: "NormativeModel",
    centiles: List[float] | np.ndarray | None = None,
    conditionals: List[float] | np.ndarray | None = None,
    covariate: str | None = None,
    covariate_range: tuple[float, float] = (None, None),  # type: ignore
    batch_effects: Dict[str, List[str]] | None | Literal["all"] = None,
    scatter_data: NormData | None = None,
    overlay_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_thrivelines: bool = False,
    z_thrive: float = 0.0,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    plt_kwargs: dict | None = None,
    color1: str | None = None,
    color2: str | None = None,
    plotname = False,
    **kwargs: Any,
) -> None:
    """Generate centile plots for response variables with optional data overlay.

    This function creates visualization of centile curves for all response variables
    in the dataset. It can optionally show the actual data points overlaid on the
    centile curves, with customizable styling based on categorical variables.

    Parameters
    ----------
    model: NormativeModel
        The model to plot the centiles for.
    centiles: List[float] | np.ndarray | None, optional
        The centiles to plot. If None, the default centiles will be used.
    conditionals: List[float] | np.ndarray | None, optional
        A list of x-coordinates for which to plot the conditionals
    covariate: str | None, optional
        The covariate to plot on the x-axis. If None, the first covariate in the model will be used.
    covariate_range: tuple[float, float], optional
        The range of the covariate to plot on the x-axis. If None, the range of the covariate that was in the train data will be used.
    batch_effects: Dict[str, List[str]] | None | Literal["all"], optional
        The batch effects to plot the centiles for. If None, the batch effect that appears first in alphabetical order will be used.
    scatter_data: NormData | None, optional
        Data to scatter on top of the centiles.
    harmonize_data: bool, optional
        Whether to harmonize the scatter data before plotting. Data will be harmonized to the batch effect for which the centiles were computed.
    hue_data: str, optional
        The column to use for color coding the data. If None, the data will not be color coded.
    markers_data: str, optional
        The column to use for marker styling the data. If None, the data will not be marker styled.
    show_other_data: bool, optional
        Whether to scatter data belonging to groups not in batch_effects.
    save_dir: str | None, optional
        The directory to save the plot to. If None, the plot will not be saved.
    show_centile_labels: bool, optional
        Whether to show the centile labels on the plot.
    show_legend: bool, optional
        Whether to show the legend on the plot.
    plt_kwargs: dict, optional
        Additional keyword arguments for the plot.
    **kwargs: Any, optional
        Additional keyword arguments for the model.compute_centiles method.

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """
    if covariate is None:
        covariate = model.covariates[0]
        assert isinstance(covariate, str)

    cov_min = covariate_range[0] or model.covariate_ranges[covariate]["min"]
    cov_max = covariate_range[1] or model.covariate_ranges[covariate]["max"]
    covariate_range = (cov_min, cov_max)
    covariate_range = (7, 16)

    if batch_effects == "all":
        if scatter_data:
            batch_effects = scatter_data.unique_batch_effects
        else:
            batch_effects = model.unique_batch_effects
    elif batch_effects is None:
        if scatter_data:
            batch_effects = {k: [v[0]] for k, v in scatter_data.unique_batch_effects.items()}
        else:
            batch_effects = {k: [v[0]] for k, v in model.unique_batch_effects.items()}

    if plt_kwargs is None:
        plt_kwargs = {}

    centile_covariates = np.linspace(covariate_range[0], covariate_range[1], 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})

    for cov in model.covariates:
        if cov != covariate:
            minc = model.covariate_ranges[cov]["min"]
            maxc = model.covariate_ranges[cov]["max"]
            centile_df[cov] = (minc + maxc) / 2

    # Batch effects are the first ones in the highlighted batch effects
    for be, v in batch_effects.items():
        centile_df[be] = v[0]
    # Response vars are all 0, we don't need them
    for rv in model.response_vars:
        centile_df[rv] = 0
    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=model.response_vars,
        batch_effects=list(batch_effects.keys()),
    )  # type:ignore

    conditionals_data = []
    if conditionals is not None:
        for c in conditionals:
            # Compute the endpoints of the conditional curve (0.01th and 0.99th centile)
            centile = copy.deepcopy(centile_data).isel(observations=[0, 1])
            centile.X.loc[{"covariates": covariate}] = c
            model.compute_centiles(centile, centiles=[0.01, 0.99])

            # Compute the curve in between the endpoints
            conditional_d = copy.deepcopy(centile_data)
            conditional_d.X.loc[{"covariates": covariate}] = c
            for rv in model.response_vars:
                conditional_d.Y.loc[{"response_vars": rv}] = np.linspace(
                    *(centile.centiles.sel(observations=0, response_vars=rv).values.tolist()), 150
                )
            if not hasattr(conditional_d, "logp"):
                model.compute_logp(conditional_d)
            conditionals_data.append(conditional_d)

    if not hasattr(centile_data, "centiles"):
        model.compute_centiles(centile_data, centiles=centiles, **kwargs)
    if scatter_data and show_thrivelines:
        model.compute_thrivelines(scatter_data, z_thrive=z_thrive)
    if show_yhat and not hasattr(centile_data, "yhat"):
        model.compute_yhat(centile_data)

    if not model.has_batch_effect:
        batch_effects = {}

    if harmonize_data and scatter_data:
        if model.has_batch_effect:
            reference_batch_effect = {k: v[0] for k, v in batch_effects.items()}
            model.harmonize(scatter_data, reference_batch_effect=reference_batch_effect)
            model.harmonize(overlay_data, reference_batch_effect=reference_batch_effect)
        else:
            model.harmonize(scatter_data)
            model.harmonize(overlay_data)

    for response_var in model.response_vars:
        _plot_centiles(
            centile_data=centile_data,
            response_var=response_var,
            covariate=covariate,
            conditionals_data=conditionals_data,
            batch_effects=batch_effects,
            scatter_data=scatter_data,
            overlay_data=overlay_data,
            harmonize_data=harmonize_data,
            hue_data=hue_data,
            markers_data=markers_data,
            show_other_data=show_other_data,
            show_thrivelines=show_thrivelines,
            save_dir=save_dir,
            show_centile_labels=show_centile_labels,
            show_legend=show_legend,
            show_yhat=show_yhat,          
            color1 = color1,
            color2 = color2,
            plotname=plotname,
            plt_kwargs=plt_kwargs,
  
        )

def _plot_centiles(
    centile_data: NormData,
    response_var: str,
    covariate: str = None,  # type: ignore
    conditionals_data: List[NormData] | None = None,
    batch_effects: Dict[str, List[str]] = None,  # type: ignore
    scatter_data: NormData | None = None,
    overlay_data: NormData | None = None,
    harmonize_data: bool = True,
    hue_data: str = "site",
    markers_data: str = "sex",
    show_other_data: bool = False,
    show_thrivelines: bool = False,
    save_dir: str | None = None,
    show_centile_labels: bool = True,
    show_legend: bool = True,
    show_yhat: bool = False,
    color1: str | None = None,
    color2: str | None = None,
    plotname = False,
    plt_kwargs: dict = None,  # type: ignore
) -> None:
    sns.set_style("whitegrid")
    plt.figure(**plt_kwargs)

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = centile_data.sel(filter_dict)

    for centile in centile_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean == 0:
            thickness = 2
        else:
            thickness = 1
        if d_mean <= 0.25:
            style = "-"

        elif d_mean <= 0.475:
            style = "--"
        else:
            style = ":"

        sns.lineplot(
            x=filtered.X,
            y=filtered.centiles.sel(centile=centile),
            color="black",
            linestyle=style,
            linewidth=thickness,
            zorder=3,
            legend="brief",
        )

        font = FontProperties()
        font.set_style("italic") 
        font.set_size(10)
        if show_centile_labels:
            plt.text(
                s=f"{centile.item()*100:.0f}%",
                x=filtered.X[-1]+0.1,
                y=filtered.centiles.sel(centile=centile)[-1],
                color="black",
                horizontalalignment="left",
                verticalalignment="center",
                fontproperties=font,
            )
    if show_yhat:
        plt.plot(filtered.X, filtered.Yhat, color="red", linestyle="--", linewidth=thickness, zorder=2, label="$\\hat{Y}$")

    minx, maxx = plt.xlim()
    plt.xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))
    plt.xlim(7,16)
    
    if scatter_data:
        scatter_filter = scatter_data.sel(filter_dict)
        df = scatter_filter.to_dataframe()
        scatter_data_name = "Y_harmonized" if harmonize_data else "Y"
        thriveline_data_name = "thrive_Y_harmonized" if harmonize_data else "thrive_Y"
        columns = [("X", covariate), (scatter_data_name, response_var)]
        columns.extend([("batch_effects", be.item()) for be in scatter_data.batch_effect_dims])
        df = df[columns]
        df.columns = [c[1] for c in df.columns]
        if batch_effects == {}:
            sns.scatterplot(
                df,
                x=covariate,
                y=response_var,
                label=scatter_data.name,
                color="black",
                s=20,
                alpha=0.6,
                zorder=1,
                linewidth=0,
            )
            if show_thrivelines:
                plt.plot(scatter_filter.thrive_X.to_numpy().T, scatter_filter[thriveline_data_name].to_numpy().T)
        else:
            idx = np.full(len(df), True)
            for j in batch_effects:
                idx = np.logical_and(
                    idx,
                    df[j].isin(batch_effects[j]),
                )
            be_df = df[idx]
            scatter = sns.scatterplot(
                data=be_df,
                x=covariate,
                y=response_var,
                hue=hue_data if hue_data in df else None,
                palette = [color2], #just for patient-only plots
                # style=markers_data if markers_data in df else None,
                style = None,
                s=15,
                alpha=0.6,
                zorder=2,
                linewidth=0,
            )
            # plt.ylim(-1.5, 0.1)
            if overlay_data:
                overlay_filter = overlay_data.sel(filter_dict)
                overlay_df = overlay_filter.to_dataframe()
                overlay_data_name = "Y_harmonized" if harmonize_data else "Y"
                columns = [("X", covariate), (overlay_data_name, response_var)]
                columns.extend([("batch_effects", be.item()) for be in overlay_data.batch_effect_dims])
                overlay_df = overlay_df[columns]
                overlay_df.columns = [c[1] for c in overlay_df.columns]
                sns.scatterplot(
                    data=overlay_df,
                    x=covariate,
                    y=response_var,
                    hue=hue_data if hue_data in overlay_df else None,
                    palette = [color1], #just for patient-only plots
                    style = None,
                    s=15,
                    alpha=0.5,
                    zorder=1,
                    linewidth=0,
                )
            if show_thrivelines:
                plt.plot(scatter_filter.thrive_X.to_numpy().T, scatter_filter[thriveline_data_name].to_numpy().T)

            if show_other_data:
                non_be_df = df[~idx]
                markers = ["Other data"] * len(non_be_df)
                sns.scatterplot(
                    data=non_be_df,
                    x=covariate,
                    y=response_var,
                    color="black",
                    style=markers,
                    linewidth=0,
                    s=20,
                    alpha=0.4,
                    zorder=0,
                )

            if show_legend:
                legend = scatter.get_legend()
                if legend:
                    handles = legend.legend_handles
                    labels = [t.get_text() for t in legend.get_texts()]
                    plt.legend(
                        handles,
                        labels,
                        title_fontsize=10,
                    )
            else:
                plt.legend().remove()
    plt.legend(handles=[
    mpatches.Patch(color=color2, label='Preterm'),
    mpatches.Patch(color=color1, label='Controls'),
    ]  , loc='upper left', fontsize = 8)
            
    title = f"Centiles of {response_var}"
    if scatter_data:
        if harmonize_data:
            plotname = f"centiles_{response_var}_{scatter_data.name}{plotname}_preterm_comparison_harmonized"
            title = f"{title}\n With harmonized {scatter_data.name} data"
        else:
            plotname = f"centiles_{response_var}_{scatter_data.name}{plotname}_preterm_comparison"
            title = f"{title}\n With raw {scatter_data.name} data"

    if conditionals_data:
        for conditional_d in conditionals_data:
            filter_cond = conditional_d.sel(filter_dict)
            plt.plot(
                np.exp(filter_cond.logp.values) * 10 + filter_cond.X,
                filter_cond.Y,
                color="blue",
                linestyle="--",
                linewidth=1,
                zorder=3,
                label="Conditional",
            )
            # Put a text annotation on top of the plot, rotate the text 90 degrees
            plt.text(
                filter_cond.X[-1],
                filter_cond.Y[-1],
                f"{filter_cond.X[-1].values.item():.2f}",
                color="black",
                fontsize=10,
                ha="right",
                va="bottom",
                rotation=-90,
            )

    plt.title(title)
    plt.xlabel(covariate)
    plt.ylabel(response_var)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{plotname}.png"), dpi=300)
    else:
        plt.show(block=False)
    plt.close()


#%% now plot

save_dir = os.path.join(root_dir,'for_figures')

plot_centiles_custom( 
    modelvox,
    centiles=[0.01,0.05, 0.5, 0.95,0.99],  
    scatter_data=clinvox,
    overlay_data = controlvox,
    batch_effects = "all",
    show_other_data=True, 
    hue_data = 'site', 
    harmonize_data=True,
    show_legend = False,
    show_centile_labels = True,
    plotname = timepoint,
    color1 ='darkblue',
    color2 = '#0f9b8e',
    # save_dir= save_dir #commented out to show plot in IDE instead of saving it
)

