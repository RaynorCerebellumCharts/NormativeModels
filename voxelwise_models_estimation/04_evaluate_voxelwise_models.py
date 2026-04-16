#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 9 12:30:17 2025

@author Alice Chavanne

This script collects evaluation metrics for the newly estimated normative model already computed by the pcntoolkit, 
and additionally computes skew and kurstosis. All metrics are voxelwise and projected back onto nifti space. 
Also contains sanity checks and plotting (centiles, qq plots).
"""

import os
import numpy as np
import pandas as pd
import pcntoolkit.dataio.fileio as fileio
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from pcntoolkit import NormData, NormativeModel, plot_centiles, plot_centiles_advanced, plot_qq
import pickle
import matplotlib.pyplot as plt
import re
import copy
from sklearn.model_selection import train_test_split as skl_split
import collections


modality = 'log_jacs'#choose brain measure ('log_jacs', 'mod_gmv', 'mod_wmv')
prefix = 'BLR_aff_nonlin_'#prefix for model name/directory
suffix = '_wholebrain_2mm_14' #suffix same 

name = '58539_subjects'# must match the name given to the normdata objects when they were built

root_dir = '/root/dir/'
tpl_dir = os.path.join(root_dir,'tpl-MNI152NLin2009cSym')
proc_dir = os.path.join(root_dir,'models', prefix + modality + suffix )
w_dir = os.path.join(proc_dir,'models')
demographics_dir = '/demographics/dir/' # optional - directory where clinical subjects sub IDs are (e.g. ABCD preterm) used for plotting

if modality == 'log_jacs':
    mask_nii = os.path.join(tpl_dir, 'tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
if modality == 'mod_gmv':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg0.5.nii.gz')
if modality == 'mod_wmv':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg0.5.nii.gz')

vox_batch_size = 150 #must be the same value as the script to prepare normdata 
ex_nii = mask_nii #will be used to project evaluation metrics back into nifti space

#extract number of batches again
mask = ptkload(mask_nii, mask=mask_nii, vol=False).T
n_voxels = len(mask)
n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)

#where the separated evaluation metrics will be stored
if not os.path.exists(os.path.join(w_dir, 'visuals')):
    os.makedirs(os.path.join(w_dir, 'visuals'))
if not os.path.exists(os.path.join(w_dir, 'global_metrics')):
    os.makedirs(os.path.join(w_dir, 'global_metrics'))


#%% collect stats from batches into global metrics and niftis

def extract_num(col_name): #ensure the ordering of column names (voxel_x) inside each batch csv - if run with the runner, they might be disordered
    match = re.search(r'voxel_(\d+)', col_name) 
    return int(match.group(1)) if match else float('inf')  # handle non-matching names safely

collect_metrics = []
batch_metrics = []

#extract and concantenate evaluation metrics on test set for all voxels
for b in range(n_batches)[:]:
    batch = f'batch_{b}'
    print(batch)
    
    #extracting directly from csv here is faster than loading back in norm_data but needs more careful checking
    datapath = os.path.join(w_dir, batch, 'results', 'statistics_'+ name +'_test_+_abcd_subset_test_+_'+ name + '.csv')
    data = pd.read_csv(datapath)
    data.set_index("statistic", inplace = True)
  
    sorted_cols = sorted(data.columns, key=extract_num)#reorder columns
    data = data[sorted_cols]
    
    data.columns = [f'{batch}_voxel_{i}' for i in range(len(data.columns))] #incorporate batch number in column name just in case
    batch_metrics.append(data)
    
collect_metrics = pd.concat(batch_metrics, axis = 1)

#now save the metrics separately as pkl and nifti
for metric in collect_metrics.index:
    stat_data= np.array(collect_metrics.loc[metric])
    
    with open(os.path.join(w_dir, 'global_metrics', metric+'_test.pkl'), 'wb') as f:
        pickle.dump(stat_data, f)
    
    ptksave(stat_data, os.path.join(w_dir, 'visuals', metric+'_test.nii.gz'), example=ex_nii, mask=mask_nii)


#we also save a simple nifti with voxel indices, to use it later to map between fsl coordinates and estimated model for a given voxel   
ptksave(np.arange(len(stat_data)), os.path.join(w_dir,'visuals','idx.nii.gz'), example=ex_nii, mask=mask_nii, dtype='uint32')

#%% compute kurtosis and skew and also put them into niftis - this needs some more RAM

def extract_num(col_name): #handling the oredering of column names between batches
    match = re.search(r'voxel_(\d+)', col_name)
    return int(match.group(1)) if match else float('inf') 

skew_all = []
kurtosis_all = []

for b in range(n_batches):
    batch = f'batch_{b}'
    print(batch)
    
    #extract Z-scores - it's faster to do it using csv than re-loading inside norm_data objects, but then we need to be careful about ordering/column names
    Z_path = os.path.join(w_dir, batch, 'results', 'Z_'+ name +'_test_+_abcd_subset_test_+_'+ name + '.csv')
    Z = pd.read_csv(Z_path)
    Z = Z.drop(columns='observations')
    sorted_cols = sorted(Z.columns, key=extract_num)
    Z = Z[sorted_cols]
    
    Z[np.isnan(Z)] = 0
    Z[np.isinf(Z)] = 0
    
    #compute skewness and kurtosis
    n = np.shape(Z)[0]
    m1 = np.mean(Z, axis=0).astype(np.float64)
    m3 = np.zeros_like(m1)
    m4 = np.zeros_like(m1)
    
    diff = Z - m1
    m3 += np.sum(diff**3, axis=0)
    m4 += np.sum(diff**4, axis=0)
    
    s1 = np.std(Z, axis=0)
    skew = n*m3/(n-1)/(n-2)/s1**3
    kurtosis = (n * (n+1) * m4) / ((n-1) * (n-2) * (n-3) * s1**4) - (3 * (n-1)**2) / ((n-2) * (n-3))
    
    skew_all.extend(skew)
    kurtosis_all.extend(kurtosis)
    
with open(os.path.join(w_dir, 'global_metrics', 'skew_test.pkl'), 'wb') as f:
    pickle.dump(np.array(skew_all), f)
with open(os.path.join(w_dir, 'global_metrics', 'kurtosis_test.pkl'), 'wb') as f:
    pickle.dump(np.array(kurtosis_all), f)

ptksave(np.array(skew_all), os.path.join(w_dir,'visuals','skew_test.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(np.array(kurtosis_all), os.path.join(w_dir,'visuals','kurtosis_test.nii.gz'), example=ex_nii, mask=mask_nii)


#%% some histograms of metrics across all voxels

with open(os.path.join(w_dir, 'global_metrics', 'kurtosis_test.pkl'), 'rb') as f:
    kurtosis = pickle.load(f)
with open(os.path.join(w_dir, 'global_metrics', 'skew_test.pkl'), 'rb') as f:
    skew = pickle.load(f)
with open(os.path.join(w_dir, 'global_metrics', 'EXPV_test.pkl'), 'rb') as f:
    EXPV = pickle.load(f)

#choose a metric
data=kurtosis 
dataname = 'kurtosis'
bins = np.linspace(-1, 5,100)  # Adjust bin range and number as needed
counts, edges = np.histogram(data, bins=bins)

x = []
y = []
for i, count in enumerate(counts):
    x.extend([0.5 * (edges[i] + edges[i+1])] * count)
    y.extend(range(count))

# plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='orange', s=10) #orange, blue, green
plt.xlabel(dataname)
plt.ylabel("Count")
plt.gca().set_facecolor("white") 
plt.grid(True, color="lightgray", linestyle="-", linewidth=0.7)
plt.show()


#%%% Optional -  set up data preparation for centile plots of specific voxels harmonized on batch effects 
#To plot this we need a fitted normative model object and the corresponding (test) normdata object
#so first we do as we did in the running script (ABCD stratify etc.) for normdata

# deal with the clinical subjects forced into test set
g_age=pd.read_csv(os.path.join(demographics_dir, 'ABCD/phenotypes/ABCDstudyNDA_AnnualRelease4.0/Package_1199073/abcd_devhxss01.txt'), sep = '\t')
g_age = g_age.iloc[1:]
g_age.rename(columns={'subjectkey': 'sub_id','devhx_ss_12_p':'gestational_age'}, inplace=True)
g_age = g_age[['sub_id','gestational_age']]
g_age = g_age.dropna()
g_age['sub_id'] = g_age['sub_id'].str.replace('_', '', regex=False) #correspondence to sub foldernames
g_age['sub_id'] = 'sub-'+ g_age['sub_id']
g_age['gestational_age'] = 40-g_age['gestational_age'].astype(int) # convert to actual GA
g_age = g_age[g_age.gestational_age > 0] # drop the 'uncertain' 999 values
preterm = g_age[g_age['gestational_age']<=37]
preterm.reset_index(drop=True, inplace = True)
clin_subs = preterm.iloc[:,0].values
ses_suffixes = ['_BL', '_2Y', '_4Y'] #add all possible observations for these subjects because longitudinal
clin_subs =np.char.add(np.repeat(clin_subs, len(ses_suffixes)), np.tile(ses_suffixes, len(clin_subs)))


#extract ABCD sites
normdata_path = os.path.join(w_dir, 'batch_0', 'norm_data.pkl')

with open(normdata_path, 'rb') as f:
    norm_data = pickle.load(f)
abcd_sites = {np.str_('site'):[v for values in norm_data.unique_batch_effects.values() for v in values if "abcd" in str(v)]}


def subject_level_split(data, split, random_state: int = 42,):
    """
    Perform subject-level stratified train/test split on a NormData object.
    ----------
    data : NormData
        The input NormData object.
    split : float
        Proportion of data (subjects) assigned to train.
    
    Returns
    -------
    (NormData, NormData)
        Train and test NormData objects.
    """

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



#%% Now the actual data extraction and plotting

if not os.path.exists(os.path.join(root_dir,'for_figures')):
    os.makedirs(os.path.join(root_dir,'for_figures'))
save_dir = (os.path.join(root_dir,'for_figures'))
           
#load the voxel indices nifti          
idxvol = ptkload(os.path.join(w_dir,'visuals','idx.nii.gz'), mask=mask_nii, vol=True)

#specify the voxel coordinate (x,y,z,0) in fsleyes (voxel location in template, not anatomical MNI coordinates)
vox_coord = (55,42,23,0) #cerebellar white matter
# vox_coord = (36,41,27,0)#cerebellar grey matter
# vox_coord = (35,57,32,0) #R hippocampus
# vox_coord = (49,68,46,0) #4th ventricle
# vox_coord = (37,64,31,0) #amygdala
# vox_coord = (59,57,48,0) # left internal capsule
vox_coord = (39,75,58,0) #just random test voxel


# find voxel index and batch index
vox_id = int(idxvol[vox_coord])
batch_num, mod_num = divmod(vox_id, vox_batch_size)
print (f'batch_{batch_num}_voxel_{vox_id}')

# load the corresponding normdata
with open(os.path.join(w_dir, f'batch_{batch_num}', 'norm_data.pkl'), 'rb') as f:
    norm_data= pickle.load(f)

#now, proceed with identical stratification steps used for the estimation to get the final test set

#separate the clinical subjects from norm_data
clin_subs_existing = [s for s in clin_subs if s in norm_data.subject_ids.values]
clin_subset = norm_data.where(norm_data.subject_ids.isin(clin_subs_existing),drop = True)
remaining = norm_data.where(~norm_data.subject_ids.isin(clin_subs_existing), drop=True)

#re-make subsets into proper norm_data objects to get subclass methods
clin_subset = NormData(data_vars=clin_subset.data_vars,coords=clin_subset.coords,
    attrs=clin_subset.attrs,name=clin_subset.name)

remaining = NormData(data_vars=remaining.data_vars,coords=remaining.coords,
    attrs=remaining.attrs,name = remaining.name)

#take out abcd (non-preterm) to split it independently
abcd_subset = remaining.select_batch_effects(name = 'abcd_subset', batch_effects = abcd_sites)
abcd_subs = abcd_subset.subject_ids.values
abcd_subset_train, abcd_subset_test= subject_level_split(abcd_subset, split = 0.6)

remaining = norm_data.where(~remaining.subject_ids.isin(abcd_subs), drop=True)
remaining = NormData(data_vars=remaining.data_vars,coords=remaining.coords,
    attrs=remaining.attrs,name = remaining.name)
remaining.register_batch_effects()

#split and put subjects back into the test set
train, test = remaining.train_test_split(splits = [0.5,0.5])
train = train.merge(abcd_subset_train)
test = test.merge(abcd_subset_test)
test = test.merge(clin_subset) 
test.load_zscores(save_dir = os.path.join(w_dir, f'batch_{batch_num}', 'results')) #used for qqplots

#now that we have our proper test normadata, we load the model
model= NormativeModel.load(os.path.join(w_dir, f'batch_{batch_num}'))
response_var = f'voxel_{vox_id}'

#make single-voxel test normata and normative model to avoid plotting the whole batch of voxels
def select_for_response_var(model: NormativeModel, data: NormData, response_var: str):
    if response_var not in model.regression_models:
        raise ValueError(f"Response variable '{response_var}' not in model.")
    new_model = copy.deepcopy(model)
    new_model.response_vars = [response_var]
    new_model.regression_models = {response_var: new_model.regression_models[response_var]}
    new_model.outscalers = {response_var: new_model.outscalers[response_var]} if new_model.outscalers else {}
    
    new_data = data.sel({"response_vars": response_var}, drop=False)
    new_data["Y"] = new_data.Y.expand_dims("response_vars", axis=-1) #preserves 1d of norm_data.Y because single voxel, otherwise plot will error
    new_data["Z"] = new_data.Z.expand_dims("response_vars", axis=-1) #not used for centile plots but required for qq plots
    new_data = new_data.assign_coords(response_vars=[response_var])
    
    return new_model, new_data

#these are the final ingredients needed for the plot
modelvox, testvox = select_for_response_var(model, test, response_var)

plot_qq(data = testvox)

plot_centiles( 
    modelvox,
    centiles=[0.01,0.05, 0.5, 0.95,0.99],  
    scatter_data= testvox, 
    scatter_kwargs = {"alpha" : 0.3, "s":10},
    save_dir= save_dir #leave it to save figure, comment out to show plot in IDE instead
)


# #optional syntax to select for specific batch_effects
# mask = (
#     (testvox.batch_effects.sel(batch_effect_dims="sex") == "F") &
#     (testvox.batch_effects.sel(batch_effect_dims="site").str.contains("ukb"))
# )
# testvox = testvox.where(mask, drop = True)
# testvox = NormData(data_vars=testvox.data_vars,coords=testvox.coords,
#     attrs=testvox.attrs,name=testvox.name)


#optional - alternatively, use the more advanced plotting function of the toolkit (color by batch effect etc.)
plot_centiles_advanced( 
    modelvox,
    centiles=[0.01,0.05, 0.5, 0.95,0.99],  
    scatter_data= testvox,
    batch_effects = "all",# Put None if we want to single out the base harmonization subjects
    show_other_data=False, 
    markers_data = None,
    hue_data='site',
    covariate_range = (2,90), #reasonable age coverage
    harmonize_data=True, #harmonizes on the first unique combination of batch effects
    show_legend = False,
    show_centile_labels = True,
    # save_dir= save_dir #can comment out to show plot in IDE
)


#%% Optional: make model ready to export for potential transfer/extend - copy only the models and tar them
import glob
import shutil
import joblib
from joblib import Parallel, delayed
import tarfile

dest_dir = os.path.join(root_dir, 'models', prefix + modality + suffix + '_models_only', 'models')
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

#grab models for all batches
model_paths = glob.glob(os.path.join(w_dir,'*','model'))

def grab_models(model_path, dest_dir):
        batch_name = os.path.basename(os.path.dirname(model_path))  # take only the folder’s name
        dest_path = os.path.join(dest_dir, batch_name, 'model')
    
        if os.path.exists(dest_path):
            print(f"Skipping {batch_name}, already exists at {dest_path}")
        else:
            shutil.copytree(model_path, dest_path)
            print(f"Copied {batch_name} model")
            
Parallel(n_jobs=-1,verbose =5)(delayed(grab_models)(model_path, dest_dir) for model_path in model_paths[:])

#quick check if models are missing from a batch (estimation failed etc.)
#if all is well, only the last batch should come up (as it has less voxels than the rest)
for b in range(n_batches):
    batch = f'batch_{b}'
    if len(os.listdir(os.path.join(dest_dir,batch,'model')))==0:
        print(f'model from {batch} is missing !')
    if len(os.listdir(os.path.join(dest_dir,batch,'model')))!=vox_batch_size+1 :
        print(f'less than {vox_batch_size } models in {batch}!')
        
#zip the folder
source_dir = os.path.join(root_dir, 'models', prefix + modality + suffix + '_models_only')
output_tar = source_dir+'.tar'

with tarfile.open(output_tar, "w") as tar:
    tar.add(source_dir, arcname=os.path.basename(source_dir))

print(f"Created archive: {output_tar}")

