#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:40:24 2023 edited 2025

@author alicha
"""

import os
import numpy as np
import pandas as pd
import pcntoolkit.dataio.fileio as fileio
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from pcntoolkit import NormData, plot_centiles,  NormativeModel
import pickle
import dill
import matplotlib.pyplot as plt
import itertools
import re
import copy

#parameters of the estimated model
modality = 'log_jacs' #e.g. 'log_jacs', 'mod_gmv', 'mod_wmv'
extension = '_wholebrain_2mm_01'#example name
name = '58597_subjects'# this is the name initially given to the norm_data in 01_prepare_models - it is propagated to the results .csv files
batch_size = 150
split = [0.5, 0.5]

root_dir = '/path/to/root/dir/'
base_dir =os.path.join(root_dir,'models/')
proc_dir = os.path.join(base_dir,'BLR_aff_nonlin_' + modality + extension +'/')
w_dir = os.path.join(proc_dir,'models/')
if modality == 'log_jacs':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
if modality == 'mod_gmv':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg.nii.gz')
if modality == 'mod_wmv':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg.nii.gz')
    
ex_nii = mask_nii

if not os.path.exists(w_dir + '/visuals/'):
    os.makedirs(w_dir + '/visuals/')
if not os.path.exists(w_dir + '/global_metrics/'):
    os.makedirs(w_dir + '/global_metrics/')

#%% collect stats from batches into global metrics and niftis

mask = ptkload(mask_nii, mask=mask_nii, vol=False).T #load all nonzero voxels of the mask
n_voxels = len(mask)
n_batches = n_voxels // batch_size + int(n_voxels % batch_size != 0)
  
def extract_num(col_name): #handling the ordering of column names between batches (the runner does not always put them back in order)
    match = re.search(r'voxel_(\d+)', col_name)
    return int(match.group(1)) if match else float('inf')  # handle non-matching names safely

collect_metrics = []
batch_metrics = []

for b in range(n_batches):
    batch = f'batch_{b}'
    print(batch)
    
    datapath = os.path.join(w_dir, batch, 'results', 'statistics_'+ name +'_test.csv')
    data = pd.read_csv(datapath)
    data.set_index("statistic", inplace = True)
  
    sorted_cols = sorted(data.columns, key=extract_num) #re-order response vars
    data = data[sorted_cols]
    
    data.columns = [f'{batch}_voxel_{i}' for i in range(len(data.columns))]
    batch_metrics.append(data)
    
collect_metrics = pd.concat(batch_metrics, axis = 1)
    
for metric in collect_metrics.index:
    stat_data= np.array(collect_metrics.loc[metric])
    
    ###optional for early testing: dummy padding to have the right amount of voxels in case some batches were not run - turn off when everything was properly run
    # pad_width = n_voxels - len(stat_data)
    # stat_data = np.pad(stat_data, (0, pad_width), mode='constant')
    
    with open(os.path.join(w_dir, 'global_metrics', metric+'_test.pkl'), 'wb') as f:
        pickle.dump(stat_data, f)
    
    ptksave(stat_data, os.path.join(w_dir, 'visuals', metric+'_test.nii.gz'), example=ex_nii, mask=mask_nii)
    
ptksave(np.arange(len(stat_data)), os.path.join(w_dir,'visuals','idx.nii.gz'), example=ex_nii, mask=mask_nii, dtype='uint32')#just get total number of voxels indexed

#%% compute kurtosis and skew and also put them into niftis 

mask = ptkload(mask_nii, mask=mask_nii, vol=False).T
n_voxels = len(mask)
n_batches = n_voxels // batch_size + int(n_voxels % batch_size != 0)


def extract_num(col_name): #handling the oredering of column names between batches
    match = re.search(r'voxel_(\d+)', col_name)
    return int(match.group(1)) if match else float('inf') 

skew_all = []
kurtosis_all = []

for b in range(n_batches):
# for b in range(5):
    batch = f'batch_{b}'
    print(batch)
    
    Z_path = os.path.join(w_dir, batch, 'results', 'Z_'+ name +'_test.csv') #you can also get those from the norm_data object but it is slower
    Z = pd.read_csv(Z_path)
    Z = Z.drop(columns='observations')
    sorted_cols = sorted(Z.columns, key=extract_num)
    Z = Z[sorted_cols]
    
    #print(len(Z.columns))
    Z[np.isnan(Z)] = 0
    Z[np.isinf(Z)] = 0
    
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
    
    
# ##optional early tesing : dummy padding for a model that has not finished estimating 
# pad_width = n_voxels - len(skew_all)
# skew_all = np.pad(skew_all, (0, pad_width), mode='constant')
# kurtosis_all = np.pad(kurtosis_all, (0, pad_width), mode='constant')
   
with open(os.path.join(w_dir, 'global_metrics', 'skew_test.pkl'), 'wb') as f:
    pickle.dump(np.array(skew_all), f)
with open(os.path.join(w_dir, 'global_metrics', 'kurtosis_test.pkl'), 'wb') as f:
    pickle.dump(np.array(kurtosis_all), f)

ptksave(np.array(skew_all), os.path.join(w_dir,'visuals','skew_test.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(np.array(kurtosis_all), os.path.join(w_dir,'visuals','kurtosis_test.nii.gz'), example=ex_nii, mask=mask_nii)


#%%QC - check voxels with very high kurtosis 
idxvol = ptkload(os.path.join(w_dir,'visuals','idx.nii.gz'), mask=mask_nii, vol=True)
vox_coord = (37,67,45,0)#2mm voxel coords in fsleyes
vox_id = int(idxvol[vox_coord])

# find batch id
batch_num, mod_num = divmod(vox_id, batch_size)
batch_num = batch_num + 1 # batch indexing starts at 1
print (f'batch_{batch_num}_response_var_{mod_num}')

with open(os.path.join(w_dir, f'batch_{batch_num}', 'norm_data.pkl'), 'rb') as f:
    normdata= dill.load(f)
    
train, test = normdata.train_test_split(splits = split) #this gets us the proper name etc. for the object
testvox = test.sel({"response_vars":f"response_var_{mod_num}"})
sub = int(testvox.Y.argmin())
subid = testvox.subjects[sub]
print(f"outlier subject: {subid.values}")
np.sort(testvox.Y)[:10][::1] 

#how many outliers really look outlying
out_num = 1 #adjust based on how many you see with the output from above
out_indices = np.argpartition(testvox.Y.values, out_num)[:out_num]
out_subjects = testvox.subjects[out_indices]
out_values = testvox.Y[out_indices]
print(f"outlier subjects: {out_subjects.values} with values: {out_values.values}")



#%% optional extra QC : pull subjects that have a very large number of outlier voxels 
data_dir = '/path/to/data/dir'

mask = ptkload(mask_nii, mask=mask_nii, vol=False).T
n_voxels = len(mask)
n_batches = n_voxels // batch_size + int(n_voxels % batch_size != 0)

voxel_threshold = 0.05 #taking out subjects with more than 5% of total voxels being outliers - value can be changed
threshold = 7 #outlier being defined here as more than 7 std away from the mean

def extract_num(col_name): #handling the oredering of column names between batches
    match = re.search(r'voxel_(\d+)', col_name)
    return int(match.group(1)) if match else float('inf')  # handle non-matching names safely

outliers_per_batch = []
outliers_train_per_batch = []
outliers_test_per_batch= []

for b in range(n_batches):
    batch = f'batch_{b}'
    print(batch)
    
    with open(os.path.join(w_dir, batch, 'norm_data.pkl'), 'rb') as f:
        norm_data= pickle.load(f)
        
    ## Look at the Z
    train, test = norm_data.train_test_split(splits = split) #this gets us the proper name etc. for the object
    train.load_zscores(save_dir = os.path.join(w_dir, batch, 'results'))
    test.load_zscores(save_dir = os.path.join(w_dir, batch, 'results'))
    
    outliers_train = np.abs(train.Z.values) > threshold
    outliers_train = outliers_train.sum(axis=1)
    outliers_test = np.abs(test.Z.values) > threshold
    outliers_test = outliers_test.sum(axis=1)
    outliers_train_per_batch.append(outliers_train[:,np.newaxis])
    outliers_test_per_batch.append(outliers_test[:,np.newaxis])
 
outliers_train_tot = np.concatenate(outliers_train_per_batch, axis = 1)
outliers_train_per_sub = outliers_train_tot.sum(axis=1)
outlier_train_summary = np.vstack((train.subjects.values, outliers_train_per_sub)).T #how many extreme voxels each subject has
outliers_test_tot = np.concatenate(outliers_test_per_batch, axis = 1)
outliers_test_per_sub = outliers_test_tot.sum(axis=1)
outlier_test_summary = np.vstack((test.subjects.values, outliers_test_per_sub)).T #how many extreme voxels each subject has
outlier_summary = np.vstack([outlier_train_summary, outlier_test_summary])

outlier_summary_sorted = outlier_summary[outlier_summary[:,1].argsort()]
outliers_to_exclude = outlier_summary[outlier_summary[:,1]>n_voxels*voxel_threshold, 0] 

outliers_to_exclude = pd.DataFrame(outliers_to_exclude)
cohort = 'UKB' #in our case, all outliers were from UKB
outliers_to_exclude.to_csv(os.path.join(data_dir,cohort,'qc','T1_ants',f'sub_exclude_outliers_{int(voxel_threshold*100)}PercentofVoxels_{modality}.txt'), index = None, header = None)

#%% some histograms 

with open(os.path.join(w_dir, 'global_metrics', 'kurtosis_test.pkl'), 'rb') as f:
    kurtosis = pickle.load(f)
with open(os.path.join(w_dir, 'global_metrics', 'skew_test.pkl'), 'rb') as f:
    skew = pickle.load(f)
with open(os.path.join(w_dir, 'global_metrics', 'EXPV_test.pkl'), 'rb') as f:
    EXPV = pickle.load(f)

data=EXPV#choose one of the above
bins = np.linspace(0, 1,100)  # Adjust bin range and number as needed
counts, edges = np.histogram(data, bins=bins)

x = []
y = []
for i, count in enumerate(counts):
    x.extend([0.5 * (edges[i] + edges[i+1])] * count)
    y.extend(range(count))

# Plot the dots
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='green', s=10) #e.g. orange, blue, green
plt.xlabel("EXPV")
plt.ylabel("Count")
plt.gca().set_facecolor("white") 
plt.grid(True, color="lightgray", linestyle="-", linewidth=0.7)
plt.show()

#%%% for figure : scatterplots of specific voxels sorting out the harmonization on batch 

if not os.path.exists(os.path.join(w_dir,'visuals','for_figures')):
    os.makedirs(os.path.join(w_dir,'visuals','for_figures'))
save_dir = (os.path.join(w_dir,'visuals','for_figures'))
           
idxvol = ptkload(os.path.join(w_dir,'visuals','idx.nii.gz'), mask=mask_nii, vol=True)
vox_coord = (46,91,30,0)#2mm coords in fsleyes
vox_id = int(idxvol[vox_coord])

# find batch id
batch_num, mod_num = divmod(vox_id, batch_size)
print (f'batch_{batch_num}_voxel_{mod_num}')


### get the real test set for the plot
# deal with the clinical subjects forced into test set
preterm=pd.read_csv(os.path.join(root_dir, 'subjects_ids_clinical.txt'), header = None)
clin_subs = preterm.iloc[:,0].values

with open(os.path.join(w_dir, f'batch_{batch_num}', 'norm_data.pkl'), 'rb') as f:
    norm_data= dill.load(f)
    
#separate the clinical subjects from norm_data 
clin_subs_existing = [s for s in clin_subs if s in norm_data.subjects.values]
clin_subset = norm_data.where(norm_data.subjects.isin(clin_subs_existing),drop = True)
remaining = norm_data.where(~norm_data.subjects.isin(clin_subs_existing), drop=True)

#re-make subsets into proper norm_data objects to get subclass methods
clin_subset = NormData(data_vars=clin_subset.data_vars,coords=clin_subset.coords,
    attrs=clin_subset.attrs,name=clin_subset.name)

remaining = NormData(data_vars=remaining.data_vars,coords=remaining.coords,
    attrs=remaining.attrs,name = remaining.name)

#split and put subjects back into the test set
train, test = remaining.train_test_split(splits = split)
test = test.merge(clin_subset) 

model= NormativeModel.load(os.path.join(w_dir, f'batch_{batch_num}'))
response_var = f'voxel_{mod_num}'

#select a single voxel to avoid plotting the whole batch
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


modelvox, testvox = select_for_response_var(model, test, response_var)

#general plot
plot_centiles( 
    modelvox,
    centiles=[0.01,0.05, 0.5, 0.95,0.99],  
    scatter_data=testvox, 
    batch_effects = "all",# Put None if we want to single out the base harmonization subjects
    show_other_data=True, 
    hue_data = None,
    harmonize_data=True, #harmonizes on the first elvel
    show_legend = False,
    show_centile_labels = True,
    save_dir= save_dir
)


###optional batch_effect-specific graphs : filtering for one batch effect (sex) while keeping all of the other effects (site)
# cond = testvox.batch_effects.sel(batch_effect_dims="sex")== "0"
# cond = cond.reset_coords(drop=True)
# subset = testvox.where(cond,drop = True)

# subset = NormData(data_vars=subset.data_vars,coords=subset.coords,
#     attrs=subset.attrs,name = subset.name)

# plot_centiles( 
#     modelvox,
#     centiles=[0.05, 0.25, 0.5, 0.75, 0.95],  
#     scatter_data=subset, 
#     #batch_effects={"sex": ["0"]} "site": ["site_name"]},  # Highlight women, choose one site
#     show_other_data=True,  
#     harmonize=True,  
#     show_yhat=False,
#     show_legend = False,
# )


##optional syntax if we want to select plotting on a batch effect while keeping all values of the other : 
# batch_effects = {
# "sex": list(testvox.batch_effects.sel(batch_effect_dims="sex").values),
# "site": list(testvox.batch_effects.sel(batch_effect_dims="site").values),
# },

#%% optional - make model ready to export for potential transfer/extend, i.e. make a new folder containing only the estiated models 
# this is slow, so parallelization helps

import glob
import shutil
from joblib import Parallel, delayed

dest_dir = os.path.join(base_dir,'BLR_aff_nonlin_' + modality + extension + '_models_only', 'models')
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    
model_paths = glob.glob(os.path.join(w_dir,'*','model'))
def grab_models(model_path):
        batch_name = os.path.basename(os.path.dirname(model_path))  # take only the folder’s name
        dest_path = os.path.join(dest_dir, batch_name, 'model')
    
        if os.path.exists(dest_path):
            print(f"Skipping {batch_name}, already exists at {dest_path}")
        else:
            shutil.copytree(model_path, dest_path)
            print(f"Copied {batch_name} model")
            
Parallel(n_jobs=-1,verbose =5)(delayed(grab_models)(model_path)for model_path in model_paths[:])

