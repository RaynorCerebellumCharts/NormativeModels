#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:00:33 2025

@author: Alice Chavanne
This scripts loads the norm_data objects and runs the normative model estimation for each batch of response variables (voxel values). 
This can be done on a computing cluster like slur or torque, but can also be run with local parallelization. 
One normative model is estimated per voxel.
"""

import warnings
import pandas as pd
from pcntoolkit import (
    BLR,
    BsplineBasisFunction,
    NormativeModel,
    NormData,
    Runner,
    dataio,
)
import pcntoolkit.util.output
import logging
import os
import dill as dill
import sys
import numpy as np
import copy
from sklearn.model_selection import train_test_split as skl_split
import re    
import collections

# Suppress some warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(False)


#%% Global settings
modality = 'log_jacs'#choose brain measure ('log_jacs', 'mod_gmv', 'mod_wmv')
prefix = 'BLR_aff_nonlin_'#prefix for model name/directory
suffix = '_wholebrain_2mm_14' #suffix same 

venv_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable))) #path to toolkit
root_dir = '/path/to/root_dir/'
tpl_dir = os.path.join(root_dir,'tpl-MNI152NLin2009cSym')
proc_dir = os.path.join(root_dir,'models', prefix + modality + suffix )
w_dir = os.path.join(proc_dir,'models')

if modality == 'log_jacs':
    mask_nii = os.path.join(tpl_dir, 'tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
if modality == 'mod_gmv':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg.nii.gz')
if modality == 'mod_wmv':
    mask_nii = os.path.join(tpl_dir,'tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg.nii.gz')

vox_batch_size = 150 #must be the same value as the script to prepare normdata 


# %% prepare to get the clinical subjects out of the training set (e.g. ABCD preterm subjects)
dem_dir = '/path/to/demographics_dir/'
g_age=pd.read_csv(os.path.join(dem_dir,'ABCD/phenotypes/ABCDstudyNDA_AnnualRelease4.0/Package_1199073/abcd_devhxss01.txt'), sep = '\t')
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


#%% prepare to stratify ABCD so that the same subject doesn't end up in both train and test

#First, extract all ABCD site names from any normdata object
# we need them to set aside ABCD subjects later
normdata_path = os.path.join(w_dir, 'batch_0', 'norm_data.pkl')

with open(normdata_path, 'rb') as f:
    norm_data = dill.load(f)
abcd_sites = {np.str_('site'):[v for values in norm_data.unique_batch_effects.values() for v in values if "abcd" in str(v)]}


def subject_level_split(data, split, random_state: int = 42,):

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

    #one label per unique subject - take last one if several (this also covers baseline scanners that were not used in follow-ups)
    #this makes an imperfect stratification for subjects that changed scanners, but ensures that same-subject scans end up in the same set after splitting
    #true site labels are not overwritten
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


#%%Load data, configure and run model estimation for each voxel batch
# N.B. The runner is usually convenient to handle whole datasets and slice it up as parallel jobs, 
# but voxelwise data is too big for that, which is why we have sliced the data and saved the normdata as batches of voxels beforehand.
# Here, the runner will only split the estimation of each voxel batch into two parallel jobs.

mask = dataio.fileio.load(mask_nii, mask=mask_nii, vol=False).T #load all nonzero voxels of the mask
n_voxels = len(mask)
n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)

#configure model template
template_blr = BLR(
    name="template",
    basis_function_mean = BsplineBasisFunction(),
    fixed_effect = True,
    fixed_effect_var = False,
    warp_name="WarpSinhArcsinh",
    warp_reparam = True, 
    heteroskedastic= False,  
    optimizer="powell",
    ard=False,
)


for b in range(n_batches)[:]:
    batch = f'batch_{b}'
    print(batch)
    
    #configure normative model object for this batch
    model = NormativeModel(
        template_regression_model=template_blr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=False,
        save_dir= os.path.join(w_dir,batch),
        inscaler="standardize",
        outscaler="standardize",
    )
    
    #initialize runner for this batch
    runner = Runner(
        cross_validate=False,
        parallelize=True, 
        n_batches = 2, # the runner needs at least 2 jobs
        environment=venv_path,
        job_type="slurm",
        time_limit="12:00:00",
        memory = "2GB",
        n_cores=1,
        log_dir=os.path.join(proc_dir,'logs'),
        temp_dir=os.path.join(proc_dir,'tmp'),
        preamble = "module load gcc/13.3.0; module load anaconda3" 
    )
    
    #grab the data of the batch
    normdata_path = os.path.join(w_dir, batch, 'norm_data.pkl')
    
    with open(normdata_path, 'rb') as f:
        norm_data = dill.load(f)
    
    #separate the clinical subjects from the rest 
    clin_subs_existing = [s for s in clin_subs if s in norm_data.subject_ids.values]
    clin_subset = norm_data.where(norm_data.subject_ids.isin(clin_subs_existing),drop = True)
    remaining = norm_data.where(~norm_data.subject_ids.isin(clin_subs_existing), drop=True)
    
    #make subsets into proper norm_data objects again to get subclass methods
    clin_subset = NormData(data_vars=clin_subset.data_vars,coords=clin_subset.coords,
        attrs=clin_subset.attrs,name=clin_subset.name)

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
    
    #split the rest, put non-preterm abcd back in both train and test, and push clinical subjects into the test set
    train, test = remaining.train_test_split(splits = [0.5,0.5])
    train = train.merge(abcd_subset_train)
    test = test.merge(abcd_subset_test)
    test = test.merge(clin_subset)
    
    runner.fit_predict(model, train, test, observe=False)
    
#%% Optional : visualize demographics of aggregated sample, just keep any train and test norm_data from above

print ("train set")
print (f"subjects total {len(train.subject_ids.values)}")
print(f"unique sites {len(list(train.unique_batch_effects.values())[1])}")
print(f"female {100*(train.batch_effects.isel(batch_effect_dims=0) == '0').sum().item()/len(train.subject_ids.values):.1f} percent")
print(f"male {100*(train.batch_effects.isel(batch_effect_dims=0) == '1').sum().item()/len(train.subject_ids.values):.1f} percent")
print(f"mean age train {np.mean(train.X.values):.1f} sd {np.std(train.X.values):.1f}")
print ("test set")
print (f"subjects total {len(test.subject_ids.values)}")
print(f"unique sites {len(list(test.unique_batch_effects.values())[1])}")
print(f"female {100*(test.batch_effects.isel(batch_effect_dims=0) == '0').sum().item()/len(test.subject_ids.values):.1f} percent")
print(f"male {100*(test.batch_effects.isel(batch_effect_dims=0) == '1').sum().item()/len(test.subject_ids.values):.1f} percent")
print(f"mean age test {np.mean(test.X.values):.1f} sd {np.std(test.X.values):.1f}")


#%%Optional for prototyping : try to fit just the first two voxels locally
import matplotlib.pyplot as plt
import seaborn as sns

b = 0
batch = f'batch_{b}'
print(batch)
normdata_path = os.path.join(w_dir, batch, 'norm_data.pkl')

with open(normdata_path, 'rb') as f:
    norm_data = dill.load(f) 
    
features_to_model = ["voxel_1","voxel_2"]
norm_data = norm_data.sel({"response_vars":features_to_model}) #gives an error if just one respvar
train, test = norm_data.train_test_split(splits = [0.5,0.5]) #just simplest split to test


#visualize the site distribution and raw voxel values for one voxel
feature_to_plot = features_to_model[1] #voxel 2
df = test.to_dataframe()
fig, ax = plt.subplots(1, 2, figsize=(15, 15))

sns.countplot(data=df, y=("batch_effects", "site"), hue=("batch_effects", "sex"), ax=ax[0], orient="h")
ax[0].legend(title="sex")
ax[0].set_title("count of sites")
ax[0].set_xlabel("site")
ax[0].set_ylabel("count")

sns.scatterplot(
    data=df,
    x=("X", "age"),
    y=("Y", feature_to_plot),
    hue=("batch_effects", "site"),
    ax=ax[1],
)
ax[1].legend([], [])
ax[1].set_title(f"Scatter plot of age vs {feature_to_plot}")
ax[1].set_xlabel("Age")
ax[1].set_ylabel(feature_to_plot)

plt.show()


#still need to configure model
template_blr = BLR(
    name="template",
    basis_function_mean = BsplineBasisFunction(),
    fixed_effect = True,
    fixed_effect_var = False,
    warp_name="WarpSinhArcsinh",
    warp_reparam = True, 
    heteroskedastic= False,  
    optimizer="powell",
    ard=False,
)

model = NormativeModel(
    template_regression_model=template_blr,
    savemodel=True,
    evaluate_model=True,
    saveresults=True,
    saveplots=False,
    save_dir= os.path.join(w_dir,batch),
    inscaler="standardize",
    outscaler="standardize",
)

#run
model.fit_predict(train, test)

