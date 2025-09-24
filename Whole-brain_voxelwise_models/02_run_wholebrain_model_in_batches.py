#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:00:33 2025

@author: alicha

N.B This script used a cluster to fit the models, base on the runner tool of the pcntoolkit. 
Please check the 04_transfer_extend_whole_model_in_batches jupyter noteebook to get some code for the estimation of models on a local machine instead.
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pcntoolkit import (
    BLR,
    BsplineBasisFunction,
    LinearBasisFunction,
    NormativeModel,
    NormData,
    Runner,
    plot_centiles,
    plot_qq,
    dataio,
)
import pcntoolkit.util.output
import seaborn as sns
import logging
import os
import dill 
import sys
import numpy as np

# Suppress some annoying warnings and logs
pymc_logger = logging.getLogger("pymc")
pymc_logger.setLevel(logging.WARNING)
pymc_logger.propagate = False

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pcntoolkit.util.output.Output.set_show_messages(False)

#%% Global settings
modality = 'log_jacs' #'log_jacs', 'mod_gmv', 'mod_wmv'
extension = '_wholebrain_2mm_09'#naming purposes
batch_size = 150
split = [0.5,0.5] #train-test s-plit for model estimation

venv_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable))) #path to toolkit
root_dir = '/path/to/root/dir/'
base_dir =os.path.join(root_dir,'models/')
proc_dir = os.path.join(base_dir,'BLR_aff_nonlin_' + modality + extension +'/')
w_dir = os.path.join(proc_dir,'models/')

# mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_T1w_BrainExtractionBrain.nii.gz')
if modality == 'log_jacs':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
if modality == 'mod_gmv':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg.nii.gz')
if modality == 'mod_wmv':
    mask_nii = os.path.join(root_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg.nii.gz')


# %% prepare to get the clinical subjects out of the training set (e.g. preterm subjects)
patients=pd.read_csv(os.path.join(root_dir, 'subjects_id_clinical.txt'), header = None)
clin_subs = patients.iloc[:,0].values

#%% configure model template

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

#%%load data and run for each batch

mask = dataio.fileio.load(mask_nii, mask=mask_nii, vol=False).T #load all nonzero voxels of the mask
n_voxels = len(mask)
n_batches = n_voxels // batch_size + int(n_voxels % batch_size != 0)

for b in range(n_batches)[:]: #slice this based on job submission limits
    batch = f'batch_{b}'
    print(batch)
    
    model = NormativeModel(
        template_regression_model=template_blr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=True,
        save_dir= w_dir+batch+'/',
        inscaler="standardize",
        outscaler="standardize",
    )
    
    runner = Runner(
        cross_validate=False,
        parallelize=True, 
        n_batches = 2, # can't just run one per batch otherwise throws an error
        environment=venv_path,
        job_type="slurm",  # or "torque" if you are on a torque cluster
        time_limit="12:00:00",
        memory = "2GB",
        n_cores=1,
        log_dir=os.path.join(proc_dir,'logs/'),
        temp_dir=os.path.join(proc_dir,'tmp/'),
        preamble = "module load gcc/13.3.0; module load anaconda3" #have to add the gcc versionto avoid  an error between the toolkit requirements and default cluster gcc version
    )#if the correct gcc version is installed in conda, can probably omit the explicit gcc load

    normdata_path = os.path.join(w_dir, batch, 'norm_data.pkl')
    
    with open(normdata_path, 'rb') as f:
        norm_data = dill.load(f)
    
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
    
    runner.fit_predict(model, train, test,observe=False)
    

#%%Optional debugging /testing

##try one batch if it fails in the runner 
b = 0
batch = f'batch_{b}'

model = NormativeModel(
    template_regression_model=template_blr,
    savemodel=True,
    evaluate_model=True,
    saveresults=True,
    saveplots=True,
    save_dir= w_dir+batch+'/',
    inscaler="standardize",
    outscaler="standardize",
)

print(batch)
normdata_path = os.path.join(w_dir, batch, 'norm_data.pkl')

with open(normdata_path, 'rb') as f:
    norm_data = dill.load(f) 
    
train, test = norm_data.train_test_split(splits = split)
model.fit_predict(train, test)

##can even try just two voxels for quick test
with open(normdata_path, 'rb') as f:
    norm_data = dill.load(f) 
features_to_model = ["voxel_1","voxel_2"] #change the names based on the actual voxels of the batch though
norm_data = norm_data.sel({"response_vars":features_to_model}) #gives an error if just one response var hence two voxels
train, test = norm_data.train_test_split(splits = split)
model.fit(train)

# #visualize data for just one voxel for sanity check
feature_to_plot = features_to_model[1]
df = test.to_dataframe()
fig, ax = plt.subplots(1, 2, figsize=(15, 15))

sns.countplot(data=df, y=("batch_effects", "site"), hue=("batch_effects", "sex"), ax=ax[0], orient="h")
ax[0].legend(title="Sex")
ax[0].set_title("Count of sites")
ax[0].set_xlabel("Site")
ax[0].set_ylabel("Count")

sns.scatterplot(
    data=df,
    x=("X", "age"),
    y=("Y", feature_to_plot),
    hue=("batch_effects", "site"),
    style=("batch_effects", "sex"),
    ax=ax[1],
)
ax[1].legend([], [])
ax[1].set_title(f"Scatter plot of age vs {feature_to_plot}")
ax[1].set_xlabel("Age")
ax[1].set_ylabel(feature_to_plot)

plt.show()

