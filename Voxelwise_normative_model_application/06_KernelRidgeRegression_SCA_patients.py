#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:35:32 2024

@author: Alice Chavanne

This script conducts a kernel ridge regression analysis evaluated out-of-sample in patients with spinocerebellar ataxia type 1 and 3,
from two independent datasets. The predictive performance of raw, whole-brain voxelwise measures (log jacobian, GMV & WMV combined ) and normative measures 
(Z-scores) are explored separately using the options set.

"""

import numpy as np
from pcntoolkit.dataio.fileio import save as ptksave
import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import  LeaveOneOut, GridSearchCV
from sklearn.metrics import  r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge 
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib
from nibabel import processing
import nilearn as nil
from nilearn import image
from pcntoolkit import dataio
import joblib
from joblib import Parallel, delayed


data_dir = '/path/to/data/dir/with/raw/ants/outputs/'
out_dir = os.path.join(data_dir,'clinical_predictions')
extension = ''#optional extra suffix added to output folder/filenames
model_dir = '/path/to/dir/where/normative/models/live'
datasets = ['SCA1_Nijmegen', 'SCA3_Mexico'] #some indices are hardcoded based on this order so have be careful


#%% Choose options for regression

target = {
    "BL_SARA" : 0,
    "Y1_SARA" : 0,
    "Slope_BL_Y1_SARA" : 1,
    }

train_and_test = {
    "test_Nijmegen_CV":1,
    "test_Nijmegen_and_Mexico_CV" : 0,    
    }

featureset = {
    "BL_wholebrain_log_jacs_Z" : 1,#can increase RAM if 1 here
    "BL_wholebrain_log_jacs_raw" : 0,
    "BL_wholebrain_mod_gmv_and_wmv_Z": 0,
    "BL_wholebrain_mod_gmv_and_wmv_raw": 0,
    }

options = {
    "gridsearch" : 0, 
}

#account for features with non-standardized values, those that need smoothing or resample
if featureset["BL_wholebrain_mod_gmv_and_wmv_raw"] :
    options["Xstandard"] = 1
    resample = True
    smooth = True
elif featureset["BL_wholebrain_log_jacs_raw"] :
    options["Xstandard"] = 1
    resample = True
    smooth = False
else : 
    options["Xstandard"] = 0
    resample = False
    smooth = False    
    
seed = 42       

filename = "_".join([key for key, value in target.items() if value] +
                     ["KRR_with"]+
                     [key for key, value in featureset.items() if value] +
                     [key for key, value in train_and_test.items() if value] +
                     [key for key, value in options.items() if value]+
                     [extension])

final_dir = os.path.join(out_dir,filename)
os.makedirs(final_dir, exist_ok=True)
os.chdir(final_dir)


#%% Nifti specifics

if (featureset["BL_wholebrain_log_jacs_raw"] or featureset["BL_wholebrain_log_jacs_Z"]):
    mask_nii = os.path.join(model_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_T1w_mask_BrainExtractionBrain.nii.gz')
    ex_nii = mask_nii
    proc_dir = os.path.join(model_dir,'models','BLR_aff_nonlin_log_jacs_wholebrain_2mm_14')
    
if (featureset["BL_wholebrain_mod_gmv_and_wmv_raw"] or featureset["BL_wholebrain_mod_gmv_and_wmv_Z"]): # combine the two masks
    mask_nii_gm = os.path.join(model_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-GM_mask_probseg.nii.gz')
    mask_nii_wm = os.path.join(model_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_label-WM_mask_probseg.nii.gz')
    mask_nii_combined = os.path.join(model_dir,'tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-2_combined_label-GM_WM_mask.nii.gz')
    proc_dir_gm= os.path.join(model_dir,'models','BLR_aff_nonlin_mod_gmv_labelmask_2mm_smoothed_fwhm6_07') 
    proc_dir_wm= os.path.join(model_dir,'models','BLR_aff_nonlin_mod_wmv_labelmask_2mm_smoothed_fwhm6_07') 
    proc_dir = proc_dir_gm #just to grab the demographics - gm and wm are identical on that front
    
    #just for plotting back onto the combined mask 
    ex_nii = mask_nii_combined
    mask_nii = mask_nii_combined
    
#details for brain data extraction from niftis
vox_size = [2, 2, 2] #2mm
fwhm = 6 #if smooth is true
vox_batch_size = 150

#%% Grab X and y data

#utils
def load_file(subpath, mask_nii, resample, smooth, vox_size):
    
    img = nib.load(subpath)
    if resample : 
        img = nib.processing.resample_to_output(img, vox_size)
    if smooth:
        img = nil.image.smooth_img(img,fwhm)
    data = img.get_fdata()
    data = dataio.fileio.vol2vec(data, nib.load(mask_nii).get_fdata())
    
    return data

# prepare for gmv and wmv, load GM and WM images, filter them by their respective masks, and combine them into a single vector consistent with the combined mask.
def load_files_and_combine(sub_path_gm, sub_path_wm, mask_nii_gm, mask_nii_wm, mask_nii_combined,resample, smooth, vox_size) :
                          
    gm_img = nib.load(sub_path_gm)
    wm_img = nib.load(sub_path_wm)

    if resample:
        gm_img = nib.processing.resample_to_output(gm_img, vox_size)
        wm_img = nib.processing.resample_to_output(wm_img, vox_size)
    if smooth:
        gm_img = image.smooth_img(gm_img, fwhm)
        wm_img = image.smooth_img(wm_img, fwhm)

    gm_data = gm_img.get_fdata()
    wm_data = wm_img.get_fdata()

    gm_mask = nib.load(mask_nii_gm).get_fdata().astype(bool)
    wm_mask = nib.load(mask_nii_wm).get_fdata().astype(bool)
    combined_mask = nib.load(mask_nii_combined).get_fdata().astype(bool)

    combined_data = np.zeros_like(gm_data).astype(float)  
    combined_data[gm_mask] = gm_data[gm_mask]
    combined_data[wm_mask] = wm_data[wm_mask]

    data = combined_data[combined_mask]
    return data
    

# now grab data from all datasets
df_sca = []
y_sca = []
X_sca = []

for dataset in datasets:
    #grab subjects and clinical scores
    df_clin = pd.read_csv(os.path.join(proc_dir, 'transfer_' + dataset, 'dem_patients.csv'))
    
    #grab y
    if target["BL_SARA"]:
        y_clin = df_clin['SARA_score']
    if target["Y1_SARA"]:
        if 'Y1_SARA' in df_clin.columns:
            df_clin.dropna(subset=['Y1_SARA'], inplace=True)
            df_clin= df_clin.reset_index()
            y_clin = df_clin.get('Y1_SARA', pd.Series([], name='Y1_SARA'))
        else :
            df_clin = pd.DataFrame()
    if target["Slope_BL_Y1_SARA"]:
        if 'Slope_BL_Y1_SARA' in df_clin.columns:
            df_clin.dropna(subset=['Slope_BL_Y1_SARA'], inplace=True)
            df_clin= df_clin.reset_index()
            y_clin = df_clin.get('Slope_BL_Y1_SARA', pd.Series([], name='Slope_BL_Y1_SARA'))
        else :
            df_clin = pd.DataFrame()
            
    sub_list = df_clin['sub_id'] if 'sub_id' in df_clin.columns else pd.Series([], dtype=str)

    # ensure this doesn't break even for datasets with missing timepoints
    if len(sub_list) == 0:
        df_sca.append(df_clin)
        y_sca.append(pd.Series([], name='y_empty'))
        X_sca.append(pd.DataFrame())  
        continue  
        
        
    # grab X
    if (featureset["BL_wholebrain_log_jacs_raw"]) :
        file_name = 'T1_BrainNorm_Jacobian.nii.gz'
        if dataset == 'SCA1_Nijmegen' : 
            f_pattern = os.path.join(data_dir, 'subjects_data','{}','ses-01','anat','T1_MNI152NLin2009cSym', file_name)
        if dataset == 'SCA3_Mexico' : 
            f_pattern =os.path.join(data_dir,'additional_SCA3_data', 'T1','{}','T1_MNI152NLin2009cSym', file_name)

        filepaths_clin = [f_pattern.format(sub) for sub in sorted(sub_list.tolist())]
        resp_data_clin = Parallel(n_jobs=-1,verbose =5)(delayed(load_file)(sub_path, mask_nii,resample, smooth, vox_size)for sub_path in filepaths_clin)
        resp_data_clin = np.stack(resp_data_clin, axis=0)
        
    if (featureset["BL_wholebrain_mod_gmv_and_wmv_raw"]) :
        file_name_gm = 'T1_BrainNorm_GMV.nii.gz'
        file_name_wm = 'T1_BrainNorm_WMV.nii.gz'
        
        if dataset == 'SCA1_Nijmegen' : 
            gm_pattern = os.path.join(data_dir, 'subjects_data','{}','ses-01','anat','T1_MNI152NLin2009cSym', file_name_gm)
            wm_pattern = os.path.join(data_dir, 'subjects_data','{}','ses-01','anat','T1_MNI152NLin2009cSym', file_name_wm)
        if dataset == 'SCA3_Mexico' : 
            gm_pattern =os.path.join(data_dir,'additional_SCA3_data', 'T1','{}','T1_MNI152NLin2009cSym', file_name_gm)
            wm_pattern =os.path.join(data_dir,'additional_SCA3_data', 'T1','{}','T1_MNI152NLin2009cSym', file_name_wm)
            
        resp_data_clin = Parallel(n_jobs=-1,verbose =5)(delayed(load_files_and_combine)(
            gm_pattern.format(sub), wm_pattern.format(sub), 
            mask_nii_gm, mask_nii_wm, mask_nii_combined,
            resample, smooth, vox_size) 
            for sub in sorted(sub_list.tolist()))
        resp_data_clin = np.stack(resp_data_clin, axis=0)

    if (featureset["BL_wholebrain_log_jacs_Z"]) :
        mask = dataio.fileio.load(mask_nii, mask=mask_nii, vol=False).T
        n_voxels = len(mask)
        n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)
        voxel_indices = list(range(n_voxels))

        Z_all = []
        for b in range(n_batches)[:]:
            Z_batch = []
            batch = f'batch_{b}'
           #grab Zs from normdata
            with open(os.path.join(proc_dir,'transfer_' + dataset, batch, 'norm_data_clin.pkl'), 'rb') as f:
                norm_data= pickle.load(f)
            norm_data.load_zscores(save_dir = os.path.join(proc_dir,'transfer_' + dataset, batch,'results'))
            
            for sub in sub_list.tolist() :
                idx = np.where(norm_data.subject_ids.values == sub)[0]
                sub_batch = norm_data.isel(observations=idx)
            
                Z_sub = pd.DataFrame(sub_batch.Z.values, columns=sub_batch.Z.response_vars.values)
                Z_batch.append(Z_sub)
            Z_batch = pd.concat(Z_batch)
            Z_all.append(Z_batch) 

        if len(Z_all)>0 :  
            resp_data_clin = pd.concat(Z_all, axis = 1) 
        else : 
            resp_data_clin = Z_all
            
            
    if (featureset["BL_wholebrain_mod_gmv_and_wmv_Z"]) :
        
        #grab GM Zs from normdata
        mask = dataio.fileio.load(mask_nii_gm, mask=mask_nii_gm, vol=False).T
        n_voxels = len(mask)
        n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)
        
        Z_all_gm = []
        for b in range(n_batches)[:]:
            Z_batch = []
            batch = f'batch_{b}'
            
            with open(os.path.join(proc_dir_gm,'transfer_' + dataset, batch, 'norm_data_clin.pkl'), 'rb') as f:
                norm_data= pickle.load(f)
            norm_data.load_zscores(save_dir = os.path.join(proc_dir_gm,'transfer_' + dataset, batch,'results'))
            
            for sub in sub_list.tolist() :
                idx = np.where(norm_data.subject_ids.values == sub)[0]
                sub_batch = norm_data.isel(observations=idx)
            
                Z_sub = pd.DataFrame(sub_batch.Z.values, columns=sub_batch.Z.response_vars.values)
                Z_batch.append(Z_sub)
            Z_batch = pd.concat(Z_batch)
            Z_all_gm.append(Z_batch) 

        Z_all_gm = np.array(pd.concat(Z_all_gm, axis=1))

        #grab GM Zs from normdata
        mask = dataio.fileio.load(mask_nii_wm, mask=mask_nii_wm, vol=False).T
        n_voxels = len(mask)
        n_batches = n_voxels // vox_batch_size + int(n_voxels % vox_batch_size != 0)

        Z_all_wm = []
        for b in range(n_batches)[:]:
            Z_batch = []
            batch = f'batch_{b}'
            
            with open(os.path.join(proc_dir_wm,'transfer_' + dataset, batch, 'norm_data_clin.pkl'), 'rb') as f:
                norm_data= pickle.load(f)
            norm_data.load_zscores(save_dir = os.path.join(proc_dir_wm,'transfer_' + dataset, batch,'results'))
            
            for sub in sub_list.tolist() :
                idx = np.where(norm_data.subject_ids.values == sub)[0]
                sub_batch = norm_data.isel(observations=idx)
            
                Z_sub = pd.DataFrame(sub_batch.Z.values, columns=sub_batch.Z.response_vars.values)
                Z_batch.append(Z_sub)
            Z_batch = pd.concat(Z_batch)
            Z_all_wm.append(Z_batch) 
        
        Z_all_wm = np.array(pd.concat(Z_all_wm, axis = 1))
            
        #combine GM and WM Zs with mask handling
        mask_combined = nib.load(mask_nii_combined)
        mask_combined = mask_combined.get_fdata().astype(bool)
        
        mask_gm = nib.load(mask_nii_gm)
        mask_gm = mask_gm.get_fdata().astype(bool)
        
        mask_wm = nib.load(mask_nii_wm)
        mask_wm = mask_wm.get_fdata().astype(bool)
        
        Z_all_combined = []
        
        for s in range(len(sub_list.tolist())):
            
            #re-project both Z vectors into nifti space
            Z_gm = Z_all_gm[s,:]
            Z_wm = Z_all_wm[s,:]

            #combine them
            Z_combined_vol = np.zeros_like(mask_gm).astype(float)
            Z_combined_vol[mask_gm] = Z_gm
            Z_combined_vol[mask_wm] = Z_wm

            #re-vectorize using combined mask
            Z_combined =  Z_combined_vol[mask_combined]
            Z_all_combined.append(Z_combined) 
                
        resp_data_clin = Z_all_combined
        
    #if we age-residualize, append the age column to what will be X, the transformer will take care of it
    if (featureset["BL_wholebrain_log_jacs_raw"] or featureset["BL_wholebrain_mod_gmv_and_wmv_raw"]):
        resp_data_clin = np.concatenate((resp_data_clin, df_clin["age"].to_numpy().reshape(-1, 1)), 1)
    
    df_sca.append(df_clin)
    y_sca.append(y_clin)
    X_sca.append(pd.DataFrame(resp_data_clin))

    
#Now, only get those necessary for the prediction
    
if train_and_test["test_Nijmegen_CV"]:
    X_all = X_sca[0]
    X_all = pd.DataFrame(X_all) 
    y_all = y_sca[0]
    df_test = df_sca[0] #we take the df for the end residuals graph
    
if train_and_test["test_Nijmegen_and_Mexico_CV"]:
    X_all = pd.concat([X_sca[0],X_sca[1]])
    X_all = pd.DataFrame(X_all) 
    y_all = pd.concat([y_sca[0], y_sca[1]])
    y_all = y_all.reset_index(drop=True) #otehrwsie will throw error 
    df_test =  pd.concat([df_sca[0],df_sca[1]]) #we take the df for the end residuals graph
    df_test = df_test.reset_index(drop=True)

    
#%%Optional - grab demographics

for dataset in datasets:
    df_clin = pd.read_csv(os.path.join(proc_dir, 'transfer_' + dataset, 'dem_patients.csv'))
    df_ctrl = pd.read_csv(os.path.join(proc_dir, 'transfer_' + dataset, 'dem_controls.csv'))
    
    print(f"{dataset} SCA patients")
    print(f"subjects total {len(df_clin.index.tolist())}")
    print(f"unique scanners {len(np.unique(df_clin["site"]))}")
    print(f"female {100*df_clin["sex"].isin(['F']).sum()/len(df_clin["sex"]):.1f} percent")
    print(f"male {100*df_clin["sex"].isin(['M']).sum()/len(df_clin["sex"]):.1f} percent")
    print(f"age mean {np.mean(df_clin["age"]):.1f}")
    print(f"age sd {np.std(df_clin["age"]):.1f}")
    print(f"SARA mean {np.mean(df_clin["SARA_score"]):.1f}")
    print(f"SARA sd {np.std(df_clin["SARA_score"]):.1f}")
    
    
    print(f"{dataset} controls")
    print(f"subjects total {len(df_ctrl.index.tolist())}")
    print(f"unique scanners {len(np.unique(df_ctrl["site"]))}")
    print(f"female {100*df_ctrl["sex"].isin(['F']).sum()/len(df_ctrl["sex"]):.1f} percent")
    print(f"male {100*df_ctrl["sex"].isin(['M']).sum()/len(df_ctrl["sex"]):.1f} percent")
    print(f"age mean {np.mean(df_ctrl["age"]):.1f}")
    print(f"age sd {np.std(df_ctrl["age"]):.1f}")
    
#%% Configure prediction pipeline

scaler = StandardScaler(with_mean = True, with_std = True)
splitter = LeaveOneOut()
inner_splitter = LeaveOneOut()

#transformer to safely manage age residualization within pipeline
class AgeResidualizer(BaseEstimator, TransformerMixin):
    def __init__(self, center_age=True):
        self.center_age = center_age

    def fit(self, X, y=None):
        
        age = X.iloc[:, -1].to_numpy().reshape(-1, 1)# last column of X is assumed to be age
        voxels = X.iloc[:, :-1].to_numpy()

        if self.center_age:
            self.age_mean_ = age.mean()
            age = age - self.age_mean_
        else:
            self.age_mean_ = 0.0

        Z = np.hstack([np.ones((age.shape[0], 1)), age])
        self.B_ = np.linalg.solve(Z.T @ Z, Z.T @ voxels)
        return self

    def transform(self, X):
        age = X.iloc[:, -1].to_numpy().reshape(-1, 1)
        voxels = X.iloc[:, :-1].to_numpy()

        age = age - self.age_mean_
        Z = np.hstack([np.ones((age.shape[0], 1)), age])
        voxels_res = voxels - Z @ self.B_
        
        return voxels_res #important: here we return only voxel columns and not the age one
    
#we have to account for bias in KRR by adding an intercept column
def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((X, intercept))

add_intercept_transformer = FunctionTransformer(add_intercept)
clf = KernelRidge()

if options["Xstandard"] :
    pipeline = Pipeline(steps =[
                ('resid', AgeResidualizer()), #this pipeline expects age as last column of X_train and X_test and will remove it after
    			('scaler', scaler),
                ('intercept', add_intercept_transformer),
                ('clf', clf)])
elif not options["Xstandard"] :
    pipeline = Pipeline(steps =[
                ('intercept', add_intercept_transformer),
                ('clf', clf)])

#%% Run prediction

fitted_models = []
ys_pred = []
ys_test = []
ys_train = []
Xs_train_final = []
Xs_test_final = []
feature_coefs = []
eval_scores = pd.DataFrame(columns = ['RMSE'])

for train_idx, test_idx in splitter.split(X_all):
    
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    ys_test.append(y_test)
    ys_train.append(y_train)
    
    #fit
    pipeline.fit(X_train, y_train)
    best_pipeline = pipeline
    krr_model = best_pipeline.named_steps['clf']
    
    #extract outputs
    X_train_final = best_pipeline[:-1].transform(X_train) #extract scaled and possibly selected features
    X_test_final = best_pipeline[:-1].transform(X_test)
    Xs_train_final.append(X_train_final)
    Xs_test_final.append(X_test_final)
    
    coefs = krr_model.dual_coef_
    coefs = np.dot(coefs, X_train_final) #this only works for linear kernel of krr
    feature_coefs.append(coefs)
    
    y_pred = best_pipeline.predict(X_test)
    fitted_models.append(best_pipeline)
    ys_pred.append(y_pred)
    
    rmse = root_mean_squared_error(y_true = y_test, y_pred = y_pred)
    eval_scores.loc[len(eval_scores)] = [rmse]
    
    
#%% Save outputs

#keep syntax straight
y_test = np.concatenate(ys_test)
y_pred = np.concatenate(ys_pred)
X_train_final = np.concatenate(Xs_train_final)
X_test_final = np.concatenate(Xs_test_final)

eval_scores['r2_score'] = [r2_score( y_true = y_test, y_pred = y_pred)] + [np.nan] * (len(eval_scores)-1)

correlation_coef, p_value =pearsonr(y_test, y_pred)
eval_scores['Pearson r'] = [correlation_coef] + [np.nan] * (len(eval_scores)-1)
eval_scores['Pearson p-value'] = ["{:.2e}".format(p_value)] + [np.nan] * (len(eval_scores)-1)
eval_scores.to_csv(os.path.join(final_dir,"_".join(['eval_scores']+[key for key, value in featureset.items() if value] +
                                                   [key for key, value in train_and_test.items() if value] + 
                                                   [key for key, value in options.items() if value] )+ extension +'.csv'), 
                                                   index = False, na_rep = '')
print(eval_scores.iloc[0,:])
with open(os.path.join(final_dir,'fitted_models.pkl'), 'wb') as file:
    pickle.dump(fitted_models, file)
with open(os.path.join(final_dir,'ys_pred.pkl'), 'wb') as file:
    pickle.dump(y_pred, file)    
with open(os.path.join(final_dir,'ys_test.pkl'), 'wb') as file:
    pickle.dump(y_test, file)  
with open(os.path.join(final_dir,'Xs_train.pkl'), 'wb') as file:
    pickle.dump(X_train_final, file) 
with open(os.path.join(final_dir,'Xs_test.pkl'), 'wb') as file:
    pickle.dump(X_test_final, file) 
with open(os.path.join(final_dir,'feature_coefs.pkl'), 'wb') as file:
    pickle.dump(feature_coefs, file)

#map selected voxelwise feature mean coefs across folds back onto a nifti
feature_coefs = [array[:-1] for array in feature_coefs] #remove last coefficient due to the added intercept (column of ones)
average_coefs = np.mean(feature_coefs, axis = 0)
ptksave(average_coefs, os.path.join(final_dir,'Signed_feature_coefs.nii.gz'), example=ex_nii, mask=mask_nii)

#Plot residuals to check systematic over- or under-fitting
diff_y = y_pred - y_test

fig = plt.figure(figsize=(8, 6))
y_name = ''.join(key for key, value in target.items() if value) #grab outcome name
plt.title(f'{y_name}' + '(pred) - ' +f'{y_name}'+'(true)', fontsize=16)
plt.plot(diff_y, marker='o', linestyle='')
plt.gca().axes.xaxis.set_ticks([])
plt.ylim(-20, 20)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
texts = []
buffer = 0.7
if target["Slope_BL_Y1_SARA"] : #adapt plotlims
     plt.ylim(-0.025, 0.025)
     buffer = 0.0008
for i in range(len(diff_y)):
    text = plt.text(i, diff_y[i]+buffer,f'{df_test["sub_id"][i]}', ha='center', va='baseline',fontsize=8)
plt.show()
fig.savefig(os.path.join(final_dir,'y_pred-y_true.png'), dpi = 300, bbox_inches = 'tight')

