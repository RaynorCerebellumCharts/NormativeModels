#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:35:32 2024

@author: alicha
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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


root_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia'
proc_dir = os.path.join(root_dir,'transfer_model_char/vox_powell_01/')
out_dir = os.path.join(root_dir,'clinical_predictions/')
extension = ''#if desired, add an extra suffix to output folder/filenames

mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'

#this is just a random subject to give nifti geometry when we want to map the 1D array onto a nifti 
ex_nii = os.path.join(root_dir, 'raynor_data/subjects_data/sub-002/ses-01/anat/sub-002_ses-01_acq-mprage_T1w.anat/T1_to_MNI_nonlin_jac.nii.gz')

#%% Choose options and load

target = {
    "BL_SARA" : 1,
    "Y1_SARA" : 0,
    "Y2_SARA" : 0, 
    "Slope_BL_Y1_SARA" : 0,
    "Slope_BL_Y2_SARA" : 0
    }

featureset = {
    "BL_wholebrain_Zs" : 1,
    "BL_wholebrain_jacs" : 0,
    "BL_ROI_Zs" : 0,
    "BL_ROI_vols" : 0,
    "BL_Y1_wholebrain_Z_diffs" : 0,
    "BL_Y1_slope_wholebrain_jacs" : 0,
    "BL_Y1_Z_diffs_ROI_vols" : 0,
    "BL_Y1_slope_ROI_vols" : 0,
    "BL_Y2_wholebrain_Z_diffs" : 0,
    "BL_Y2_slope_wholebrain_jacs" : 0,
    "BL_Y2_Z_diffs_ROI_vols" : 0,
    "BL_Y2_slope_ROI_vols" : 0
    }

options = {
    "gridsearch" : 1, 
}

if (featureset["BL_ROI_vols"] or #account for features with raw values for parcellated measures
    featureset["BL_wholebrain_jacs"] or
    featureset["BL_Y1_slope_ROI_vols"] or 
    featureset["BL_Y1_slope_wholebrain_jacs"] or
    featureset["BL_Y2_slope_ROI_vols"] or
    featureset["BL_Y2_slope_wholebrain_jacs"]):
    options["Xstandard"] = 1
else:
    options["Xstandard"] = 0
    
if (featureset["BL_ROI_vols"] or #account for features with outliers values for parcellated measures
    featureset["BL_ROI_Zs"] or
    featureset["BL_Y1_Z_diffs_ROI_vols"] or 
    featureset["BL_Y1_slope_ROI_vols"] or 
    featureset["BL_Y2_Z_diffs_ROI_vols"] or 
    featureset["BL_Y2_slope_ROI_vols"]):
    options["outliers-3std"] = 1

seed = 42       

filename = "_".join([key for key, value in target.items() if value] +
                     ["KRR_with"]+
                     [key for key, value in featureset.items() if value] +
                     [key for key, value in options.items() if value]+
                     [extension])

final_dir = os.path.join(out_dir,filename)
os.makedirs(final_dir, exist_ok=True)
os.chdir(final_dir)



#%% Grab X and y data

cov = pd.read_csv(os.path.join(root_dir,'raynor_data/SCA1_50s_covariates.csv'))
cov_sca = cov[cov['SCA1']==1]    
   

#Grab y
y_sca = cov_sca['SARA_score']
if target["BL_SARA"]:
    y_sca = cov_sca['SARA_score']
if target["Y1_SARA"]:
    y_sca = cov_sca['Y1_SARA']
if target["Y2_SARA"]:
    y_sca = cov_sca['Y2_SARA']
if target["Slope_BL_Y1_SARA"]:
    y_sca = cov_sca['Slope_BL_Y1_SARA']
elif target["Slope_BL_Y2_SARA"] :
    y_sca = cov_sca['Slope_BL_Y2_SARA']

#Grab X


def exclude_features_with_outliers(X):
    X_means = X.mean()
    X_stds = X.std()
    outliers = np.abs(X-X_means) > 3 * X_stds
    X_cleaned = X.loc[:, ~(outliers.any(axis = 0))]
    return X_cleaned

if featureset["BL_wholebrain_Zs"] : #jacobian Z_scores
    X_sca = pd.read_pickle(os.path.join(proc_dir,'Z_transfer.pkl'))

elif featureset["BL_wholebrain_jacs"] : #just jacobians
    X_sca = pd.read_pickle(os.path.join(root_dir, 'raynor_data/raw_resp_sca48s.pkl'))

elif featureset["BL_ROI_vols"] : 
    X_sca = pd.read_csv(os.path.join(root_dir,'raynor_data/freesurfer/All_Destrieux_and_SubCort_measures_CrossSec_ses-01.csv'))
    X_sca = X_sca.drop(columns = 'subject')
    X_sca = X_sca.reset_index (drop = True)
    if options["outliers-3std"]:
        X_sca = exclude_features_with_outliers(X_sca)

elif featureset["BL_ROI_Zs"]:
    X_sca = pd.read_csv(os.path.join(root_dir,'transfer_model_braincharts/lifespan_57K_82sites/Z_transfer_CrossSec_ses-01.csv'))
    X_sca = exclude_features_with_outliers(X_sca)
    if options["outliers-3std"]:
        X_sca = exclude_features_with_outliers(X_sca)
        
elif featureset["BL_Y1_wholebrain_Z_diffs"]: 
    X_sca = pd.read_pickle(os.path.join(proc_dir,'Z_diffs_SCA_cheat_ses-01_ses-02.pkl') )
   
elif featureset["BL_Y2_wholebrain_Z_diffs"]: 
    X_sca = pd.read_pickle(os.path.join(proc_dir,'Z_diffs_SCA_cheat_ses-01_ses-03.pkl'))

elif featureset["BL_Y1_slope_wholebrain_jacs"]: 
    X_sca = pd.read_pickle(os.path.join(root_dir,'raynor_data/Slope_jacs_SCA_ses-01_ses-02.pkl') )

elif featureset["BL_Y2_slope_wholebrain_jacs"]: 
    X_sca = pd.read_pickle(os.path.join(root_dir,'raynor_data/Slope_jacs_SCA_ses-01_ses-03.pkl') )
  
elif featureset["BL_Y1_Z_diffs_ROI_vols"]: 
    X_sca = pd.read_csv(os.path.join(root_dir,'transfer_model_braincharts/lifespan_57K_82sites/Z_diffs_SCA_cheat_ses-01_ses-02.csv') )
    if options["outliers-3std"]:
        X_sca = exclude_features_with_outliers(X_sca)
        
elif featureset["BL_Y2_Z_diffs_ROI_vols"]: 
    X_sca = pd.read_csv(os.path.join(root_dir,'transfer_model_braincharts/lifespan_57K_82sites/Z_diffs_SCA_cheat_ses-01_ses-03.csv') )
    if options["outliers-3std"]:
        X_sca = exclude_features_with_outliers(X_sca)
        
elif featureset["BL_Y1_slope_ROI_vols"]: 
    X_sca = pd.read_csv(os.path.join(root_dir,'raynor_data/Slope_FreesurferParcellated_SCA_ses-01_ses-02.csv') )
    if options["outliers-3std"]:
        X_sca = exclude_features_with_outliers(X_sca)
        
elif featureset["BL_Y2_slope_ROI_vols"]: 
    X_sca = pd.read_csv(os.path.join(root_dir,'raynor_data/Slope_FreesurferParcellated_SCA_ses-01_ses-03.csv') )
    if options["outliers-3std"]:
        X_sca = exclude_features_with_outliers(X_sca)


#%% Configure pipeline

scaler = StandardScaler(with_mean = True, with_std = True)
    
splitter = LeaveOneOut()
inner_splitter = LeaveOneOut()

#we have to account for bias in KRR by adding an intercept column
def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((X, intercept))

add_intercept_transformer = FunctionTransformer(add_intercept)

## Estimator

clf = KernelRidge()


if options["Xstandard"]:
    pipeline = Pipeline(steps =[
    			('scaler', scaler),
                ('intercept', add_intercept_transformer),
                ('clf', clf)])
elif not options["Xstandard"] :
    pipeline = Pipeline(steps =[
                ('intercept', add_intercept_transformer),
                ('clf', clf)])
else :
    print("Something went wrong with chosen pipeline options.")


if options["gridsearch"]:
    param_grid = {'clf__alpha':[1, 10, 100, 1000]}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_splitter, scoring='neg_mean_squared_error', n_jobs=-1)

#%% CV loop 

fitted_models = []
ys_pred = []
ys_test = []
ys_train = []
Xs_train_final = []
Xs_test_final = []
Xs_test_final_names = []
feature_coefs = []
eval_scores = pd.DataFrame(columns = ['RMSE'])

for train_idx, test_idx in splitter.split(X_sca):
    
    X_train, X_test = X_sca.iloc[train_idx], X_sca.iloc[test_idx]
    y_train, y_test = y_sca[train_idx], y_sca[test_idx]
    

    ys_test.append(y_test)
    ys_train.append(y_train)
    
    if options["gridsearch"]:
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_ 
    else :
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline

    krr_model = best_pipeline.named_steps['clf']

    if options["Xstandard"]:
        X_train_final = best_pipeline[:-1].transform(X_train) #extract scaled and possibly selected features
        X_test_final = best_pipeline[:-1].transform(X_test)
        X_test_final_names = X_train.columns
        Xs_test_final_names.append(X_test_final_names)
            
    else :
        X_train_final = X_train
        X_test_final = X_test
        
    Xs_train_final.append(X_train_final)
    Xs_test_final.append(X_test_final)
    
    coefs = krr_model.dual_coef_
    coefs = np.dot(coefs, X_train_final) #this only works for linear kernel of krr
    feature_coefs.append(coefs)
    
    y_pred = best_pipeline.predict(X_test)
    y_train_pred = best_pipeline.predict(X_train)    
    
    
    fitted_models.append(best_pipeline)
    ys_pred.append(y_pred)
    
    rmse = root_mean_squared_error(y_true = y_test, y_pred = y_pred)
    eval_scores.loc[len(eval_scores)] = [rmse]
    
    
    
#%% Save outputs

eval_scores['r2_score'] = [r2_score( y_true = ys_test, y_pred = ys_pred)] + [np.nan] * (len(eval_scores)-1)
correlation_coef, p_value =pearsonr(np.concatenate(ys_test), np.concatenate(ys_pred))
eval_scores['Pearson r'] = [correlation_coef] + [np.nan] * (len(eval_scores)-1)
eval_scores['Pearson p-value'] = ["{:.2e}".format(p_value)] + [np.nan] * (len(eval_scores)-1)
eval_scores.to_csv(os.path.join(final_dir,"_".join(['eval_scores']+[key for key, value in featureset.items() if value] +
                                                   [key for key, value in options.items() if value] )+ extension +'.csv'), na_rep = '')

with open(os.path.join(final_dir,'fitted_models.pkl'), 'wb') as file:
    pickle.dump(fitted_models, file)
with open(os.path.join(final_dir,'ys_pred.pkl'), 'wb') as file:
    pickle.dump(ys_pred, file)    
with open(os.path.join(final_dir,'ys_test.pkl'), 'wb') as file:
    pickle.dump(ys_test, file)  
with open(os.path.join(final_dir,'Xs_train.pkl'), 'wb') as file:
    pickle.dump(Xs_train_final, file) 
with open(os.path.join(final_dir,'Xs_test.pkl'), 'wb') as file:
    pickle.dump(Xs_test_final, file) 
with open(os.path.join(final_dir,'feature_coefs.pkl'), 'wb') as file:
    pickle.dump(feature_coefs, file)

#for wholebrain featuresets, map selected feature mean coefs across folds back onto a nifti
if (featureset["BL_wholebrain_Zs"] or featureset["BL_wholebrain_jacs"] or 
featureset["BL_Y1_wholebrain_Z_diffs"] or featureset["BL_Y1_slope_wholebrain_jacs"] or
featureset["BL_Y2_wholebrain_Z_diffs"] or featureset["BL_Y2_slope_wholebrain_jacs"]):
    
        feature_coefs = [array[:-1] for array in feature_coefs] #remove last coefficient for the column of ones
        average_coefs = np.mean(feature_coefs, axis = 0)
        threshold = np.partition(average_coefs, -1000)[-1000]
        avg_coefs_thresholded = np.where(average_coefs >= threshold, average_coefs, 0) 
        ptksave(np.abs(average_coefs), os.path.join(final_dir,'Abs_feature_coefs.nii.gz'), example=ex_nii, mask=mask_nii)
        ptksave(np.abs(avg_coefs_thresholded)*100, os.path.join(final_dir,'Abs100_feature_coefs_thresholded_1000.nii.gz'), example=ex_nii, mask=mask_nii)
        if target["Slope_BL_Y1_SARA"] or target["Slope_BL_Y2_SARA"]: # change scales of slopes for visualisation purposes
            ptksave(np.abs(avg_coefs_thresholded)*10000, os.path.join(final_dir,'Abs10000_feature_coefs_thresholded_1000.nii.gz'), example=ex_nii, mask=mask_nii)
       
#for parcellated featuresets, make graph of the top 15 contributing features with average weights across folds  
if (featureset["BL_ROI_Zs"] or featureset["BL_ROI_vols"] or
    featureset["BL_Y1_Z_diffs_ROI_vols"] or  featureset["BL_Y1_slope_ROI_vols"] or 
    featureset["BL_Y2_Z_diffs_ROI_vols"] or featureset["BL_Y2_slope_ROI_vols"] ) :

    fig = plt.figure(figsize=(8, 8))
        
    feature_coefs = [array[:-1] for array in feature_coefs]#remove last coefficient (i.e. the column of ones as intercept)
    average_coefs = np.mean(feature_coefs, axis = 0)
    feature_coef_pairs = pd.DataFrame({'Coefficient': average_coefs,'Feature':Xs_test_final_names[0].tolist()})
    top_features = feature_coef_pairs.reindex(feature_coef_pairs['Coefficient'].abs().sort_values(ascending=False).index).head(15)
    plt.barh(top_features.iloc[:,1], top_features.iloc[:,0], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 15 feature contributions', fontsize=16)
    plt.xlabel('Feature coefficients', fontsize=13)
    fig.savefig(final_dir + '/Feature_coefficients_top15.png', dpi = 300, bbox_inches = 'tight')
    
    
#Plot residuals to check systematic over- or under-fitting
diff_y = np.concatenate(ys_pred) - np.concatenate(ys_test)

fig = plt.figure(figsize=(8, 6))
y_name = ''.join(key for key, value in target.items() if value) #grab outcome name
plt.title(f'{y_name}' + '(pred) - ' +f'{y_name}'+'(true)', fontsize=16)
plt.plot(diff_y, marker='o', linestyle='')
plt.gca().axes.xaxis.set_ticks([])
plt.ylim(-20, 20)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
texts = []
buffer = 0.7
if target["Slope_BL_Y1_SARA"] or target["Slope_BL_Y2_SARA"] :
     plt.ylim(-0.025, 0.025)
     buffer = 0.0008
for i in range(len(diff_y)):
    text = plt.text(i, diff_y[i]+buffer,f'{cov_sca["subject"][i]}', ha='center', va='baseline',fontsize=8)

fig.savefig(final_dir + '/y_pred-y_true.png', dpi = 300, bbox_inches = 'tight')






