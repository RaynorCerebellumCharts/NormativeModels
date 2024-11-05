#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:45:03 2024

@author: alicha
"""

import os
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import xarray as xr
from pcntoolkit.normative import predict, evaluate
from pcntoolkit.util.utils import create_design_matrix
import itertools


root_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia/transfer_model_braincharts/'
os.chdir('/project/4290000.01/alicha/Cerebellar_Ataxia/transfer_model_braincharts/')
data_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia/raynor_data/'

# which model do we wish to use?
work_dir = os.path.join(root_dir, 'lifespan_57K_82sites')
site_names = 'site_ids_ct_82sites.txt'

sessions = ['ses-01', 'ses-02', 'ses-03']

design = 'Long' #CrossSec or Long  - this matches the suffix we previously gave for Freesurfer values .csv filenames
session = 'ses-01' # session of interest

if design == 'Long': # in case there aren't enough healthy subjects to have both a site adaptation set and a Z-diff estimation set, we can "cheat"
    option = '_cheat'
    outputprefix = 'predict_'
else:
    option = ''
    outputprefix = 'transfer_'

extension = design + '_' + session + option
outputsuffix = outputprefix + extension


# load a set of site ids from this model. This must match the training data
with open(os.path.join(work_dir, site_names)) as f:
    site_ids_tr = f.read().splitlines()

 
test_data = os.path.join(data_dir,'freesurfer','All_Destrieux_and_SubCort_measures_'+ design + '_' + session +'.csv')

df_resp = pd.read_csv(test_data)


excluded_subjects = [] # possible exclusions

#loading demographics
df_dem = pd.read_csv(os.path.join(data_dir,'SCA1_50s_covariates.csv'))

# delete rows with bad quality/Nan subjects
df_dem = df_dem[~df_dem['subject'].isin(excluded_subjects)]
df_dem = df_dem.reset_index(drop=True)

df_te = df_dem.loc[df_dem['SCA1'] == 1]
df_ad =  df_dem.loc[df_dem['SCA1'] == 0]        

df_resp_te = df_resp[df_resp['subject'].isin(df_te['subject'])]
df_resp_ad = df_resp[df_resp['subject'].isin(df_ad['subject'])]


# extract a list of unique site ids from the test set
site_ids_te =  sorted(set(df_te['site'].to_list()))
site_ids_ad =  sorted(set(df_ad['site'].to_list()))

if not all(elem in site_ids_ad for elem in site_ids_te):
    print('Warning: some of the testing sites are not in the adaptation data')

#take all column names of the freesurfer extarction
idp_ids = df_resp_te.columns[1:].tolist()

# which data columns do we wish to use as covariates? 
if session == 'ses-01':
    cols_cov = ['age_baseline','sex']
if session == 'ses-02':
    cols_cov = ['age_Y1','sex']
if session == 'ses-03':
    cols_cov = ['age_Y2','sex']

# limits for cubic B-spline basis 
xmin = -5 
xmax = 110

# Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
outlier_thresh = 7


#%% Cross-sectional transfer or predict 

for idp_num, idp in enumerate(idp_ids): 
    
    idp_dir = os.path.join(work_dir, idp)
    if os.path.isdir(idp_dir):
        print('Running IDP', idp_num, idp, ':')
        os.chdir(idp_dir)
        proc_dir = os.path.join(idp_dir, outputsuffix)
        if not os.path.isdir(proc_dir):
            os.mkdir(proc_dir)
        # extract and save the response variables for the test set
        y_te = df_resp_te[idp].to_numpy()
        
        # save the response variables
        resp_file_te = os.path.join(proc_dir, 'resp_te'+extension+'.txt') 
        np.savetxt(resp_file_te, y_te)
            
        # configure and save the design matrix
        cov_file_te = os.path.join(proc_dir, 'cov_bspline_te'+extension+'.txt')
        X_te = create_design_matrix(df_te[cols_cov], 
                                    site_ids = df_te['site'],
                                    all_sites = site_ids_tr,
                                    basis = 'bspline', 
                                    xmin = xmin, 
                                    xmax = xmax)
        np.savetxt(cov_file_te, X_te)
        
        # check whether all sites in the test set are represented in the training set
        if all(elem in site_ids_tr for elem in site_ids_te):
            print('All sites are present in the training data - predicting')
            # configure and save the design matrix
            cov_file_te = os.path.join(proc_dir, 'cov_bspline_te'+extension+'.txt')
            X_te = create_design_matrix(df_te[cols_cov], 
                                        site_ids = df_te['site'],
                                        all_sites = site_ids_tr,
                                        basis = 'bspline', 
                                        xmin = xmin, 
                                        xmax = xmax)
            np.savetxt(cov_file_te, X_te)
            
            # just make predictions
            yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg='blr', 
                                        respfile=resp_file_te, 
                                        model_path=os.path.join(idp_dir,'Models'),
                                        outputsuffix = outputsuffix,
                                        save_path = proc_dir)
        elif option == '_cheat':
            print('Keeping adaptation set for Z-diffs estimating and ignoring site effect difference - predicting')
            
            #cheat for site
            df_te.loc[:,'site'] = 'ABCD_01'
            df_ad.loc[:,'site'] = 'ABCD_01'
            
            #just so that I can directly grab them when calculating Zdiffs
            df_te.to_csv(os.path.join(work_dir, 'dem_'+extension+'SCA.txt'),sep=' ', index=False)
            df_ad.to_csv(os.path.join(work_dir, 'dem_'+extension+'HC.txt'),sep=' ', index=False)
                
                
            X_te = create_design_matrix(df_te[cols_cov], site_ids = df_te['site'], all_sites = site_ids_tr,
                                        basis = 'bspline', xmin = xmin, xmax = xmax)
            X_ad = create_design_matrix(df_ad[cols_cov], site_ids = df_ad['site'], all_sites=site_ids_tr,
                                        basis = 'bspline', xmin = xmin, xmax = xmax)

            cov_file_ad = os.path.join(proc_dir, 'cov_bspline_ad'+extension+'.txt')          
            np.savetxt(cov_file_ad, X_ad)
            
            # save the responses for the adaptation data
            resp_file_ad = os.path.join(proc_dir, 'resp_ad'+extension+'.txt') 
            y_ad = df_resp_ad[idp].to_numpy()
            np.savetxt(resp_file_ad, y_ad)
            
            yhat_te, s2_te, Z_te, y_te = predict(cov_file_te, 
                                        alg='blr', 
                                        respfile=resp_file_te, 
                                        model_path=os.path.join(idp_dir,'Models'),
                                        outputsuffix = outputsuffix + 'SCA',
                                        save_path = proc_dir,
                                        return_y = True)
            yhat_HC, s2_HC, Z_HC, y_ad = predict(cov_file_ad, 
                                        alg='blr', 
                                        respfile=resp_file_ad, 
                                        model_path=os.path.join(idp_dir,'Models'),
                                        outputsuffix = outputsuffix + 'HC',
                                        save_path = proc_dir,
                                        return_y = True)
            
            
            
        else:
            print('Some sites missing from the training data - adapting & transferring')
            # configure and save the design matrix
            
            X_te = create_design_matrix(df_te[cols_cov], 
                                        site_ids = df_te['site'],
                                        all_sites = site_ids_tr,
                                        basis = 'bspline', 
                                        xmin = xmin, 
                                        xmax = xmax)
            cov_file_te = os.path.join(proc_dir, 'cov_bspline_te'+extension+'.txt')
            np.savetxt(cov_file_te, X_te)
            
            # save the covariates for the adaptation data
            X_ad = create_design_matrix(df_ad[cols_cov], 
                                        site_ids = df_ad['site'],
                                        all_sites = site_ids_tr,
                                        basis = 'bspline', 
                                        xmin = xmin, 
                                        xmax = xmax)
            cov_file_ad = os.path.join(proc_dir, 'cov_bspline_ad'+extension+'.txt')          
            np.savetxt(cov_file_ad, X_ad)
            
            # save the responses for the adaptation data
            resp_file_ad = os.path.join(proc_dir, 'resp_ad'+extension+'.txt') 
            y_ad = df_resp_ad[idp].to_numpy()
            np.savetxt(resp_file_ad, y_ad)
           
            # save the site ids for the adaptation data
            sitenum_file_ad = os.path.join(proc_dir, 'sitenum_ad'+extension+'.txt') 
            site_num_ad = df_ad['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_ad, site_num_ad)
            
            # save the site ids for the test data 
            sitenum_file_te = os.path.join(proc_dir, 'sitenum_te'+extension+'.txt')
            site_num_te = df_te['sitenum'].to_numpy(dtype=int)
            np.savetxt(sitenum_file_te, site_num_te)
             
            yhat_te, s2_te, Z = predict(cov_file_te, 
                                        alg = 'blr', 
                                        respfile = resp_file_te, 
                                        model_path = os.path.join(idp_dir,'Models'),
                                        adaptrespfile = resp_file_ad,
                                        adaptcovfile = cov_file_ad,
                                        adaptvargroupfile = sitenum_file_ad,
                                        testvargroupfile = sitenum_file_te,
                                        outputsuffix = outputsuffix,
                                        save_path = proc_dir)

#%% Evaluate transfer
# which sex do we want to plot? 
sex = 1 # 1 = male 0 = female
if sex == 1: 
    clr = 'blue';
else:
    clr = 'red'

# create dummy data for visualisation
print('configuring dummy data ...')
xx = np.arange(xmin, xmax, 0.5)
X0_dummy = np.zeros((len(xx), 2))
X0_dummy[:,0] = xx
X0_dummy[:,1] = sex

# create the design matrix
X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_ids=None, all_sites=site_ids_tr)

# save the dummy covariates
cov_file_dummy = os.path.join(root_dir,'cov_bspline_dummy_mean.txt')
np.savetxt(cov_file_dummy, X_dummy)


sns.set(style='whitegrid')

for idp_num, idp in enumerate(idp_ids): 
    print('Running IDP', idp_num, idp, ':')
    idp_dir = os.path.join(work_dir, idp)
    os.chdir(idp_dir)
    
    # load the true data points
    yhat_te = pd.read_csv(os.path.join(idp_dir, 'yhat_transfer'+ design + session +'.txt'))
    s2_te = pd.read_csv(os.path.join(idp_dir, 'ys2_transfer'+ design + session +'.txt'))
    y_te = pd.read_csv(os.path.join(idp_dir, 'resp_te'+ design + '_' + session +'.txt'))
            
    # set up the covariates for the dummy data
    print('Making predictions with dummy covariates (for visualisation)')
    yhat, s2 = predict(cov_file_dummy, 
                       alg = 'blr', 
                       respfile = None, 
                       model_path = os.path.join(idp_dir,'Models'), 
                       outputsuffix = '_dummy')
    
    # load the normative model
    with open(os.path.join(idp_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
        nm = pickle.load(handle) 
    
    # get the warp and warp parameters
    W = nm.blr.warp
    warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        
    # first, we warp predictions for the true data and compute evaluation metrics
    med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
    med_te = med_te[:, np.newaxis]
    print('metrics:', evaluate(y_te, med_te))
    
    # then, we warp dummy predictions to create the plots
    med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
    
    # extract the different variance components to visualise
    beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
    s2n = 1/beta # variation (aleatoric uncertainty)
    s2s = s2-s2n # modelling uncertainty (epistemic uncertainty)
    
    # plot the data points
    y_te_rescaled_all = np.zeros_like(y_te)
    for sid, site in enumerate(site_ids_te):
        # plot the true test data points 
        if all(elem in site_ids_tr for elem in site_ids_te):
            # all data in the test set are present in the training set
            
            # first, we select the data points belonging to this particular site
            idx = np.where(np.bitwise_and(X_te[:,2] == sex, X_te[:,sid+len(cols_cov)+1] !=0))[0]
            if len(idx) == 0:
                print('No data for site', sid, site, 'skipping...')
                continue
            
            # then directly adjust the data
            idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
            y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])
        else:
            # we need to adjust the data based on the adaptation dataset 
            
            # first, select the data point belonging to this particular site
            idx = np.where(np.bitwise_and(X_te[:,2] == sex, (df_te['site'] == site).to_numpy()))[0]
            
            # load the adaptation data
            y_ad = pd.read_csv(os.path.join(idp_dir, 'resp_ad'+ design + '_' + session +'.txt'))
            X_ad = pd.read_csv(os.path.join(idp_dir, 'cov_bspline_ad'+ design + '_' + session +'.txt'))
            idx_a = np.where(np.bitwise_and(X_ad[:,2] == sex, (df_ad['site'] == site).to_numpy()))[0]
            if len(idx) < 2 or len(idx_a) < 2:
                print('Insufficent data for site', sid, site, 'skipping...')
                continue
            
            # adjust and rescale the data
            y_te_rescaled, s2_rescaled = nm.blr.predict_and_adjust(nm.blr.hyp, 
                                                                   X_ad[idx_a,:], 
                                                                   np.squeeze(y_ad[idx_a]), 
                                                                   Xs=None, 
                                                                   ys=np.squeeze(y_te[idx]))
        # plot the (adjusted) data points
        plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color=clr, alpha = 0.1)
       
    # plot the median of the dummy data
    plt.plot(xx, med, clr)
    
    # fill the gaps in between the centiles
    junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
    junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
    junk, pr_int99 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
    plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
    plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
    plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)
            
    # make the width of each centile proportional to the epistemic uncertainty
    junk, pr_int25l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.25,0.75])
    junk, pr_int95l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.05,0.95])
    junk, pr_int99l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.01,0.99])
    junk, pr_int25u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.25,0.75])
    junk, pr_int95u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.05,0.95])
    junk, pr_int99u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.01,0.99])    
    plt.fill_between(xx, pr_int25l[:,0], pr_int25u[:,0], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int95l[:,0], pr_int95u[:,0], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int99l[:,0], pr_int99u[:,0], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int25l[:,1], pr_int25u[:,1], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int95l[:,1], pr_int95u[:,1], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int99l[:,1], pr_int99u[:,1], alpha = 0.3,color=clr)

    # plot actual centile lines
    plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)
    
    plt.xlabel('Age')
    plt.ylabel(idp) 
    plt.title(idp)
    plt.xlim((0,90))
    plt.savefig(os.path.join(idp_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
    plt.show()
    
    
    
#%% Collect cross-sectional measures (mostly Z) in clean csv file across idps

measures = ['Z']

measures_all= {}


for measure in measures:
    measure_all_idp = {}
    for idp_num, idp in enumerate(idp_ids): 
    
        idp_dir = os.path.join(work_dir, idp)
    
        if os.path.isdir(idp_dir):
            print('Grabbing IDP measure', measure, idp_num, idp, session, ':')
            os.chdir(idp_dir)
            proc_dir = os.path.join(idp_dir, outputsuffix)
            measure_all_idp[f"{idp}"] = pd.read_csv(os.path.join(proc_dir, measure + '_' + outputsuffix.replace('_', '') + '.txt'),header = None)
              
    measures_all[measure] = measure_all_idp    
   
    
Z_arr = pd.concat(measures_all['Z'], axis= 1)
Z_arr.columns = Z_arr.columns.droplevel(1)
Z_arr.to_csv(os.path.join(work_dir, 'Z_transfer_' +extension+'.csv'), index=False)



#%% Compute Z-diffs


session_pairs = list(itertools.combinations(sessions, 2))

extension1 = 'Long' 
extension3 = '_cheat'
outputprefix = 'predict_'
measures = ['yhat', 'Y']


for pair in session_pairs : 

    # take measures of interest at both timepoints   
    
    df_sca_v1 = pd.read_csv(os.path.join(work_dir,'dem_'+ extension1 + '_'+ pair[0]+'_cheatSCA.txt'), sep=' ')
    df_HC_v1 = pd.read_csv(os.path.join(work_dir,'dem_'+ extension1 + '_'+ pair[0]+'_cheatHC.txt'), sep=' ')

    df_sca_v2 = pd.read_csv(os.path.join(work_dir,'dem_'+ extension1 + '_'+ pair[1]+'_cheatSCA.txt'), sep=' ')
    df_HC_v2 = pd.read_csv(os.path.join(work_dir,'dem_'+ extension1 + '_'+ pair[1]+'_cheatHC.txt'), sep=' ')
    
    
    #take overlapping subject indices
    common_df_sca_v1 = df_sca_v1[df_sca_v1['subject'].isin(df_sca_v2['subject'])].index
    common_df_HC_v1= df_HC_v1[df_HC_v1['subject'].isin(df_HC_v2['subject'])].index
    
    common_df_sca_v2 = df_sca_v2[df_sca_v2['subject'].isin(df_sca_v1['subject'])].index
    common_df_HC_v2 = df_HC_v2[df_HC_v2['subject'].isin(df_HC_v1['subject'])].index  
    

    # take measures of interest at both timepoints for overlapping subjects 
    measures_sca = {}
    measures_hc = {}
    for v in pair:
        measures_sca_v = {}
        measures_hc_v = {}
        outputsuffixv= outputprefix + extension1 + '_' + v + extension3
        
        
        for measure in measures:
            measure_sca_idp = {}
            measure_hc_idp = {}
            for idp_num, idp in enumerate(idp_ids): 
            
                idp_dir = os.path.join(work_dir, idp)
        
                if os.path.isdir(idp_dir):
                    print('Running IDP Z-diffs', measure, idp_num, idp, pair, ':')
                    os.chdir(idp_dir)
                    proc_dir_v = os.path.join(idp_dir, outputsuffixv)
                    measure_sca_idp[f"{idp}"] = pd.read_csv(os.path.join(proc_dir_v, measure + '_' + outputsuffixv.replace('_', '') + 'SCA.txt'),header = None)
                    measure_hc_idp[f"{idp}"] = pd.read_csv(os.path.join(proc_dir_v, measure + '_' + outputsuffixv.replace('_', '') + 'HC.txt'),header = None)
                    
                    if v == pair[0]:
                        measure_sca_idp[f"{idp}"]  = measure_sca_idp[f"{idp}"].iloc[common_df_sca_v1].reset_index(drop=True)
                        measure_hc_idp[f"{idp}"] = measure_hc_idp[f"{idp}"].iloc[common_df_HC_v1].reset_index(drop=True)
                        
                    elif v == pair[1]:
                        measure_sca_idp[f"{idp}"]  = measure_sca_idp[f"{idp}"].iloc[common_df_sca_v2].reset_index(drop=True)
                        measure_hc_idp[f"{idp}"] = measure_hc_idp[f"{idp}"].iloc[common_df_HC_v2].reset_index(drop=True)
                        
                measures_sca_v[measure] = measure_sca_idp    
                measures_hc_v[measure] = measure_hc_idp 
        measures_sca[v] = measures_sca_v
        measures_hc[v] = measures_hc_v    
   
    
    #wrap neuroimaging measures

    xr_sca_v1 = xr.concat([
                xr.DataArray(pd.concat(measures_sca[pair[0]][measures[0]], axis = 1), dims=['subject', 'idps'], name=measures[0]),
                xr.DataArray(pd.concat(measures_sca[pair[0]][measures[1]], axis = 1), dims=['subject', 'idps'], name=measures[1]),
                ],
                pd.Index(measures, name = 'features'))
    
    xr_sca_v2 = xr.concat([
                xr.DataArray(pd.concat(measures_sca[pair[1]][measures[0]], axis = 1), dims=['subject', 'idps'], name=measures[0]),
                xr.DataArray(pd.concat(measures_sca[pair[1]][measures[1]], axis = 1), dims=['subject', 'idps'], name=measures[1]),
                ],
                pd.Index(measures, name = 'features'))
    
    xr_HC_v1 = xr.concat([
                xr.DataArray(pd.concat(measures_hc[pair[0]][measures[0]], axis = 1), dims=['subject', 'idps'], name=measures[0]),
                xr.DataArray(pd.concat(measures_hc[pair[0]][measures[1]], axis = 1), dims=['subject', 'idps'], name=measures[1]),
                ],
                pd.Index(measures, name = 'features'))
    
    xr_HC_v2 = xr.concat([
                xr.DataArray(pd.concat(measures_sca[pair[1]][measures[0]], axis = 1), dims=['subject', 'idps'], name=measures[0]),
                xr.DataArray(pd.concat(measures_sca[pair[1]][measures[1]], axis = 1), dims=['subject', 'idps'], name=measures[1]),
                ],
                pd.Index(measures, name = 'features'))

    
    xr_sca = xr.concat([xr_sca_v1,xr_sca_v2],  pd.Index(['V1', 'V2'], name = 'visit'))
    
    xr_HC = xr.concat([xr_HC_v1,xr_HC_v2],  pd.Index(['V1', 'V2'], name = 'visit'))
    
 
    
    
    # 1) Estimate the variance in healthy controls
    HC_sqrt = np.sqrt(
        pd.DataFrame(
    (
            (xr_HC.sel(visit='V2', features='Y')- xr_HC.sel(visit='V2', features='yhat'))
            -
            (xr_HC.sel(visit='V1', features='Y') - xr_HC.sel(visit='V1', features='yhat'))
    
    ).var(axis=0).to_pandas() #variance across subjects for each feature
    ).T
    )
    
    
    # 2) Substract the two patient visits
    sca_diff =((xr_sca.sel(visit='V2', features='Y') - xr_sca.sel(visit='V2', features='yhat'))
                    -
                (xr_sca.sel(visit='V1', features='Y') - xr_sca.sel(visit='V1', features='yhat'))
                ).to_pandas()
    
    # 3) Compute the zdiff score
    sca_zdiff = sca_diff.div(HC_sqrt.squeeze(), axis='columns')

    sca_zdiff.columns = sca_zdiff.columns.droplevel(1)
    
    sca_zdiff.to_csv(os.path.join(work_dir, 'Z_diffs_SCA_cheat_'+ pair[0] + '_' + pair[1] +'.csv'), index=False)

