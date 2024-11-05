#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:58:05 2024

@author: alicha
"""
import os
import pandas as pd
from pcntoolkit import dataio
import glob


sessions = ['ses-01', 'ses-02', 'ses-03']
session_pairs = [('ses-01','ses-02'),('ses-01','ses-03')]

root_dir = '/project/4290000.01/alicha/Cerebellar_Ataxia'
data_dir = os.path.join(root_dir,'raynor_data')

df_dem = pd.read_csv(os.path.join(data_dir,'SCA1_50s_covariates.csv'))
df_sca = df_dem.loc[df_dem['SCA1'] == 1]


#%%wholebrain jacs slope - with fsl processing

mask_nii = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
proc_dir = os.path.join(data_dir,'subjects_data')
nii_suffix = 'nonlin_jac.nii.gz'

    
for pair in session_pairs : 
    dataframes = {}
    for session in pair:
        df_sca_ses = pd.DataFrame()
        
        excluded_subjects = []
        if session  == 'ses-01':
            excluded_subjects.append('sub-xxx') #session-specific exclusions for FSL processing
            
        elif session  == 'ses-02':
            excluded_subjects.append('sub-xxx')
            
        elif session  == 'ses-03':
            excluded_subjects.append('sub-xxx')
            
        df_sca_ses = df_sca[~df_sca['subject'].isin(excluded_subjects)]
        df_sca_ses = df_sca_ses.reset_index(drop=True)
        dataframes[session] = df_sca_ses

    df_sca_v1 = dataframes[pair[0]]
    df_sca_v2 = dataframes[pair[1]]
    
    #take overlapping subject indices
    common_df_sca_v1 = df_sca_v1[df_sca_v1['subject'].isin(df_sca_v2['subject'])].index
    common_df_sca_v2 = df_sca_v2[df_sca_v2['subject'].isin(df_sca_v1['subject'])].index
    
    df_sca_v1 = df_sca_v1.iloc[common_df_sca_v1].reset_index(drop=True)
    df_sca_v2 = df_sca_v2.iloc[common_df_sca_v2].reset_index(drop=True)
    
    x = []
    for s in df_sca_v1['subject']:
        if pair == ('ses-01','ses-02'):
            time = df_sca_v1[df_sca_v1['subject'] == s]['Time_BL_Y1'].iloc[0]
        if pair == ('ses-01','ses-03'):
            time = df_sca_v1[df_sca_v1['subject'] == s]['Time_BL_Y2'].iloc[0]
        jac_path_v1 = glob.glob(os.path.join(proc_dir, s ,pair[0],'anat','*'+'.anat', '*'+ nii_suffix))[0]
        jac_path_v2 = glob.glob(os.path.join(proc_dir, s ,pair[1],'anat','*'+'.anat', '*'+ nii_suffix))[0]
        flat_sub_v1 = dataio.fileio.load(jac_path_v1, mask=mask_nii, vol=False).T
        flat_sub_v2 = dataio.fileio.load(jac_path_v2, mask=mask_nii, vol=False).T
        flat_sub_slope = (flat_sub_v2 - flat_sub_v1)/time
        x.append(flat_sub_slope)
    df_x = pd.DataFrame(x)
    dataio.fileio.save(df_x.values,os.path.join(data_dir,'Slope_jacs_SCA_'+pair[0]+'_'+pair[1]+'.pkl'))

#%%parcellated measures slope - with freesurfer processing


for pair in session_pairs : 
    dataframes = {}
    for session in pair:
        df_sca_ses = pd.DataFrame()
        excluded_subjects = []
        if session== 'ses-01':
            excluded_subjects.append('sub-xxx') #session-specific exclusion for Freesurfer processing
            
        elif session == 'ses-02':
            excluded_subjects.append('sub-xxx')
            
        elif session == 'ses-03':
            excluded_subjects.append('sub-xxx')
        
        df_sca_ses = df_sca[~df_sca['subject'].isin(excluded_subjects)]
        df_sca_ses = df_sca_ses.reset_index(drop=True)
        dataframes[session] = df_sca_ses

    df_sca_v1 = dataframes[pair[0]]
    df_sca_v2 = dataframes[pair[1]]
    
    #take overlapping subject indices
    common_df_sca_v1 = df_sca_v1[df_sca_v1['subject'].isin(df_sca_v2['subject'])].index
    common_df_sca_v2 = df_sca_v2[df_sca_v2['subject'].isin(df_sca_v1['subject'])].index
    
    df_sca_v1 = df_sca_v1.iloc[common_df_sca_v1].reset_index(drop=True)
    df_sca_v2 = df_sca_v2.iloc[common_df_sca_v2].reset_index(drop=True)

    measures_v1 = pd.read_csv(os.path.join(data_dir,'freesurfer','All_Destrieux_and_SubCort_measures_Long_'+ pair[0]+'.csv'))
    measures_v2 = pd.read_csv(os.path.join(data_dir,'freesurfer','All_Destrieux_and_SubCort_measures_Long_'+ pair[1]+'.csv'))
    measures_v1 = measures_v1[measures_v1['subject'].isin(df_sca_v1['subject'])].reset_index(drop=True)
    measures_v2 = measures_v2[measures_v2['subject'].isin(df_sca_v2['subject'])].reset_index(drop=True)

    x = []
    time = []
    for s in df_sca_v1['subject']:
        if pair == ('ses-01','ses-02'):
            time = df_sca_v1[df_sca_v1['subject'] == s]['Time_BL_Y1'].iloc[0]
        if pair == ('ses-01','ses-03'):
            time = df_sca_v1[df_sca_v1['subject'] == s]['Time_BL_Y2'].iloc[0]
        
        measures_sub_slope = (measures_v1[measures_v1['subject'] == s].iloc[:,1:] - measures_v2[measures_v2['subject'] == s].iloc[:,1:])/time
        x.append(measures_sub_slope)
    df_x = pd.concat(x, axis=0)
    df_x.to_csv(os.path.join(data_dir, 'Slope_FreesurferParcellated_SCA_' +pair[0]+'_'+pair[1]+'.csv'),index= False)
