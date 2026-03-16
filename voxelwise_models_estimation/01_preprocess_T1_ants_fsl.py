#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:06:33 2025

@author: Andre Marquand and YaPing Wang, adapted by Alice Chavanne

This script uses a separate utility script to submit preprocessing jobs to a local slurm cluster, and is only useful
to show image preprocessing pipeline and parameters.
Adaptation to other computing clusters or run on local machine (not recommended) is necessary.
"""

from __future__ import print_function
import os
import glob
import subprocess
import nibabel as nib
import numpy as np


cohort = 'cohortname'
root_dir = '/path/to/root/dir/'
cluster_log_dir = os.path.join(root_dir, 'logs')

memory = '10gb'
time = "12:00:00"

cmd_qsub_base = ['/path/to/slurmjob_submission_script.py',
                 '-length', time,
                 '-memory', memory,
                 '-logfiledir', cluster_log_dir,
                ]

#specify template
tf_dir = '/path/to/templateflow/'
template = 'MNI152NLin2009cSym'
num_threads = 5
log_prefix = f'{cohort}_ants_'

tpl_dir = os.path.join(tf_dir, f'tpl-{template}')
tpl_file = os.path.join(tpl_dir, f'tpl-{template}_res-1_T1w.nii.gz')
tpl_mask = os.path.join(tpl_dir, f'tpl-{template}_res-1_desc-brain_mask.nii.gz')

#grab subject directories
sub_dirs = sorted(glob.glob(os.path.join(root_dir,cohort, 'subject', 'anat','T1' ,'*')))

#%% Run ANTS : brain extraction, regsitration, and creation of Jacobian determinants niftis
for s in sub_dirs[:]:
    subid = os.path.splitext(os.path.basename(os.path.dirname(s)))[0]
    log_name = log_prefix + subid
    t1_file = sorted(glob.glob(os.path.join(s, 'anat', 'T1.nii.gz')))[0]
    ants_dir = os.path.join(s, 'anat','T1_ants_'+str(template))

    be_file = os.path.join(ants_dir, 'T1_BrainExtractionBrain.nii.gz')
    jac_file = os.path.join(ants_dir, 'T1_BrainNorm_Jacobian.nii.gz')
        
    os.makedirs(ants_dir, exist_ok=True)

    # brain extraction
    ants_be_cmd = ['module load ANTs ; ',
                   'antsBrainExtraction.sh -d 3 ', 
                   #'-k 1', # keep temporary files
                   '-a', t1_file, 
                   '-e', tpl_file,
                   '-m', tpl_mask, 
                   '-o', os.path.join(ants_dir, 'T1_'),
                   ';'
                   ]
    # registration
    ants_rg_cmd = ['antsRegistrationSyN.sh -d 3 ',
                   '-f', tpl_file, 
                   '-m', be_file,
                   '-o', os.path.join(ants_dir,'T1_BrainNorm_'),
                   '-x', tpl_mask,
                   '-n', str(num_threads),
                   ';'
                  ]
    # Compose transform - combine affine and nonlinear transform 
    ants_cp_cmd = ['ComposeMultiTransform 3 ',
                   os.path.join(ants_dir,'T1_BrainNorm_NewAffWarp.nii.gz'),
                   '-R', tpl_file,
                   '-i', os.path.join(ants_dir, 'T1_BrainNorm_1Warp.nii.gz'),
                   '-i', os.path.join(ants_dir, 'T1_BrainNorm_0GenericAffine.mat'),
                    ';']
    
    # create Jacobian determinant map from combined transform
    ants_jc_cmd = ['CreateJacobianDeterminantImage 3',
                    os.path.join(ants_dir, 'T1_BrainNorm_AffWarp.nii.gz'), 
                    os.path.join(ants_dir, 'T1_BrainNorm_Jacobian.nii.gz'),
                    ' 1 1'] #log, geometric jacobian
    
    # create command string
    cmd_str = f'"{cmd_ants}"'
    
    cmd_qsub = cmd_qsub_base + ['-name', log_name,
                                '-command', str(cmd_str),
                                '-clusterspec','slurm',
                                '-scriptfile', os.path.join(ants_dir,log_name + '.sh')
                               ]

    subprocess.Popen(' '.join(cmd_qsub), shell=True)

#%% Run FSL to get segmentation outputs

verbose = False

for s in sub_dirs:
    subid = os.path.splitext(os.path.basename(s))[0]
    
    t1_file = os.path.join(s, 'anat', 'T1.nii.gz') 
    out_dir = os.path.join(s, 'anat', 'T1.anat')
        
    if verbose:
         print('processing subject',subid)
            
    cmd_fsl_anat = ['fsl_anat --clobber -i ', t1_file]
    cmd_qsub = cmd_qsub_base + ['-name', f'{cohort}_fsl_anat_' + subid,'-command', cmd_fsl_anat]

    subprocess.Popen(' '.join(cmd_qsub), shell=True)
    
#%%  Apply ANTs registration transforms to FSL segmentation output

for sub_dir in sub_dirs[:]:
    subid = os.path.splitext(os.path.basename(sub_dir))[0]
    aff_transform_file = os.path.join(sub_dir,'T1_ants_'+str(template), 'T1_BrainNorm_0GenericAffine.mat')
    nlin_transform_file = os.path.join(sub_dir,'T1_ants_'+str(template), 'T1_BrainNorm_1Warp.nii.gz')

    #grey matter
    log_name = 'warp_fsl_gmseg_' + subid
    input_file = os.path.join(sub_dir,'anat','T1.anat','T1_fast_pve_1.nii.gz') #default fsl output name for segmented grey matter
    output_file = os.path.join(sub_dir,'T1_ants_'+str(template), 'T1_BrainNorm_fast_pve_1.nii.gz')
    
    ants_t_cmd = f"module load ANTs ; antsApplyTransforms -d 3 -i {input_file} -r {tpl_file} -t {nlin_transform_file} -t {aff_transform_file} -o {output_file}"
    cmd_ants = f'"{ants_t_cmd}"'
    cmd_qsub = cmd_qsub_base + ['-name', log_name , '-command',cmd_ants, '-clusterspec', 'slurm']
    
    subprocess.Popen(' '.join(cmd_qsub), shell = True)
    
    #white matter
    log_name = 'warp_fsl_wmseg_' + subid
    input_file = os.path.join(sub_dir,'anat','T1.anat','T1_fast_pve_2.nii.gz')#default fsl output name for segmented white matter
    output_file = os.path.join(sub_dir,'T1_ants_'+str(template), 'T1_BrainNorm_fast_pve_2.nii.gz')
    
    ants_t_cmd = f"module load ANTs ; antsApplyTransforms -d 3 -i {input_file} -r {tpl_file} -t {nlin_transform_file} -t {aff_transform_file} -o {output_file}"
    cmd_ants = f'"{ants_t_cmd}"'
    cmd_qsub = cmd_qsub_base + ['-name', log_name , '-command',cmd_ants, '-clusterspec', 'slurm']
    
    subprocess.Popen(' '.join(cmd_qsub), shell = True)
  
#%% Modulate registered grey and white matter probability maps by jacobian determinants to obtain GMV and WMV

for sub_dir in sub_dirs[:]:
    
        subid = os.path.splitext(os.path.basename(sub_dir))[0]
        log_jac_file = os.path.join(sub_dir,'T1_ants_MNI152NLin2009cSym', 'T1_BrainNorm_Jacobian.nii.gz')
        
        if os.path.exists(log_jac_file):
            #exponentiate the log jacobian (modulation requires jacobian)
            tmp_jac_file = os.path.join(sub_dir,'T1_ants_MNI152NLin2009cSym','T1_BrainNorm_Jacobian_exp.nii.gz')
            log_jac_img = nib.load(log_jac_file)
            jac_data = np.exp(log_jac_img.get_fdata())
            jac_img = nib.Nifti1Image(jac_data, affine=log_jac_img.affine, header=log_jac_img.header)
            nib.save(jac_img, tmp_jac_file)
        
            #do both grey and white matter modulation in the same job to avoid doubling the nibabel compute 
            log_name = 'modul_seg_' + subid
            
            #modulate the gmp by non-log jacobirn determinants
            input_file = os.path.join(sub_dir,'T1_ants_MNI152NLin2009cSym', 'T1_BrainNorm_fast_pve_1.nii.gz')
            output_file = os.path.join(sub_dir,'T1_ants_MNI152NLin2009cSym', 'T1_BrainNorm_GMV.nii.gz')
            ants_gm_cmd = f"module load ANTs ; ImageMath 3 {output_file} m {input_file} {tmp_jac_file} ;"
        
            #modulate the wmp by non-log jacobian determinants
            input_file = os.path.join(sub_dir,'T1_ants_MNI152NLin2009cSym', 'T1_BrainNorm_fast_pve_2.nii.gz')
            output_file = os.path.join(sub_dir,'T1_ants_MNI152NLin2009cSym', 'T1_BrainNorm_WMV.nii.gz')
            ants_wm_cmd = f"ImageMath 3 {output_file} m {input_file} {tmp_jac_file} ;"
            
            clean_cmd =  f"rm {tmp_jac_file}" #delete the temporary non-log jacobian determinants to save diskspace
            cmd_ants = f'"{ants_gm_cmd} {ants_wm_cmd } {clean_cmd}"'
            cmd_qsub = cmd_qsub_base + ['-name', log_name , '-command',cmd_ants, '-clusterspec', 'slurm']
        
            subprocess.Popen(' '.join(cmd_qsub), shell = True)

