# Folder contents
This folder contains the scripts necessary to extract normative deviation scores (Z-scores) from a longitudinal Spinocerebellar Ataxia type 1 (SCA1) clinical dataset using pre-existing normative models. It also contains a script for a clinical score (SARA scale) out-of-sample prediction using both Z-scores and non-normative brain measures. 

This analysis was conducted both in a voxelwise fashion, using wholebrain jacobian determinants from a nonlinear image registration obtained with FSL, and in a parcellated fashion, using Freesurfer parcellated output (cortical thickness and subcortical volumes from the Destrieux atlas).

Parcellated Z-scores were extracted from parcellated Freesurfer output using the normative model from Rutherford et al. (https://doi.org/10.7554/eLife.72904), and wholebrain Z-scores were extracted voxelwise from jacobian determinants using a normative model from Fraza et al. (example here: https://github.com/CharFraza/CNV_normative_modeling/), adapted from the one in Holz et al. (https://doi.org/10.1038/s41593-023-01410-8). All models were estimated using the PCNtoolkit (https://github.com/amarquand/PCNtoolkit/).

### Scripts include :
* Preparation of wholebrain data for normative transfer of trained models (from preprocessed nifti to the required .pkl format)
  
(Parcellated freesurfer output does not require any other preparation than using the aparcstats2table and asegstats2table functions from Freesurfer to put together measures from all subjects).

* Transfer of pre-trained normative models for both voxelwise and parcellated data to extract cross-sectional Z-scores

* Transfer of pre-trained normative models for both voxelwise and parcellated data to extract longitudinal Z-diff scores (see Bučková et al. for the Z-diff method: https://doi.org/10.7554/eLife.95823.2)

* Extraction of longitudinal slopes for both wholebrain jacobian determinants, and parcellated thickness and subcortical volumes measures

* Kernel Ridge regression to predict cross-sectional SARA score and SARA score longitudinal change, exploring the neuroimaging featuresets separately (normative/non-normative, cross-sectional/longitudinal, wholebrain/parcellated)
