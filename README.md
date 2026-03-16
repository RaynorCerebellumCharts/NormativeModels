# NormativeModels
This repository includes all code used for the estimation of voxelwise normative models of volumetry across the whole brain, and to apply these estimated models to samples of interest (subject born preterm and patients with spinocerebellar ataxia). The corresponding manuscript is available at (preprint url).
 
All voxelwise models herein have been estimated using masks of the MNI152NLin2009cSym templates from templateflow resampled to 2mm resolution - the masks required to run the scripts are included in the /template_data folder. Three volume measures were used : log of jacobian determinants (log(JD)), modulated grey matter volume (GMV), and modulated white matter volume (WMV).

## General workflow for model estimation
* Preprocess MRI data:

This is done with the ANTs (Advanced Normalization Tools) toolbox and, additionally for GMV and WMV, with fsl. Inputs are raw T1 scans, outputs are preprocessed T1_BrainNorm_Jacobian.nii, T1_BrainNorm_GMV.nii, and T1_Brain_Norm_WMV.nii files.
  
* Prepare normative model estimation:

Extract voxel values from the preprocessed MRI data files and, together with demographic data of the sample (age as covariate, sex and site as batch effects), build norm_data objects required by the pcntoolkit for normative modeling operations.
This was done separately for batches of 150 voxels to keep reasonable variable size and estimation runtime.
  
* Run model estimation:

Using the norm_data objects, a normative model is fitted for each voxel separately. Output metrics (z-scores, logp, evaluation statistics) are all stored within the model folders and can be accessed directly or loaded back in the norm_data objects. A general rule of thumb with the pcntoolkit is to leave about 10 times more storage space available in the model folders compared to the initial norm_data file size - voxelwise models specifically are memory-intensive. 
  
* Evaluate fit of models:

Voxelwise evaluation metrics such as kurtosis, skew and explained variance are extracted and plotted onto brain images to examine how well the models fit the data.


## General workflow for model transfer
A template transfer notebook is included in this repository for other users to freely download and apply the estimated voxelwise normative models to their own data. Please also see this paper (pu25 url) for a practical overview of normative modeling analysis pipelines, as implemented in the pcntoolkit v1.0 and above.

* Preprocess MRI data: (use 01 script)

Done as explained above. To be fully compatible with estimated models, the same preprocessing pipeline should be used.

* Transfer models: (use transfer notebook)

Download models from public repository, build norm_data objects, run the transfer either in a cluster (with the runner utility of the pcntoolkit) or on a local machine, and outputs subject-level Z-score brain maps. Can adjust voxel batch size.

* Examine normative deviations in samples of interest:

This can be done using a variety of group-level  or individual-level analyses. Some examples can be found in the 05 script (group-dependent local and whole-brain burden of extreme deviations, TFCE test of mean group differences, etc.) and 06 script (patient-specific voxelwise deviation map, cross-validated regression analyses of symptom score), but this is of course entirely dependent on the research question.

