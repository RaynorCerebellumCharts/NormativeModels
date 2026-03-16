# NormativeModels
This repository includes all scripts used for the estimation of voxelwise normative models of volumetry across the whole brain, as well as those used to apply these estimated models to samples of interest (subject born preterm and patients with spinocerebellar ataxia). The corresponding publication is available at (preprint url).
 
Importantly, a template transfer notebook is also included for other users to freely download and apply the estimated voxelwise normative models to their own data. Please also see this paper (pu25 url) for a general and practical overview of normative modeling analysis pipelines, as implemented in the pcntoolkit v1.0 and above.

All voxelwise models herein have been estimated using masks of the MNI152NLin2009cSym templates from templateflow - the required masks are included here in the /template_data folder. 


This folder
* Transferring or extending models : preparing the new data (as described above), running the models either in a cluster with the runner or on a local machine with adjustable new batch size, getting subject-level Zscore brain maps
