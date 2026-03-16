# Folder contents

This folder contains all necessary scripts to estimate, evaluate, and transfer or extend whole-brain voxelwise normative models. As voxelwise data can be quite heavy to hold in memory for large datasets, the pipeline proceeds in batches of voxels (size adjustable based on available computational resources).

These scripts were used for normative models of log jacobian determinants (extracted directly with ANTs) and modulated gray and white matter (the ANTs registration transforms were applied to the fsl_anat pve-1 and pve-2 segmented files, and they were then multiplied by the jacobian determinants from ANTs). The entire preprocessing and modelling pipeline was run with the MNI152NLin2009cSym brain template.

### Scripts include :
* Preparing data : grabbing and cleaning demographic (chronological age, biological sex and site) data from several datasets, grabbing nifti data and extracting batches of voxel values, wrapping both into norm_data objects for every batch
* Running models : setting model parameters, splitting the norm_data into train and test, optionally pushing clinical subjects into the test set, and using the runner tool of the pcntoolkit to submit estimation jobs to a cluster
* Evaluating models : grabbing fit statistics from each batch, calculating kurtosis and skew, saving each metric into nifti brain maps, plotting centiles plots with pcntooltkit utils, optionally checking subjects with large amounts of outlier voxels
* Transferring or extending models : preparing the new data (as described above), running the models either in a cluster with the runner or on a local machine with adjustable new batch size, getting subject-level Zscore brain maps
