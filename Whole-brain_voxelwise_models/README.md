# Folder contents

This folder contains all necessary scripts to estimate, evaluate, and transfer or extend whole-brain voxelwise normative models (log jacobians, or modulated gray and white matter volume). As voxelwise data can be quite heavy to hold in memory for large datasets, the pipeline proceeds in batches of voxels (size adjustable based on available ocmputational resources).

### Scripts include :
* Preparing data : grabbing and cleaning demographic (age ,sex and site) data from several datasets, grabbing nifti data and extracting batches of voxel values, wrapping both into norm_data objects for every batch
* Running the models : splitting the norm_Data into train and test, optionally pushing clinical subjects into the test set, and using with the runner tool of the pcntoolkit to submit estimation jobs to a cluster
* Evaluate the models : grabbing fit statistics from each batch, calculating kurtosis and skew, saving each metric nifti brain maps, plotting centiles plots with pcntooltkit utils, optionaly checking subjects with large amounts of outlier voxels
* Transfer or extend the models :
