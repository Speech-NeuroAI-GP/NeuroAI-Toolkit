Core Modules Overview
=====================

`neursen_reveal` is structured into several key modules, each designed to handle a specific stage of the neuroimaging analysis pipeline. Below is a brief overview of each:

.. toctree::
   :maxdepth: 1

   cleaning
   feature_extraction
   voxelwise_modeling
   stats
   parcellation
   rsa
   full_pipeline
:py:mod:`neursen_reveal.fmri_cleaning`
----------------------------------------
This module provides functionality to preprocess fMRI BOLD data, including confound regression and surface-based spatial smoothing. Essential for preparing your data for robust modeling.

:py:mod:`neursen_reveal.feature_extraction`
---------------------------------------------------------
Enables the extraction of rich, high-dimensional features from various stimulus types (e.g., audio, images) using state-of-the-art deep learning models. These features serve as the input for brain modeling.

:py:mod:`neursen_reveal.voxelwise_modeling`
--------------------------------------------
The core of brain encoding. This module allows you to build and train voxel-wise encoding models, mapping AI-derived features to observed brain activity. It computes R-squared scores, indicating how well the model predicts neural responses.

:py:mod:`neursen_reveal.stats`
--------------------------------
Provides tools for statistical analysis of model results, including identifying statistically significant voxels after correcting for multiple comparisons (e.g., using FDR correction).

:py:mod:`neursen_reveal.parcellation`
-----------------------------------------------------
Facilitates the visualization of brain maps. It can render scores onto inflated brain surfaces and parcellate significant regions according to functional brain atlases, offering intuitive insights into brain-AI correspondences.

:py:mod:`neursen_reveal.full_pipeline`
---------------------------------------
Execute the entire fMRI voxel-wise modeling workflow from data cleaning to visualization with a single, configurable function.

:py:mod:`neursen_reveal.rsa`
-----------------------------
Implements Representational Similarity Analysis (RSA), a powerful technique to compare the representational spaces of AI models and brain regions. It computes and visualizes dissimilarity matrices for both.


