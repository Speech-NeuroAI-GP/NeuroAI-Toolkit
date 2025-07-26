.. _cleaning:

fMRI Data Cleaning Module
==========================

In the realm of fMRI research, raw BOLD signals are often contaminated by various sources of noise, including physiological artifacts (e.g., heart rate, respiration), head motion, and scanner drifts. These confounds can obscure true neuronal signals and lead to spurious findings, making robust preprocessing a critical first step for any reliable brain analysis.

The `fmri_cleaning` module addresses this challenge by providing a comprehensive and user-friendly pipeline for fMRI data cleaning. It's designed to minimize the impact of non-neuronal fluctuations, ensuring that subsequent analyses, such as voxel-wise modeling or RSA, are performed on data with enhanced signal-to-noise ratio. This module simplifies complex cleaning procedures, allowing researchers to focus on the scientific questions rather than the intricacies of preprocessing. It currently supports confound regression and surface-based spatial smoothing, delivering clean, standardized data ready for advanced neural decoding.

Function Reference
-------------------

The primary function in this module is `run_cleaning_pipeline`.

.. autofunction:: neursen_reveal.fmri_cleaning.run_cleaning_pipeline

Example Usage
--------------

Here are common ways to use the `run_cleaning_pipeline` function:

        ```
        from neursen_reveal.fmri_cleaning import run_cleaning_pipeline

        import os

        # Define placeholder paths for your data

        BIDS_FMRIPREP_ROOT = "/path/to/your/fmriprep_data"

        CLEANED_OUTPUT_DIR = "/path/to/your/cleaned_data_output"

        LEFT_SURFACE_TEMPLATE = "/path/to/your/surface_templates/tpl-fsLR_hemi-L_den-32k_sphere.surf.gii"

        RIGHT_SURFACE_TEMPLATE = "/path/to/your/surface_templates/tpl-fsLR_hemi-R_den-32k_sphere.surf.gii"

        # Ensure the output directory exists

        os.makedirs(CLEANED_OUTPUT_DIR, exist_ok=True)

        print("--- Example 1: Cleaning all BOLD data with common confound regressors and smoothing ---")

        run_cleaning_pipeline(

            bids_fmriprep_root_dir=BIDS_FMRIPREP_ROOT,

            output_base_dir=CLEANED_OUTPUT_DIR,

            left_surface_template=LEFT_SURFACE_TEMPLATE,

            right_surface_template=RIGHT_SURFACE_TEMPLATE,

            selected_confounds=[

                'trans_x', 'trans_y', 'trans_z', # Translation motion parameters

                'rot_x', 'rot_y', 'rot_z',       # Rotation motion parameters

                'a_comp_cor_00', 'a_comp_cor_01', # ACompCor components (adjust count as needed)

                'global_signal',                 # Global signal (use with caution)

                'white_matter', 'csf'            # Signal from white matter and CSF masks
            ],

            smoothing_kernel_size=2 # Apply 2mm FWHM surface smoothing
        )

        print("Cleaning for all specified data complete.")

        print("\n--- Example 2: Cleaning data for a specific subject, session, and task without smoothing ---")

        run_cleaning_pipeline(

            bids_fmriprep_root_dir=BIDS_FMRIPREP_ROOT,

            output_base_dir=CLEANED_OUTPUT_DIR,

            left_surface_template=LEFT_SURFACE_TEMPLATE, # Still needed even if smoothing_kernel_size is None

            right_surface_template=RIGHT_SURFACE_TEMPLATE, # Still needed even if smoothing_kernel_size is None

            selected_confounds=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],

            smoothing_kernel_size=None, # No smoothing

            target_subject_ids=['sub-01'],

            target_session_ids=['ses-01'],

            target_task_ids=['task-myexperiment'],
        )
        
        print("Cleaning for specified subject/session/task complete.")