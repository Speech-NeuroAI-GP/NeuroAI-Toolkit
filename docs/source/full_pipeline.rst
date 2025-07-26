.. _full_pipeline:

Full Unified Voxel-wise Modeling Pipeline
==========================================

Navigating the complexities of fMRI data preprocessing, AI feature extraction, brain modeling, statistical validation, and visualization can be daunting, especially when trying to ensure consistency and reproducibility across all stages. The `full_pipeline` module within `neursen_reveal` offers a game-changing solution: a single, unified function that orchestrates the entire voxel-wise encoding analysis from start to finish.

This integrated pipeline is invaluable for:
* **Streamlined Workflow:** Execute all necessary steps (cleaning, feature extraction, modeling, stats, plotting) with a single command, significantly reducing manual effort and potential errors.
* **Consistency:** Ensures that all modules work together seamlessly, maintaining consistent data handling and parameters throughout the analysis.
* **Rapid Prototyping and Exploration:** Quickly test different AI models, cleaning parameters, or statistical thresholds across your dataset.
* **Reproducibility:** By encapsulating the entire workflow, it inherently promotes reproducibility of your analyses.

Whether you're new to neursen_reveal or an experienced researcher looking to accelerate your workflow, this comprehensive pipeline provides a robust and efficient way to uncover the relationship between AI representations and brain activity, from raw data to interpretable brain maps.

Function Reference
-------------------

The comprehensive function for running the full pipeline is `run_full_voxelwise_pipeline`.

.. autofunction:: neursen_reveal.full_pipeline.run_full_voxelwise_pipeline

Example Usage
--------------

This example demonstrates how to run the entire end-to-end voxel-wise modeling pipeline.

```
        from neursen_reveal.full_pipeline import run_full_voxelwise_pipeline

        import os

        # Define placeholder paths for your input data and where results will be saved

        RAW_BIDS_FMRIPREP_ROOT = "/path/to/your/fmriprep_output_dir"

        FULL_PIPELINE_OUTPUT_DIR = "/path/to/your/analysis_results/full_pipeline_output"

        STIMULI_ROOT = "/path/to/your/raw_stimuli_files"

        LEFT_SURFACE_TEMPLATE = "/path/to/your/surface_templates/tpl-fsLR_hemi-L_den-32k_sphere.surf.gii"

        RIGHT_SURFACE_TEMPLATE = "/path/to/your/surface_templates/tpl-fsLR_hemi-R_den-32k_sphere.surf.gii"

        # Ensure the main output directory exists

        os.makedirs(FULL_PIPELINE_OUTPUT_DIR, exist_ok=True)

        print("--- Running the Full Unified Voxel-wise Modeling Pipeline ---")

        run_full_voxelwise_pipeline(

            bids_data_root_dir=RAW_BIDS_FMRIPREP_ROOT,

            output_base_dir=FULL_PIPELINE_OUTPUT_DIR,

            voxelwise_model_bold_test_runs=['task-runX'], # Specify run(s) for test set

            stimuli_base_dir=STIMULI_ROOT,

            ai_model_id='ecapa-tdnn', # AI model for feature extraction

            ai_layer_name='final',    # Layer of the AI model

            tr=2.0,                   # fMRI TR in seconds

            voxelwise_model_delayers=[1, 2, 3, 4], # Hemodynamic delays

            voxelwise_model_n_runs_out=1,
            
            perform_cleaning=True, # Enable cleaning step

            selected_confounds=[

                'trans_x', 'trans_y', 'trans_z',

                'rot_x', 'rot_y', 'rot_z',

                'a_comp_cor_00', 'a_comp_cor_01'
            ],

            smoothing_kernel_size=2, # Apply 2mm surface smoothing

            left_surface_template=LEFT_SURFACE_TEMPLATE,

            right_surface_template=RIGHT_SURFACE_TEMPLATE,
            
            statistical_significance_testing=True, # Perform statistical correction

            significance_alpha=0.05,
            
            plot_threshold="90%", # Plotting threshold for final maps

            pipeline_title="Comprehensive Encoding Analysis with ECAPA-TDNN",

            # reduce_features_to_dim=128 # Optional: Reduce feature dimensionality (e.g., for performance)

        )
        print("Full pipeline execution completed successfully. Check output directory for results.")

        # If you had precomputed features and wanted to skip the feature extraction step:

        # precomputed_features_path = "/path/to/your/precomputed_features.h5"

        # run_full_voxelwise_pipeline(

        #     bids_data_root_dir=RAW_BIDS_FMRIPREP_ROOT,

        #     output_base_dir=FULL_PIPELINE_OUTPUT_DIR,

        #     voxelwise_model_bold_test_runs=['task-runY'],

        #     precomputed_features_h5_path=precomputed_features_path, # Provide precomputed features

        #     tr=2.0, # TR is still needed for BOLD alignment if not done in cleaning.

        #     # ... other parameters remain the same ...
        
        # )