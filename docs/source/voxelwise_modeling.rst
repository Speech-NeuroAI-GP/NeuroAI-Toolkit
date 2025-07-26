.. _voxelwise_modeling:

Voxel-wise Modeling Module
===========================

Voxel-wise encoding models represent a cornerstone of modern fMRI analysis, allowing researchers to quantify how well various stimulus features (e.g., from AI models) predict brain activity at the level of individual voxels or vertices across the cortex. This approach moves beyond traditional univariate analyses to build predictive models of brain responses, offering fine-grained insights into neural representations.

The `voxelwise_modeling` module within `neursen_reveal` provides the tools to construct and evaluate these encoding models. It aligns AI-derived features with BOLD fMRI signals, trains linear regression models for each brain location, and computes metrics like R-squared scores, which indicate the proportion of variance in brain activity explained by the AI model's features. This direct mapping helps answer questions about what information specific brain regions encode, and how closely artificial systems' representations align with biological ones. By providing flexible parameters like delayers and cross-validation strategies, this module enables rigorous and robust brain modeling.

Function Reference
-------------------

This module includes functions for generating model scores, rendering these scores on brain surfaces, averaging model scores across subjects then rendering the averaged score on brain surface.

.. autofunction:: neursen_reveal.voxelwise_modeling.generate_scores_for_specific_model_across_subjects
.. autofunction:: neursen_reveal.voxelwise_modeling.render_scores
.. autofunction:: neursen_reveal.voxelwise_modeling.render_averaged_scores


Example Usage
--------------

This example illustrates the two-step process: first, generating voxel-wise R-squared scores, and then visualizing the averaged results across subjects on a brain surface.

        ```
        
        from neursen_reveal.voxelwise_modeling import generate_scores_for_specific_model_across_subjects

        from neursen_reveal.voxelwise_modeling import render_averaged_scores

        import os

        import numpy as np # May be needed for advanced custom plotting after generation

        # --- Step 1: Generate Voxel-wise Scores ---

        # Define placeholder paths for your data and output

        CLEANED_BOLD_ROOT = "/path/to/your/cleaned_bids_data"

        FEATURES_FILE = "/path/to/your/extracted_features/my_ai_model_features.h5"

        SCORES_OUTPUT_DIR = "/path/to/your/voxelwise_scores_output"

        # Ensure the output directory for scores exists

        os.makedirs(SCORES_OUTPUT_DIR, exist_ok=True)

        print("--- Generating voxel-wise R-squared scores across subjects ---")

        generate_scores_for_specific_model_across_subjects(

            Bold_dataset_dir=CLEANED_BOLD_ROOT,

            features_file_path=FEATURES_FILE,

            saving_dir=SCORES_OUTPUT_DIR,

            bold_test=['task-testrun'], # Specify the functional run(s) to use as test set

            delayers=[1, 2, 3, 4],       # Common hemodynamic delays in TRs

            n_runs_out=1,                # Leave one run out for cross-validation

            verbose=True,

            random_seed=42               # For reproducibility

        )

        print(f"Voxel-wise scores generated for each subject and saved to: {SCORES_OUTPUT_DIR}")

        # --- Step 2: Render Averaged Scores on Brain Surface ---

        print("\n--- Rendering averaged R-squared scores on brain surface ---")

        # `render_averaged_scores` expects the directory containing the subject-level scores.npy files.

        render_averaged_scores(

            data_dir=SCORES_OUTPUT_DIR,

            threshold="95%", # Show only the top 5% of scores

            title="Averaged Encoding Performance (AI Model X)"

        )
        
        print("Interactive brain plot of averaged scores generated (will open in browser/notebook).")
