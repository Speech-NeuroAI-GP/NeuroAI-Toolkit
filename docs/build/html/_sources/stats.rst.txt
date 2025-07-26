.. _stats:

Statistical Significance Module
==================================

In neuroimaging, obtaining brain maps (e.g., R-squared values from encoding models) is only the first step. Critically, we need to determine which of these findings are statistically robust and unlikely to have occurred by chance. The brain is vast, and without proper statistical correction for multiple comparisons, the risk of false positives is extremely high.

The `stats` module in `neursen-reveal` provides essential tools to rigorously evaluate the statistical significance of your voxel-wise model scores. It implements methods like one-sample t-tests against zero to assess whether scores are reliably greater than chance across a group of subjects, and importantly, applies False Discovery Rate (FDR) correction to control for the inflated false positive rate inherent in analyzing thousands of brain locations simultaneously. By using this module, researchers can confidently identify and report brain regions that show genuine, statistically significant encoding performance, enhancing the reliability and interpretability of their findings.

Function Reference
--------------------

The primary function in this module is `get_significant_model_scores`.

.. autofunction:: neursen_reveal.stats.get_significant_model_scores

Example Usage
--------------

This example demonstrates how to compute statistically significant brain scores from your subject-level results and save the resulting map.

        
        from neursen_reveal.stats import get_significant_model_scores

        import numpy as np

        import os

        # Define the directory where your subject-specific R-squared scores are saved
        
        SCORES_INPUT_DIR = "/path/to/your/voxelwise_scores_output" # From voxelwise_modeling module

        # Define the output path for the statistically significant scores map

        SIGNIFICANT_SCORES_PATH = "/path/to/your/analysis_results/significant_encoding_map.npy"

        # Ensure the output directory exists

        os.makedirs(os.path.dirname(SIGNIFICANT_SCORES_PATH), exist_ok=True)

        print("--- Identifying statistically significant model scores across subjects ---")

        # Alpha set to 0.05 (default) for 5% False Discovery Rate

        significant_scores_map = get_significant_model_scores(

            results_dir=SCORES_INPUT_DIR,

            alpha=0.05

        )

        print(f"Statistically significant scores identified. Non-significant voxels are masked.")

        # Save the significant scores map for later visualization or further analysis

        np.save(SIGNIFICANT_SCORES_PATH, significant_scores_map)

        print(f"Statistically significant scores map saved to: {SIGNIFICANT_SCORES_PATH}")

        # You can now use `significant_scores_map` with the parcellation module

        # to visualize only the regions that show reliable encoding.

        from neursen_reveal.parcellation import plot_parcellation_of_peak_intensity_regions
        
        plot_parcellation_of_peak_intensity_regions(data=significant_scores_map, threshold=0.01")