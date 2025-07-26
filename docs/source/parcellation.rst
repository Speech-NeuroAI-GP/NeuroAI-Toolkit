.. _parcellation:

Brain Parcellation and Plotting Module
========================================

After performing complex analyses like voxel-wise modeling or RSA, the results often come as high-dimensional brain maps (e.g., R-squared values across thousands of cortical vertices). Interpreting these maps and communicating findings effectively requires robust visualization tools. Moreover, understanding how these findings relate to known functional brain networks or anatomical regions is crucial for drawing neuroscientifically meaningful conclusions.

The `parcellation` module in `neursen_reveal` simplifies this critical step. It provides intuitive functions to render your brain maps onto inflated cortical surfaces, making it easy to identify spatial patterns of activity or model performance. Crucially, it integrates functional brain atlases, allowing you to automatically parcellate and visualize peak activity regions according to well-established functional networks (e.g., Yeo 17-network parcellation). This dual capability — clear visualization and network-level interpretation — transforms raw numerical output into compelling and insightful brain images, facilitating deeper understanding of brain-AI correspondences.

Function Reference
--------------------

This module offers functions to plot various brain maps on surfaces and to parcellate peak regions.

.. autofunction:: neursen_reveal.parcellation.plot_parcellation_of_peak_intensity_regions

Example Usage
---------------

These examples demonstrate how to use the parcellation features for your brain maps.

        ```

        from neursen_reveal.parcellation import plot_parcellation_of_peak_intensity_regions

        import numpy as np

        import os

        # Assume you have a pre-computed significant or averaged scores map

        # For demonstration, we'll use a placeholder path.

        # In a real scenario, this would be the output from your stats or voxelwise_modeling steps.

        SIGNIFICANT_SCORES_MAP_PATH = "/path/to/your/analysis_results/significant_encoding_map.npy"

        AVERAGED_SCORES_MAP_PATH = "/path/to/your/voxelwise_scores_output/averaged_scores_for_plotting.npy" # Example if you averaged manually


        # Load your computed scores

        significant_map = np.load(SIGNIFICANT_SCORES_MAP_PATH)

        averaged_map = np.load(AVERAGED_SCORES_MAP_PATH)


        print("--- Example 1: Plotting peak significant regions parcellated by network ---")

        # This function identifies peak regions and colors them by functional network

        parcellated_view = plot_parcellation_of_peak_intensity_regions(

            data=significant_map,

            threshold="90%" # Display the top 10% of significant values

        )
        # The `parcellated_view` object can be displayed directly in a Jupyter environment.

        # If running outside, you might save it: parcellated_view.save_as_html("parcellated_map.html")
        
        print("Interactive parcellation plot generated (display in notebook/browser).")


