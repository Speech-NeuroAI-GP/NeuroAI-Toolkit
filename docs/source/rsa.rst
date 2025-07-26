.. _rsa:

Representational Similarity Analysis (RSA) Module
==================================================

Representational Similarity Analysis (RSA) is a powerful, flexible framework for comparing information representations across different modalities, such as between artificial intelligence models and the human brain. Instead of trying to directly map features to brain activity (as in encoding models), RSA asks: "Do two systems represent information in a similar way, i.e., are their 'neural codes' or 'feature spaces' structurally alike?"

The `rsa` module within `neursen_reveal` enables you to perform this high-level comparison. It involves:
1.  **Computing Representational Dissimilarity Matrices (RDMs):** These matrices capture the pairwise "distances" or dissimilarities between representations of different stimuli within a given system (e.g., how different two spoken words are in an AI model's internal representation, or how different their corresponding brain activation patterns are).
2.  **Comparing RDMs:** The core of RSA is then comparing these RDMs (e.g., using correlation) to quantify the similarity of the representational geometries. A high correlation suggests that the two systems encode information in a structurally similar manner.

This module provides an integrated pipeline for RSA, allowing researchers to gain insights into the correspondence between artificial neural networks and specific brain regions or networks, contributing to our understanding of the computational principles underlying cognition.

Function Reference
-------------------

The primary function for conducting RSA is `run_rsa_pipeline`.

.. autofunction:: neursen_reveal.rsa.run_rsa_pipeline

Example Usage
--------------

This example demonstrates a full RSA pipeline, including on-the-fly feature extraction, for a specific functional run and brain network.

        ```
        
        from neursen_reveal.rsa import run_rsa_pipeline

        import os

        # Define placeholder paths for your data

        CLEANED_BOLD_ROOT = "/path/to/your/cleaned_bids_data"

        RAW_STIMULI_ROOT = "/path/to/your/raw_stimuli_files" # Only needed if extract_features_on_the_fly=True

        print("--- Running Representational Similarity Analysis (RSA) Pipeline ---")

        # Example: Compute RSA for 'task-storyX' in the 'auditory' network

        features_rdm, bold_rdm, rsa_score = run_rsa_pipeline(

            bold_data_root_dir=CLEANED_BOLD_ROOT,

            functional_run_name='task-storyX',       # The specific functional run to analyze (e.g., from BIDS file name)

            network_of_interest='auditory',          # Focus RSA on the auditory functional network

            h_delay_trs=2,                           # Hemodynamic delay of 2 TRs

            perform_statistical_significance_testing=True, # Perform statistical test

            extract_features_on_the_fly=True,        # Extract AI features during the RSA run

            stimuli_base_path=RAW_STIMULI_ROOT,      # Path to your raw stimulus files

            ai_model_id='ecapa-tdnn',                # Specify the AI model for feature extraction

            ai_layer_name='final',                   # Layer from which to extract features

            tr=2.0,                                  # fMRI TR in seconds

            rsa_metric='cosine',                     # Use cosine dissimilarity for RDMs

            plot_rdms=True                           # Generate and display RDM plots
        )

        print(f"\nRSA Results for functional run 'task-storyX' in 'auditory' network:")

        print(f"Shape of Features RDM: {features_rdm.shape}")

        print(f"Shape of BOLD RDM: {bold_rdm.shape}")

        print(f"Overall RSA Score (correlation between RDMs): {rsa_score:.4f}")

        # The RDM plots will be displayed if plot_rdms is True.
        
        # You can further analyze or save the `features_rdm`, `bold_rdm`, and `rsa_score`.