.. _feature_extraction:

Deep Feature Extraction Module
===============================

The `feature_extraction` module within `neursen_reveal` is your gateway to leveraging state-of-the-art deep learning models for brain research. Instead of relying on hand-crafted, low-level stimulus features, this module allows you to extract rich, high-dimensional representations directly from your raw stimuli (e.g., audio files, images, videos).

This is crucial because deep neural networks, particularly those trained on large datasets, learn complex, hierarchical representations that often mirror the processing stages found in biological brains. By extracting features from specific layers of these AI models, researchers can test hypotheses about functional specialization in the brain, investigate the neural encoding of semantic meaning, or compare computational principles between artificial and biological systems. This module provides a streamlined way to generate these powerful features, which then serve as the predictive variables in subsequent brain encoding models.

Function Reference
-------------------

The primary function in this module is `extract_features`.

.. autofunction:: neursen_reveal.feature_extraction.extract_features


Example Usage
--------------

This example illustrates how to extract features using the `ecapa-tdnn` model for speech stimuli, saving them to an HDF5 file.

        from neursen_reveal.deep_feature_extraction import extract_features

        import os

        # Define placeholder paths for your data and output

        BIDS_BOLD_ROOT = "/path/to/your/bids_bold_data" # Directory containing sub-XX/func/*_events.tsv

        STIMULI_ROOT = "/path/to/your/stimuli_files"    # Directory containing your raw audio/image files

        OUTPUT_FEATURES_FILE = "/path/to/output/ai_features/speech_model_features.h5"

        # Ensure the output directory for features exists

        os.makedirs(os.path.dirname(OUTPUT_FEATURES_FILE), exist_ok=True)

        print("--- Extracting features from speech stimuli using ECAPA-TDNN ---")

        extract_features(

            bold_data_root_dir=BIDS_BOLD_ROOT,

            stimuli_base_dir=STIMULI_ROOT,

            output_file_path=OUTPUT_FEATURES_FILE,

            model_id="ecapa-tdnn",        # Specify the AI model (e.g., for audio)

            layer_name="final",           # Extract features from the final layer

            tr=2.0,                       # Your fMRI TR in seconds

            task_of_interest="task-storylistening" # Optional: Process only a specific task

        )

        print(f"Feature extraction complete. Features saved to: {OUTPUT_FEATURES_FILE}")

        # Example with a different model (e.g., ResNet50 for image stimuli) 

        print("\n--- Extracting features from image stimuli using a custom ResNet50 checkpoint ---")

        extract_features(

        bold_data_root_dir=BIDS_BOLD_ROOT, # Still need events.tsv

        stimuli_base_dir="/path/to/your/image_stimuli",

        output_file_path="/path/to/output/ai_features/image_model_features.h5",

        model_id="resnet50",

        layer_name="final", # A common layer for image features

        tr=1.5
        )
        
        print("Image feature extraction complete.")