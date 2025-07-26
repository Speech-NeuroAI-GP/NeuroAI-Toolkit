import os
import numpy as np
import h5py # For checking feature file content
from typing import List, Optional, Union, Any

# Import functions from your existing scripts
from neursen_reveal.fmri_cleaning import run_cleaning_pipeline
from neursen_reveal.feature_extraction import extract_features # Assuming feature_extraction.py
from neursen_reveal.voxelwise_modeling import generate_scores_for_specific_model_across_subjects, render_averaged_scores, render_scores # Assuming voxelwise_modeling.py
from neursen_reveal.stats import get_significant_model_scores # Assuming stats.py
from neursen_reveal.parcellation import plot_parcellation_of_peak_intensity_regions


def run_full_voxelwise_pipeline(
    bids_data_root_dir: str,
    output_base_dir: str,
    voxelwise_model_bold_test_runs: list[str],
    precomputed_features_h5_path: Optional[str] = None,
    stimuli_base_dir: Optional[str] = None,
    ai_model_id: Optional[str] = None,
    ai_layer_name: str = 'final',
    ai_model_checkpoint: Optional[str] = None,
    tr: float = 2.0,
    voxelwise_model_delayers: List[int] = [1, 2, 3, 4],
    voxelwise_model_n_runs_out: int = 1,
    perform_cleaning: bool = False, # Flag to enable/disable cleaning
    selected_confounds: Optional[List[str]] = None,
    smoothing_kernel_size: Optional[int] = None,
    left_surface_template: str = None,
    right_surface_template: str = None,
    statistical_significance_testing: bool = True,
    significance_alpha: float = 0.05,
    plot_threshold: Union[float, str] = "90%",
    pipeline_title: str = "Voxel-wise Modeling Results"
) -> None:
    """
    Executes a comprehensive fMRI voxel-wise modeling pipeline.

    This function orchestrates a multi-stage process:
    1. **Optional Data Cleaning:** Applies confound regression and/or spatial smoothing
       to BOLD data using `run_cleaning_pipeline` if specified.
    2. **Feature Extraction (Conditional):** If a precomputed feature file is not provided,
       it extracts AI model features from stimuli using `extract_features`.
    3. **Voxel-wise Modeling:** Trains a `VoxelWiseModeler` and computes R-squared scores
       across subjects using `generate_scores_for_specific_model_across_subjects`.
    4. **Statistical Significance Testing (Conditional):** If enabled, it identifies
       statistically significant voxels using `get_significant_model_scores`.
    5. **Visualization:** Renders brain maps of the results onto an inflated surface,
       either showing significant regions parcellated by network or averaged scores,
       and also provides a general rendering of the scores.

    Parameters
    ----------
    bids_data_root_dir : str
        Path to the root directory of your BIDS-formatted fMRI dataset.
        This should contain `sub-*` folders. If cleaning is performed, this
        should be the fMRIPrep output directory.
    output_base_dir : str
        The root directory where all generated outputs (cleaned data,
        extracted features, voxel-wise model scores, and plots) will be saved.
        Structure: `output_base_dir/cleaned_data/`, `output_base_dir/features/`,
        `output_base_dir/voxelwise_results/<model_id>/`.
    voxelwise_model_bold_test_runs : list[str], optional
        A list of string identifiers (e.g., 'task-auditory') that identify the
        BOLD signals to be used for testing within the voxel-wise model.
    precomputed_features_h5_path : str, optional
        Path to an existing HDF5 (.h5) file containing precomputed AI features.
        If provided, `stimuli_base_dir`, `ai_model_id`, `ai_layer_name`,
        `ai_model_checkpoint`, and `tr` will be ignored for feature extraction.
        Defaults to None.
    stimuli_base_dir : str, optional
        Path to the root directory containing your raw stimulus files (e.g., 'data/stimuli/').
        Required if `precomputed_features_h5_path` is None and feature extraction is needed.
    ai_model_id : str, optional
        Identifier for the AI model to use for feature extraction (e.g., 'whisper', 'ecapa-tdnn', 'clip').
        Required if `precomputed_features_h5_path` is None.
    ai_layer_name : str, optional
        Specific layer from which to extract features. Defaults to 'final'.
        See `feature_extraction.extract_features` for supported layers per model.
    ai_model_checkpoint : str, optional
        Specific checkpoint/version of the AI model (e.g., 'openai/whisper-large').
        Required if the selected model has multiple checkpoints.
    tr : float, optional
        The TR (repetition time) duration of your fMRI data in seconds.
        Used for TR-alignment during feature extraction. Defaults to 2.0.
    voxelwise_model_delayers : list[int], optional
        List of time-step delays to use for the `VoxelWiseModeler`. Defaults to `[1, 2, 3, 4]`.
    voxelwise_model_n_runs_out : int, optional
        Number of runs to leave out for validation in each cross-validation fold
        during model fitting. Defaults to 1 (Leave-One-Run-Out).
    perform_cleaning : bool, optional
        If True, the data cleaning pipeline (`run_cleaning_pipeline`) will be executed
        before voxel-wise modeling. Defaults to False.
    selected_confounds : list[str], optional
        List of confound regressors to use for cleaning (e.g., ['trans_x', 'trans_y']).
        Only relevant if `perform_cleaning` is True. If None, no confound regression.
    smoothing_kernel_size : int, optional
        FWHM smoothing kernel size in millimeters. Only relevant if `perform_cleaning`
        is True. If None, no spatial smoothing.
    left_surface_template : str
        Absolute path to the left hemisphere GIFTI surface file (e.g., a FreeSurfer
        `fsaverage`/`conte69` surface, or HCP-Minn `fs_LR.32k` surface).
        Required for smoothing if `smoothing_kernel_size` is provided.
    right_surface_template : str
        Absolute path to the right hemisphere GIFTI surface file, similar to `left_surface_template`.
        Required for smoothing if `smoothing_kernel_size` is provided.
    statistical_significance_testing : bool, optional
        If True, `get_significant_model_scores` will be used to identify and plot
        only statistically significant voxels. If False, the averaged scores
        (`render_averaged_scores`) will be plotted. Defaults to True.
    significance_alpha : float, optional
        The significance level (alpha) for FDR correction during statistical
        testing. Only relevant if `statistical_significance_testing` is True.
        Defaults to 0.05.
    plot_threshold : Union[float, str], optional
        Threshold for displaying values on the brain surface maps. Can be a percentage
        string (e.g., "90%") or a float (e.g., 0.1). Defaults to "90%".
    pipeline_title : str, optional
        A general title for the entire pipeline's results, used in plots.
        Defaults to "Voxel-wise Modeling Results".

    Returns
    -------
    None
        The function saves results (cleaned data, features, scores) to disk
        and displays interactive brain plots.

    Raises
    ------
    ValueError
        If required parameters for feature extraction are missing when
        `precomputed_features_h5_path` is None.
        If `perform_cleaning` is True but `left_surface_template` or
        `right_surface_template` are not provided.
    FileNotFoundError
        If specified paths (e.g., `bids_data_root_dir`, `stimuli_base_dir`,
        surface templates) do not exist.
    """
    print(f"--- Starting Full Voxel-wise Modeling Pipeline: {pipeline_title} ---")

    # --- 1. Determine BOLD Data Source ---
    current_bold_data_dir = bids_data_root_dir
    cleaned_output_dir = os.path.join(output_base_dir, 'cleaned_bids_data')

    if perform_cleaning:
        if not (left_surface_template and right_surface_template):
            raise ValueError(
                "Surface templates (left_surface_template, right_surface_template) "
                "are required for cleaning with smoothing."
            )
        print(f"\n--- Step 1: Performing BOLD Data Cleaning ---")
        print(f"  Input BOLD data from: {bids_data_root_dir}")
        print(f"  Saving cleaned data to: {cleaned_output_dir}")
        print(f"  Confounds selected: {selected_confounds}")
        print(f"  Smoothing kernel: {smoothing_kernel_size} mm FWHM")

        os.makedirs(cleaned_output_dir, exist_ok=True)
        try:
            run_cleaning_pipeline(
                bids_fmriprep_root_dir=bids_data_root_dir,
                output_base_dir=cleaned_output_dir,
                left_surface_template=left_surface_template,
                right_surface_template=right_surface_template,
                selected_confounds=selected_confounds,
                smoothing_kernel_size=smoothing_kernel_size,
                # Consider adding target_subject/session/task_ids if needed for cleaning
            )
            current_bold_data_dir = cleaned_output_dir
            print(f"Cleaning complete. Proceeding with cleaned data from: {current_bold_data_dir}")
        except Exception as e:
            print(f"Error during cleaning pipeline: {e}")
            raise # Re-raise the exception to stop the pipeline
    else:
        print("\n--- Step 1: Skipping BOLD Data Cleaning (using original data) ---")
        print(f"  Using BOLD data from: {current_bold_data_dir}")

    # --- 2. Feature Extraction ---
    features_h5_path = precomputed_features_h5_path
    if features_h5_path:
        print(f"\n--- Step 2: Using Precomputed Features ---")
        if not os.path.exists(features_h5_path):
            raise FileNotFoundError(f"Precomputed features file not found: {features_h5_path}")
        print(f"  Loading features from: {features_h5_path}")
    else:
        print(f"\n--- Step 2: Extracting AI Features On-the-Fly ---")
        features_output_dir = os.path.join(output_base_dir, 'extracted_features')
        os.makedirs(features_output_dir, exist_ok=True)
        # Construct a meaningful filename for the extracted features
        model_name_for_file = ai_model_id.replace('/', '_').replace('-', '_') if ai_model_id else 'unknown_model'
        layer_name_for_file = ai_layer_name if ai_layer_name else 'final'
        features_h5_path = os.path.join(
            features_output_dir,
            f"{model_name_for_file}_{layer_name_for_file}_features.h5"
        )

        if not all([stimuli_base_dir, ai_model_id, tr]):
            raise ValueError(
                "For on-the-fly feature extraction, 'stimuli_base_dir', "
                "'ai_model_id', and 'tr' are required parameters."
            )
        print(f"  Extracting features with Model: {ai_model_id}, Layer: {ai_layer_name}")
        print(f"  Stimuli Base Directory: {stimuli_base_dir}")
        print(f"  Saving extracted features to: {features_h5_path}")

        try:
            extract_features(
                bold_data_root_dir=bids_data_root_dir, # Use original BIDS root to find events.tsv
                stimuli_base_dir=stimuli_base_dir,
                output_file_path=features_h5_path,
                model_id=ai_model_id,
                layer_name=ai_layer_name,
                checkpoint=ai_model_checkpoint,
                tr=tr,
                # task_of_interest=... # Add if you need to limit feature extraction to specific tasks
            )
            print(f"Feature extraction complete.")
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise

    # --- 3. Voxel-wise Modeling ---
    voxelwise_results_dir = os.path.join(output_base_dir, 'voxelwise_results')
    # If a model ID was used for feature extraction, use it for the results subdirectory
    model_results_specific_dir = os.path.join(voxelwise_results_dir, model_name_for_file if ai_model_id else 'unknown_model')
    os.makedirs(model_results_specific_dir, exist_ok=True)
    print(f"\n--- Step 3: Performing Voxel-wise Modeling ---")
    print(f"  BOLD data source: {current_bold_data_dir}")
    print(f"  Features source: {features_h5_path}")
    print(f"  Saving model scores to: {model_results_specific_dir}")

    try:
        generate_scores_for_specific_model_across_subjects(
            Bold_dataset_dir=current_bold_data_dir,
            features_file_path=features_h5_path,
            saving_dir=model_results_specific_dir,
            save_EV=False, # As per original request, EV is not explicitly handled for saving
            verbose=True,
            random_seed=None, # You can make this a pipeline parameter if desired
            delayers=voxelwise_model_delayers,
            bold_test=voxelwise_model_bold_test_runs,
            n_runs_out=voxelwise_model_n_runs_out
        )
        print("Voxel-wise modeling complete. Scores generated for each subject.")
    except Exception as e:
        print(f"Error during voxel-wise modeling: {e}")
        raise

    # --- 4. Statistical Significance Testing & Visualization ---
    print(f"\n--- Step 4: Visualizing Results ---")

    if statistical_significance_testing:
        print(f"  Performing statistical significance testing (alpha={significance_alpha})...")
        try:
            # get_significant_model_scores returns a single array of significant mean scores
            significant_scores_map = get_significant_model_scores(
                results_dir=model_results_specific_dir,
                alpha=significance_alpha
            )
            
            # Save the significant scores map
            significant_scores_path = os.path.join(model_results_specific_dir, f'significant_scores_alpha_{str(significance_alpha).replace(".", "_")}.npy')
            np.save(significant_scores_path, significant_scores_map)
            print(f"  Statistically significant scores saved to: {significant_scores_path}")

            print(f"  Plotting significant regions parcellated by network (threshold: {plot_threshold})...")
            parcellated_view = plot_parcellation_of_peak_intensity_regions(
                data=significant_scores_map,
                threshold=plot_threshold
            )
            print("  Interactive parcellation plot generated (display in notebook/browser).")
            # You might want to save this HTML view if not in a notebook
            # parcellated_view.save_as_html(os.path.join(model_results_specific_dir, 'significant_parcellated_regions.html'))


            print(f"  Rendering significant scores on surface (threshold: {plot_threshold})...")
            # render_scores expects a single scores array
            rendered_significant_view = render_scores(
                scores=significant_scores_map,
                threshold=plot_threshold,
                title=f"{pipeline_title} - Significant Scores (alpha={significance_alpha})"
            )
            print("  Interactive surface plot of significant scores generated (display in notebook/browser).")
            # rendered_significant_view.save_as_html(os.path.join(model_results_specific_dir, 'significant_scores_render.html'))

        except Exception as e:
            print(f"Error during statistical testing or plotting significant scores: {e}")
            # Decide whether to raise or just log and continue
            raise
    else:
        print("  Skipping statistical significance testing (plotting averaged scores)...")
        # render_averaged_scores function already handles loading, averaging, and rendering
        # It also implicitly saves an 'average_scores.npy' in the results_dir.
        try:
            print(f"  Rendering averaged scores on surface (threshold: {plot_threshold})...")
            averaged_rendered_view = render_averaged_scores(
                data_dir=model_results_specific_dir,
                threshold=plot_threshold,
                title=f"{pipeline_title} - Averaged R-squared Scores"
            )
            print("  Interactive surface plot of averaged scores generated (display in notebook/browser).")
            # averaged_rendered_view.save_as_html(os.path.join(model_results_specific_dir, 'averaged_scores_render.html'))

            # # Load the saved averaged scores to pass to parcellation plot
            average_scores_path = os.path.join(model_results_specific_dir, 'average_scores.npy')
            if os.path.exists(average_scores_path):
                averaged_scores_data = np.load(average_scores_path)
                print(f"  Plotting averaged regions parcellated by network (threshold: {plot_threshold})...")
                parcellated_averaged_view = plot_parcellation_of_peak_intensity_regions(
                    data=averaged_scores_data,
                    threshold=plot_threshold
                )
                print("  Interactive parcellation plot of averaged scores generated (display in notebook/browser).")
            #     # parcellated_averaged_view.save_as_html(os.path.join(model_results_specific_dir, 'averaged_parcellated_regions.html'))
            # else:
            #     print(f"Warning: 'average_scores.npy' not found at {average_scores_path}. Cannot plot parcellation of averaged scores.")

        except Exception as e:
            print(f"Error during rendering averaged scores or parcellation: {e}")
            # Decide whether to raise or just log and continue
            raise


    print(f"\n--- Full Voxel-wise Modeling Pipeline Finished for: {pipeline_title} ---")
    print(f"All results saved in: {output_base_dir}")