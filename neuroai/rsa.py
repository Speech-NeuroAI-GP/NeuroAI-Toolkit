import numpy as np
from scipy.spatial.distance import cdist
import dcor
import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import os
import glob
from neuroai.utils import get_stimuli_features_per_bold_signal
from neuroai.parcellation import parcellate_brain_by_network
from neuroai.stats import get_significant_voxels_from_arrays
from neuroai.feature_extraction import extract_features 


def compute_distance_correlation(distance_matrix1: np.ndarray, distance_matrix2: np.ndarray) -> float:
    """
    Computes the distance correlation between two distance matrices.

    Parameters
    ----------
    distance_matrix1 : np.ndarray
        The first square distance matrix (e.g., representing feature similarities).
        Shape should be (N, N).
    distance_matrix2 : np.ndarray
        The second square distance matrix (e.g., representing BOLD signal similarities).
        Shape should be (N, N) and must match the shape of distance_matrix1.

    Returns
    -------
    float
        The computed distance correlation between the two matrices.
        A value between 0 and 1, where 1 indicates perfect dependence.
    """
    # Ensure matrices are square and of the same shape
    if distance_matrix1.shape != distance_matrix2.shape or distance_matrix1.shape[0] != distance_matrix1.shape[1]:
        raise ValueError("Both distance_matrix1 and distance_matrix2 must be square and of the same dimensions.")

    # Extract upper triangle indices (k=1 excludes the diagonal)
    # This avoids redundant comparisons and self-similarity (which is 0 in distance matrices)
    triu_indices = np.triu_indices_from(distance_matrix1, k=1)
    flat_matrix1 = distance_matrix1[triu_indices]
    flat_matrix2 = distance_matrix2[triu_indices]

    # Compute distance correlation
    dist_corr = dcor.distance_correlation(flat_matrix1, flat_matrix2)
    return dist_corr

def plot_distance_matrices(
    distance_matrix_features: np.ndarray,
    distance_matrix_bold: np.ndarray,
    plot_title_features: str = "Feature Space Representational Dissimilarity Matrix",
    plot_title_bold: str = "BOLD Signal Representational Dissimilarity Matrix"
):
    """
    Plots the distance matrices for features and bold data.
    
    Parameters
    ----------
    distance_matrix_features : np.ndarray
        The distance matrix derived from features (e.g., AI model activations).
    
    distance_matrix_bold : np.ndarray
        The distance matrix derived from BOLD signals.
    
    plot_title_features : str, optional
        Title for the plot of the feature distance matrix.
        Defaults to "Feature Space Representational Dissimilarity Matrix".
    plot_title_bold : str, optional
        Title for the plot of the BOLD signal distance matrix.
        Defaults to "BOLD Signal Representational Dissimilarity Matrix".
    
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    img1= axes[0].imshow(distance_matrix_features, cmap='viridis', interpolation='nearest')
    cbar1 = fig.colorbar(img1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Normalized Cosine Distance')
    axes[0].set_title(plot_title_features)
    axes[0].set_xlabel('Stimulus/TR Index') 
    axes[0].set_ylabel('Stimulus/TR Index')
    # axes[0].axis('off')

    img2= axes[1].imshow(distance_matrix_bold, cmap='viridis', interpolation='nearest')
    cbar2 = fig.colorbar(img2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Cosine Dissimilarity')
    axes[1].set_title(plot_title_bold)
    axes[1].set_xlabel('Stimulus/TR Index')
    axes[1].set_ylabel('Stimulus/TR Index')
    # axes[1].axis('off')
    plt.tight_layout() 
    plt.show()
    


def compute_rsa(
    features: np.ndarray,
    bold_signals: np.ndarray,
    metric: str = 'cosine',
    plot_rdms: bool = True,
    plot_title_features: str = "Feature Space Representational Dissimilarity Matrix",
    plot_title_bold: str = "BOLD Signal Representational Dissimilarity Matrix"
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Representational Similarity Analysis (RSA) between AI model features
    and fMRI BOLD signals.

    This function quantifies the similarity between the representational geometries
    of an AI model's activations and brain activity (BOLD signals) in response
    to a set of stimuli. It involves computing two Representational Dissimilarity
    Matrices (RDMs) – one for features and one for BOLD signals – and then
    calculating the distance correlation between these RDMs.

    Parameters
    ----------
    features : np.ndarray
        A 2D array where each row represents the feature vector for a specific
        TR (time repetition) and columns represent different features
        (e.g., activations from an AI model layer).
        Shape: (num_TRs, num_features).
    bold_signals : np.ndarray
        A 2D array where each row represents the BOLD signal activity for a
        specific TR across a set of voxels or ROIs.
        Shape: (num_TRs, num_voxels_or_ROIs).
        The number of rows (TRs) must match `features`.
    metric : str, optional
        The distance metric to use when computing the dissimilarity matrices.
        Defaults to 'cosine' (cosine distance). Other valid metrics
        can be found in scipy.spatial.distance.cdist documentation.
    plot_rdms : bool, optional
        If True, the computed Representational Dissimilarity Matrices (RDMs)
        will be plotted. Defaults to True.
    plot_title_features : str, optional
        Title for the plot of the feature RDM. Only relevant if `plot_rdms` is True.
        Defaults to "Feature Space Representational Dissimilarity Matrix".
    plot_title_bold : str, optional
        Title for the plot of the BOLD signal RDM. Only relevant if `plot_rdms` is True.
        Defaults to "BOLD Signal Representational Dissimilarity Matrix".

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        A tuple containing:
        - features_rdm : np.ndarray
            The normalized Representational Dissimilarity Matrix (RDM) for the features.
        - bold_rdm : np.ndarray
            The normalized Representational Dissimilarity Matrix (RDM) for the BOLD signals.
        - dist_corr : float
            The distance correlation coefficient between the two RDMs.
    """
    if features.shape[0] != bold_signals.shape[0]:
        raise ValueError(
            f"Number of stimuli/TRs in features ({features.shape[0]}) "
            f"and bold_signals ({bold_signals.shape[0]}) must match."
        )

    # Compute Representational Dissimilarity Matrices (RDMs) using the specified metric
    features_rdm = cdist(features, features, metric=metric)
    bold_rdm = cdist(bold_signals, bold_signals, metric=metric)

    # Normalize each RDM to a [0, 1] range for consistent comparison and plotting
    # Add a small epsilon to avoid division by zero if all distances are zero (e.g., single stimulus)
    epsilon = np.finfo(float).eps
    features_rdm = features_rdm / (features_rdm.max() + epsilon)
    bold_rdm = bold_rdm / (bold_rdm.max() + epsilon)

    # Compute the distance correlation between the two normalized RDMs
    dist_corr = compute_distance_correlation(features_rdm, bold_rdm)
    print(f"Computed Distance Correlation (RSA score): {dist_corr:.4f}") # Formatted output

    # Plot the RDMs if requested
    if plot_rdms:
        plot_distance_matrices(
            features_rdm,
            bold_rdm,
            plot_title_features=plot_title_features,
            plot_title_bold=plot_title_bold
        )

    return features_rdm, bold_rdm, dist_corr

def run_rsa_pipeline(
    bold_data_root_dir: str,
    functional_run_name: str, # e.g., 'inamoment' or 'coco_run-01'
    precomputed_features_h5_path: str = None, # Path to the .h5 feature file
    network_of_interest: str = None,
    h_delay_trs: int = 2,
    padding_trs: int = 10,
    desired_subjects: list = None, # List of subject IDs, e.g., ['sub-UTS01', 'sub-UTS02']
    perform_statistical_significance_testing: bool = True,
    extract_features_on_the_fly: bool = False, # If True, will call extract_features
    stimuli_base_path: str = None, # Required if extract_features_on_the_fly is True
    tr: float= 2.0, # Required if extract_features_on_the_fly is True
    ai_model_id: str = None, # Required if extract_features_on_the_fly is True
    ai_layer_name: str = 'final', 
    ai_model_checkpoint: str = None, 
    rsa_metric: str = 'cosine',
    plot_rdms: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Streamlines the RSA pipeline from raw BOLD data to RSA score.

    This function handles loading fMRI BOLD data for a specific functional run
    across multiple subjects, optionally performing statistical voxel selection,
    applying a network mask, adjusting for hemodynamic delay, filtering padding volumes,
    and loading/extracting AI features before performing Representational Similarity Analysis.

    Parameters
    ----------
    bold_data_root_dir : str
        Path to the root directory containing BIDS-like fMRI data
        (e.g., 'path/to/cifti_analysis' or 'data/bids_processed/').
    functional_run_name : str
        The common name for the functional run across subjects, corresponding to
        the `task-<label>` or `task-<label>[_run-<index>]` part in BOLD file names.
        This also serves as the key in your precomputed_features_h5_path
        (e.g., 'inamoment' or 'coco_run-01').
    precomputed_features_h5_path : str, optional
        Path to the .h5 file containing precomputed AI features, where keys are
        functional run names. Required if `extract_features_on_the_fly` is False.
    network_of_interest : str, optional
        Name of the brain network to parcellate by (e.g., 'auditory', 'language', 'visual_A').
        If None, RSA will be performed on the entire brain data (after statistical
        selection if `perform_statistical_significance_testing` is True).
    h_delay_trs : int, optional
        Hemodynamic delay in TRs. BOLD data will be shifted by this amount.
        Defaults to 2.
    silence_padding_trs : int, optional
        Number of TRs to remove from the beginning and end of both BOLD and
        feature data, typically corresponding to silence padding. Defaults to 10.
    desired_subjects : list, optional
        A list of subject IDs (e.g., ['sub-UTS01', 'sub-UTS02']) to include in
        the analysis. If None, all subjects found for the `functional_run_name`
        within `bold_data_root_dir` will be included.
    perform_statistical_significance_testing : bool, optional
        If True, `get_significant_voxels_from_arrays` will be called to select voxels
        based on statistical significance across subjects. If False, the mean
        of BOLD data across subjects will be used (if multiple subjects are involved).
        Defaults to True.
    extract_features_on_the_fly : bool, optional
        If True, features will be extracted using `extract_features` function.
        Requires `stimuli_base_path`, `ai_model_id`, `ai_layer_name`.
        If False, `precomputed_features_h5_path` must be provided. Defaults to False.
    stimuli_base_path : str, optional
        Path to the root directory of your raw stimuli (e.g., 'data/stimuli/').
        Required if `extract_features_on_the_fly` is True.
    tr : float,
        The TR (repetition time) duration of your fMRI data in seconds.
        Required if `extract_features_on_the_fly` is True for determining how to segment stimuli into TR-aligned features. Defaults to 2.
    ai_model_id : str, optional
        Identifier for the AI model to use for feature extraction (e.g., 'whisper').
        Required if `extract_features_on_the_fly` is True.
    ai_model_checkpoint : str, optional
        checkpoint for the AI model to use for feature extraction (e.g., 'whisper-base').
    ai_layer_name : str, optional
        Name of the AI model layer to extract features from.
        if not specified, final layer's features will be used.
    rsa_metric : str, optional
        The distance metric for RSA (e.g., 'cosine'). Defaults to 'cosine'.
    plot_rdms : bool, optional
        If True, plots the computed RDMs. Defaults to True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        A tuple containing:
        - features_rdm : np.ndarray
            The normalized Representational Dissimilarity Matrix (RDM) for the features.
        - bold_rdm : np.ndarray
            The normalized Representational Dissimilarity Matrix (RDM) for the BOLD signals.
        - dist_corr : float
            The distance correlation coefficient between the two RDMs.

    Raises
    ------
    ValueError
        If required paths or parameters are missing or invalid.
    FileNotFoundError
        If BOLD or feature files are not found.
    """

    # --- 1. Load BOLD Data for all relevant subjects ---
    bold_data_arrays = []

    print(f"Loading BOLD data for functional run: '{functional_run_name}'")

    
    task_label_in_filename = f"task-{functional_run_name}"

    search_pattern_cifti = os.path.join(bold_data_root_dir, 'sub-*','ses-*', 'func', f"sub-*_{task_label_in_filename}*_bold.dtseries.nii")

    found_bold_files = glob.glob(search_pattern_cifti) 
    
    if not found_bold_files:
        raise FileNotFoundError(f"No BOLD files found matching '{task_label_in_filename}' "
                                f"in '{bold_data_root_dir}' for any subject.")

    # Filter by desired_subjects if specified
    if desired_subjects:
        initial_count = len(found_bold_files)
        filtered_bold_files = []
        for fpath in found_bold_files:
            # Extract subject ID from the full path: e.g., 'sub-UTS01' from '/path/to/data/sub-UTS01/func/...'
            subject_dir_name = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            
            # Simple check if the subject_dir_name itself is the desired_subject
            # Or if desired_subjects contains the full 'sub-XXX' string
            if any(sub in subject_dir_name for sub in desired_subjects):
                filtered_bold_files.append(fpath)
        
        if not filtered_bold_files:
            raise FileNotFoundError(
                f"No BOLD files found for functional run '{functional_run_name}' "
                f"among the desired subjects: {desired_subjects}. "
                f"Found {initial_count} files initially, but none matched desired subjects."
            )
        found_bold_files = filtered_bold_files
    
    # Sort files for consistent processing order if needed (good practice)
    found_bold_files.sort()

    for fpath in found_bold_files:
        print(f"Loading BOLD file: {fpath}")
        try:
            bold_data_arrays.append(nib.load(fpath).get_fdata())
        except Exception as e:
            print(f"Warning: Could not load BOLD file {fpath}. Skipping. Error: {e}")
            continue

    if not bold_data_arrays:
        raise ValueError("No valid BOLD data could be loaded for the specified run and subjects.")

    # --- 2. Statistical Significance Testing / Mean BOLD ---
    if perform_statistical_significance_testing:
        print("Performing statistical significance testing on BOLD data across subjects...")
        # get_significant_voxels_from_arrays expects a list of arrays (TRs x Voxels)
        processed_bold_data = get_significant_voxels_from_arrays(bold_data_arrays)
    else:
        print("Averaging BOLD data across subjects (no statistical testing).")
        # If no statistical testing, simply average or take the first subject's data if only one
        if len(bold_data_arrays) > 1:
            # Ensure all arrays have the same shape before averaging
            first_shape = bold_data_arrays[0].shape
            if not all(arr.shape == first_shape for arr in bold_data_arrays):
                print("Warning: BOLD arrays have inconsistent shapes. Averaging might lead to errors. "
                      "Using the first array for shape reference.")
                # As a fallback, ensure all arrays are truncated/padded to the smallest common TR count
                min_trs = min(arr.shape[0] for arr in bold_data_arrays)
                temp_arrays = [arr[:min_trs, :] for arr in bold_data_arrays]
                processed_bold_data = np.mean(np.array(temp_arrays), axis=0)
            else:
                processed_bold_data = np.mean(np.array(bold_data_arrays), axis=0)
        else:
            processed_bold_data = bold_data_arrays[0]
            print("Only one subject's BOLD data loaded, no averaging performed.")


    # --- 3. Hemodynamic Delay Adjustment ---
    if h_delay_trs > 0:
        print(f"Applying hemodynamic delay of {h_delay_trs} TRs to BOLD data...")
        processed_bold_data = processed_bold_data[h_delay_trs:]
    
    # --- 4. Network Parcellation (if specified) ---
    parcellated_bold_data = processed_bold_data # Initialize
    if network_of_interest:
        print(f"Parcellating BOLD data by network: {network_of_interest}")
        try:
            left_mask= parcellate_brain_by_network(f"left_{network_of_interest}")
            right_mask= parcellate_brain_by_network(f"right_{network_of_interest}")
            network_mask = (left_mask | right_mask).astype(bool)
            # Ensure the mask is 1D and matches the number of voxels in BOLD data
            if network_mask.ndim > 1:
                network_mask = network_mask.flatten()
            
            # Check if mask dimension matches BOLD data voxel dimension
            if network_mask.shape[0] != parcellated_bold_data.shape[1]:
                 raise ValueError(
                    f"Network mask dimension ({network_mask.shape[0]}) does not match "
                    f"BOLD data voxel dimension ({parcellated_bold_data.shape[1]}). "
                    "Ensure parcellation resolution matches your BOLD data space (e.g., 91k vertices)."
                )

            # Apply mask
            parcellated_bold_data = parcellated_bold_data[:, network_mask == 1]
            if parcellated_bold_data.shape[1] == 0:
                raise ValueError(f"No voxels found in the '{network_of_interest}' network based on the mask. "
                                 "Check network name or parcellation resolution/alignment.")
            print(f"BOLD data parcellated. New shape: {parcellated_bold_data.shape}")
        except Exception as e:
            raise ValueError(f"Error during network parcellation for '{network_of_interest}': {e}")

    # --- 5. Load or Extract AI Features ---
    ai_features_for_run = None
    if precomputed_features_h5_path is None:
        if extract_features_on_the_fly:
            if not all([stimuli_base_path, ai_model_id]):
                raise ValueError("For on-the-fly feature extraction, 'stimuli_base_path' and "
                                "'ai_model_id' are required.")
            print(f"Extracting features for functional run '{functional_run_name}' on the fly...")
            features_h5_path = f"temp_extracted_ai_features_{functional_run_name}.h5" 
            try:
                extract_features(
                    bold_data_root_dir= bold_data_root_dir,
                    stimuli_base_dir= stimuli_base_path,
                    output_file_path= features_h5_path,
                    model_id= ai_model_id,
                    layer_name= ai_layer_name,
                    checkpoint= ai_model_checkpoint,
                    tr= tr,
                    task_of_interest= functional_run_name
                )

            except Exception as e:
                raise ValueError(f"Error during on-the-fly feature extraction: {e}")
        else:
            raise ValueError("Either 'precomputed_features_h5_path' must be provided or "
                         "'extract_features_on_the_fly' must be True with all required parameters.")
    else:
        features_h5_path = precomputed_features_h5_path # Use user-provided path


    print(f"Loading AI features for '{functional_run_name}' from: {features_h5_path}")
    try:
        with h5py.File(features_h5_path, 'r') as f:
            if functional_run_name not in f:
                raise KeyError(f"Functional run name '{functional_run_name}' not found as a key in "
                               f"'{features_h5_path}'. Available keys: {list(f.keys())}")
            ai_features_for_run = get_stimuli_features_per_bold_signal(features_h5_path, functional_run_name)
    except Exception as e:
        raise ValueError(f"Error loading AI features from '{features_h5_path}': {e}")
    
    # Adjust features for hemodynamic delay (they should be shifted *backwards* relative to BOLD)
    # If BOLD is TR[2:], then features should be TR[:-2] to align.
    if h_delay_trs > 0:
        ai_features_for_run = ai_features_for_run[:-h_delay_trs]

    # --- 6. Remove Padding ---
    if padding_trs > 0:
        print(f"Removing {padding_trs} TRs of padding from BOLD and features.")
        
        # Check if arrays are long enough after delay adjustment
        required_length = 2 * padding_trs
        if parcellated_bold_data.shape[0] < required_length:
            raise ValueError(f"BOLD data ({parcellated_bold_data.shape[0]} TRs) is too short "
                             f"to remove {required_length} padding TRs after delay adjustment.")
        if ai_features_for_run.shape[0] < required_length:
            raise ValueError(f"AI features ({ai_features_for_run.shape[0]} TRs) are too short "
                             f"to remove {required_length} padding TRs after delay adjustment.")

        parcellated_bold_data = parcellated_bold_data[padding_trs:-padding_trs, :]
        ai_features_for_run = ai_features_for_run[padding_trs:-padding_trs, :]
    
    # Final check on TR counts
    if parcellated_bold_data.shape[0] != ai_features_for_run.shape[0]:
        raise ValueError(
            f"Mismatch in TR count: "
            f"BOLD ({parcellated_bold_data.shape[0]} TRs) vs. Features ({ai_features_for_run.shape[0]} TRs). "
        )
    
    print(f"\nFinal BOLD data shape for RSA: {parcellated_bold_data.shape}")
    print(f"Final AI features shape for RSA: {ai_features_for_run.shape}")

    # --- 7. Perform RSA ---
    print(f"\nPerforming RSA for functional run: {functional_run_name}")
    features_rdm, bold_rdm, rsa_score = compute_rsa(
        features=ai_features_for_run,
        bold_signals=parcellated_bold_data,
        metric=rsa_metric,
        plot_rdms=plot_rdms,
        plot_title_features=f"AI Features RDM for '{functional_run_name}'",
        plot_title_bold=f"BOLD RDM for '{functional_run_name}'"
    )

    return features_rdm, bold_rdm, rsa_score
