from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_1samp
import numpy as np
from typing import  List
from neuroai.utils import get_subjects_results
from neuroai.parcellation import parcellate_brain_by_network


def _perform_t_test_and_fdr_correction(data_to_test: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to perform a one-sample t-test against a population mean of zero
    and then apply FDR correction to the resulting p-values.

    Parameters
    ----------
    data_to_test : np.ndarray
        The input data array. For a one-sample t-test across subjects/trials,
        the first dimension should represent samples (e.g., subjects or repetitions),
        and subsequent dimensions represent the observations (e.g., voxels, RDM values).
        Expected shape: (n_samples, ...)
    alpha : float, optional
        The significance level for the FDR correction. Defaults to 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - significant_mask : np.ndarray
            A boolean array of the same shape as the input data's dimensions
            excluding the first (sample) dimension, where True indicates
            statistically significant elements after FDR correction.
        - corrected_p_values : np.ndarray
            The p-values after FDR correction, reshaped to match the
            dimensions of the input data's observations.
    """
    # Perform one-sample t-test against popmean=0
    # ttest_1samp returns (t_statistics, p_values)
    # The p_values will have the same shape as data_to_test[0] (i.e., (...,))
    _, p_values = ttest_1samp(data_to_test, popmean=0, axis=0)

    # Flatten the p-value array to 1D for FDR correction
    p_values_flat = p_values.ravel()

    # Apply FDR correction
    significant_mask_flat, corrected_p_values_flat = fdrcorrection(p_values_flat, alpha=alpha)

    # Reshape the mask and corrected p-values back to the original p_values shape
    significant_mask = significant_mask_flat.reshape(p_values.shape)
    corrected_p_values = corrected_p_values_flat.reshape(p_values.shape)

    return significant_mask, corrected_p_values


def get_significant_voxels_from_arrays(bold_arrays: List[np.ndarray], alpha: float = 0.05) -> np.ndarray:
    """
    Identifies statistically significant voxels (features) from a list of BOLD data arrays
    (e.g., from multiple subjects or repetitions).

    It performs a one-sample t-test against zero for each voxel across the provided
    arrays and applies False Discovery Rate (FDR) correction to the p-values.
    The output is the mean BOLD signal, with non-significant voxels masked out (set to zero).

    Parameters
    ----------
    bold_arrays : list[np.ndarray]
        A list of NumPy arrays, where each array represents BOLD data for a single
        subject or repetition. Each array is expected to have shape `(n_timepoints, n_voxels)`.
        All arrays in the list are assumed to have the same `n_voxels`.
    alpha : float, optional
        The significance level for the FDR correction. Defaults to 0.05.

    Returns
    -------
    np.ndarray
        A NumPy array representing the average BOLD signal across the input `bold_arrays`,
        where only statistically significant voxels (after FDR correction) retain their
        mean values, and non-significant voxels are set to zero.
        Shape: `(n_timepoints, n_voxels)`.

    """
    if not bold_arrays:
        raise ValueError("Input `bold_arrays` list cannot be empty.")
    
    # Stack the arrays along a new axis (axis=0) for t-test across subjects/repetitions
    # Resulting shape: (n_subjects/repetitions, n_timepoints, n_voxels)
    stacked_bold_data = np.stack(bold_arrays, axis=0)
    
    # Calculate the mean across subjects/repetitions for each voxel time series
    average_bold_array = np.mean(stacked_bold_data, axis=0) # Shape: (n_timepoints, n_voxels)

    # Perform t-test and FDR correction using the helper function across the subject dimension.
    significant_mask, _ = _perform_t_test_and_fdr_correction(stacked_bold_data, alpha=alpha)

    # Apply the significant mask to the average array
    # This will set non-significant voxels to 0
    significant_mean_map = average_bold_array * significant_mask

    return significant_mean_map


def get_significant_model_scores(results_dir: str, alpha: float = 0.05) -> np.ndarray:
    """
    Identifies statistically significant voxel-wise model scores (e.g., R-squared values)
    from a directory containing subject-specific results.

    It loads scores from `scores.npy` files within `sub-XX` subdirectories
    of `results_dir`, stacks them, performs a one-sample t-test against zero for
    each voxel's score, and applies False Discovery Rate (FDR) correction.
    The output is the mean score map, with non-significant voxels masked out (set to zero).

    Parameters
    ----------
    results_dir : str
        The path to the main directory containing subject-specific result
        subdirectories (e.g., `path/to/vm_results/CLIP_features/`).
        Each subdirectory (e.g., `results_dir/sub-01/`) is expected to contain
        a `scores.npy` file.
    alpha : float, optional
        The significance level for the FDR correction. Defaults to 0.05.

    Returns
    -------
    np.ndarray
        A NumPy array representing the average model scores across subjects,
        where only statistically significant scores (after FDR correction)
        retain their mean values, and non-significant scores are set to zero.
        Shape: `(n_voxels,)`.
    """
    # Use the existing utility function to load subject results
    subjects_scores = get_subjects_results(results_dir=results_dir)

    if not subjects_scores:
        raise FileNotFoundError(
            f"No valid subject scores (`scores.npy` files) found in '{results_dir}' "
            f"or its subdirectories starting with 'sub'."
        )

    # Stack the subject score arrays. `get_subjects_results` returns a dict of arrays.
    score_arrays_list = [arr for arr in subjects_scores.values()]
    
    # Check for consistency in shapes
    if not score_arrays_list:
        raise ValueError("No score arrays were loaded, cannot perform statistical test.")
    
    first_score_shape = score_arrays_list[0].shape
    for arr in score_arrays_list:
        if arr.shape != first_score_shape:
            raise ValueError(
                f"Inconsistent score array shapes. Expected {first_score_shape}, "
                f"but found an array with shape {arr.shape}. Cannot stack."
            )

    # Stack along a new axis (axis=0) to perform t-test across subjects
    # Resulting shape: (n_subjects, n_voxels)
    subjects_scores_stack = np.stack(score_arrays_list, axis=0)

    # Calculate the mean across subjects
    mean_score_map = np.mean(subjects_scores_stack, axis=0) # Shape: (n_voxels,)

    # Perform t-test and FDR correction using the helper function
    significant_mask, _ = _perform_t_test_and_fdr_correction(subjects_scores_stack, alpha=alpha)
    
    # Apply the significant mask to the mean score map
    # This will set non-significant scores to 0
    significant_mean_map = mean_score_map * significant_mask

    return significant_mean_map


def get_significantly_different_regions_between_hemispheres(
    results_dir: str, 
    functional_network: str, 
    threshold_scores: float = 0, 
    alpha: float = 0.05
) -> None:
    """
    Tests for statistically significant lateralization (difference between left and right
    hemispheres) within a specified functional brain network, based on subject-specific
    model scores (e.g., R-squared).

    It loads scores for all subjects, filters them by a threshold, applies left and right
    hemisphere masks for the given network, calculates the mean score difference
    (Left - Right) for each subject, and then performs a one-sample t-test on these
    differences across subjects, followed by FDR correction.

    Parameters
    ----------
    results_dir : str
        The path to the main directory containing subject-specific result
        subdirectories (e.g., `path/to/vm_results/`).
        Each subdirectory (e.g., `results_dir/sub-01/`) is expected to contain
        a `scores.npy` file.
    functional_network : str
        The base name of the functional network (e.g., 'auditory', 'language', 'visual_A').
        The function will internally construct 'left_<network>' and 'right_<network>'
        for parcellation.
    threshold_scores : float, optional
        A threshold applied to individual subject scores. Only scores strictly
        greater than this threshold will be considered for averaging within
        hemispheres. This can be used to focus on 'high responses'. Defaults to 0.
    alpha : float, optional
        The significance level for the FDR correction. Defaults to 0.05.

    Returns
    -------
    None
        Prints a message indicating whether significant lateralization was found
        and the FDR-corrected p-value (q-value).

    """
    subjects_scores = get_subjects_results(results_dir=results_dir)

    if not subjects_scores:
        raise FileNotFoundError(
            f"No valid subject scores (`scores.npy` files) found in '{results_dir}' "
            f"or its subdirectories starting with 'sub'."
        )

    left_functional_network_id = f"left_{functional_network}"
    right_functional_network_id = f"right_{functional_network}"

    try:
        left_mask = parcellate_brain_by_network(left_functional_network_id)
        right_mask = parcellate_brain_by_network(right_functional_network_id)
    except Exception as e:
        raise ValueError(f"Error loading network masks for {functional_network}: {e}")
        
    # Combine masks to filter subjects' scores to only relevant voxels before analysis
    combined_network_mask = (left_mask | right_mask).astype(bool) # Boolean mask for voxels in either hemi of the network

    left_indices = np.where(left_mask == 1)[0]
    right_indices = np.where(right_mask == 1)[0]

    # Initialize a list to store mean differences for each subject
    subject_mean_differences = []

    print(f"Analyzing lateralization for '{functional_network}' network...")
    for subject_id, subject_score_array in subjects_scores.items():
        # Create a mutable copy if subject_score_array might be read-only
        processed_scores = np.copy(subject_score_array)

        # Apply a threshold and then the network mask
        processed_scores[processed_scores <= threshold_scores] = 0 # 
        processed_scores = processed_scores * combined_network_mask # Zero out voxels outside the combined network

        left_data = processed_scores[left_indices]
        right_data = processed_scores[right_indices]
        
        # Filter out zeros (due to thresholding) before calculating mean
        left_data_filtered = left_data[left_data > 0]
        right_data_filtered = right_data[right_data > 0]

        if len(left_data_filtered) == 0:
            print(f"  Warning: No significant positive scores in left '{functional_network}' for {subject_id}. Skipping mean calculation for this hemisphere.")
            mean_left = 0
        else:
            mean_left = np.mean(left_data_filtered)
        
        if len(right_data_filtered) == 0:
            print(f"  Warning: No significant positive scores in right '{functional_network}' for {subject_id}. Skipping mean calculation for this hemisphere.")
            mean_right = 0
        else:
            mean_right = np.mean(right_data_filtered)

        # Store the difference for this subject
        subject_mean_differences.append(mean_left - mean_right)
        print(f"  {subject_id}: Mean Left={mean_left:.4f}, Mean Right={mean_right:.4f}, Difference={mean_left - mean_right:.4f}")

    if not subject_mean_differences:
        raise ValueError("No valid mean differences could be calculated across subjects. Ensure scores, masks, and thresholds are appropriate.")

    # Convert the list of differences to a NumPy array for statistical testing
    differences_array = np.array(subject_mean_differences)
    
    # Perform t-test and FDR correction on the differences (shape: (n_subjects,))
    # The t-test will be 1-sample, checking if the mean difference is significantly different from 0
    _, q_value_array = _perform_t_test_and_fdr_correction(differences_array.reshape(-1, 1), alpha=alpha) # Reshape to (n_subjects, 1) for ttest_1samp
    q_value = q_value_array[0] # Get the single q-value

    print("\n--- Statistical Test Results ---")
    if q_value <= alpha:
        print(f"There is **SIGNIFICANT** lateralization in the {functional_network} network's scores.")
        print(f"  FDR-corrected p-value (q-value): {q_value:.4f} (<= {alpha})")
    else: 
        print(f"No **SIGNIFICANT** lateralization found in the {functional_network} network's scores.")
        print(f"  FDR-corrected p-value (q-value): {q_value:.4f} (> {alpha})")

    # Optional: Print mean difference and standard deviation
    print(f"  Overall mean left-right difference across subjects: {np.mean(differences_array):.4f}")
    print(f"  Standard deviation of left-right differences across subjects: {np.std(differences_array):.4f}")

    return



