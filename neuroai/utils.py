import os
import glob
import h5py
import numpy as np
import hcp_utils as hcp
import nibabel as nib
from sklearn.utils import check_random_state
import nilearn.plotting as plotting

def get_nii_filenames(directory: str) -> list[str]:
    """
    Retrieves all .nii filenames (without paths) in a specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing .nii files.

    Returns
    -------
    list[str]
        A list of .nii filenames found in the directory.
    """
    return [os.path.basename(file) for file in glob.glob(os.path.join(directory, "*.nii"))]


def get_voxels_count(subject_base_dir: str) -> int:
    """
    Determines the number of voxels (features) in the BOLD data by searching
    for the first .nii file within any 'ses-XX/func' subdirectory of the given subject's base directory.

    Assumes all BOLD files for a subject have the same number of voxels.

    Parameters
    ----------
    subject_base_dir : str
        Path to the subject's base directory (e.g., `base_dir/sub-XX/`).
        The function will search within `subject_base_dir/ses-XX/func/` for BOLD files.

    Returns
    -------
    int
        The number of voxels (second dimension of the BOLD data array).

    Raises
    ------
    FileNotFoundError
        If no .nii files are found in any 'ses-XX/func' subdirectory within the subject's base directory.
    """
    for session in os.listdir(subject_base_dir):
        session_path = os.path.join(subject_base_dir, session)
        if os.path.isdir(session_path) and session.lower().startswith("ses"):
            bold_func_dir = os.path.join(session_path, 'func')
            if os.path.isdir(bold_func_dir):
                bold_files = get_nii_filenames(bold_func_dir)
                if not bold_files:
                    continue
                # Load the first BOLD file to determine the number of voxels
                first_bold_path = os.path.join(bold_func_dir, bold_files[0])
                try:
                    bold_data = nib.load(first_bold_path).get_fdata()
                    no_voxels = bold_data.shape[1]
                    return no_voxels # Return as soon as the first valid file is found
                except Exception as e:
                    print(f"Error loading {first_bold_path}: {e}. Skipping this file.")
                    continue

    # If the loop finishes without returning, no BOLD files were found
    raise FileNotFoundError(f"No .nii files found in any 'ses-XX/func' directory within: {subject_base_dir}")



def train_test_split_of_bold_signals(
    subject_base_dir: str,
    bold_test: list[str],
    bold_excluded: list[str] = None
) -> tuple[list[str], list[str]]:
    """
    Splits paths of BOLD signal files into training and testing subsets.

    The splitting logic is primarily based on a `bold_test` list, which contains
    identifiers for test signals. Signals matching any item in `bold_test` are
    assigned to the test set. All other signals (that are not in `bold_excluded`)
    are assigned to the training set.

    Parameters
    ----------
    subject_base_dir : str
        The path to the subject's base directory (e.g., `base_dir/sub-XX/`).
        The function will search within its `ses-XX/func/` subdirectories.
    bold_test : list[str]
        A list of string identifiers (e.g., 'task-rep_run-01') used to identify
        BOLD files belonging to the test set.
    bold_excluded : list[str], optional
        A list of string identifiers for BOLD signals that should be excluded
        from both training and testing sets. Defaults to an empty list.

    Returns
    -------
    tuple[list[str], list[str]]
        - train_subset (list): Full paths of BOLD signals for the training set.
        - test_subset (list): Full paths of BOLD signals for the test set.
    """
    if bold_excluded is None:
        bold_excluded = []
    test_subset = []
    train_subset = []
    # Iterate through sessions (e.g., 'ses-01', 'ses-02')
    for session in os.listdir(subject_base_dir):
        session_path = os.path.join(subject_base_dir, session)
        if os.path.isdir(session_path) and session.lower().startswith("ses"):
            bold_dir_path = os.path.join(session_path, 'func')
            if os.path.isdir(bold_dir_path):
                bold_signals = get_nii_filenames(bold_dir_path)

                for bold_signal in bold_signals:
                    if any(excluded in bold_signal for excluded in bold_excluded):
                        continue  # Skip the current iteration
                    matched = False  # Flag to track if a test bold_signal is matched

                    for test_signal in bold_test:
                        if test_signal in bold_signal:
                            test_subset.append(os.path.join(bold_dir_path, bold_signal))
                            matched = True
                            break  # Break after finding the first match
                    if not matched:
                        train_subset.append(os.path.join(bold_dir_path, bold_signal))

    return train_subset, test_subset


def get_h5_file_keys(h5_file: str) -> list[str]:
    """
    Retrieves the top-level keys (group names) from an HDF5 file.
    In the context of this project, these often correspond to BOLD signal names.

    Parameters
    ----------
    h5_file : str
        Path to the HDF5 (.h5) file.

    Returns
    -------
    list[str]
        A list of top-level keys within the HDF5 file.
    """
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        file_keys = list(f.keys()) 
    return file_keys

def get_stimuli_features_per_bold_signal(features_file: str, bold_signal_name: str) -> np.ndarray:
    """
    Extracts the stimuli features corresponding to a specific BOLD signal name
    from an HDF5 features file.

    Parameters
    ----------
    features_file : str
        Path to the HDF5 (.h5) file containing the features.
    bold_signal_name : str
        The key (name) of the BOLD signal within the HDF5 file
        for which to retrieve features.

    Returns
    -------
    np.ndarray
        A NumPy array containing the stimuli features for the given BOLD signal.
        Expected shape: (n_TRs, n_features).
    """
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    with h5py.File(features_file, 'r') as f:
        bold_features = {key: f[bold_signal_name][key][()] for key in f[bold_signal_name]}
    return list(bold_features.values())[0]

def get_feature_space_dim(h5_file: str) -> int:
    """
    Determines the dimensionality of the feature space stored in an HDF5 file.
    It does this by loading the features for the first key (BOLD signal name)
    and returning the second dimension (number of features).

    Parameters
    ----------
    h5_file : str
        Path to the HDF5 (.h5) file containing features.

    Returns
    -------
    int
        The dimension of the feature vector (e.g., number of semantic features).

    """
    names = get_h5_file_keys(h5_file)
    if not names:
        raise ValueError(f"HDF5 file '{h5_file}' is empty or contains no keys.")
    
    # Assuming all feature sets have the same dimension
    first_bold_name = names[0]
    features_dim = get_stimuli_features_per_bold_signal(h5_file, first_bold_name).shape[1]
    return features_dim


def standardize(data_matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Standardizes a data matrix (mean 0, standard deviation 1) across a specified axis.

    Parameters
    ----------
    data_matrix : np.ndarray
        The input NumPy array to be standardized.
        E.g., for BOLD data (TRs, Voxels), axis=0 standardizes each voxel time-series.
    axis : int, optional
        The axis along which to standardize. By default (0), it standardizes
        columns independently.

    Returns
    -------
    np.ndarray
        The standardized data matrix, with the same shape as the input.
    """
    mean = data_matrix.mean(axis=axis, keepdims=True)
    std = data_matrix.std(axis=axis, keepdims=True)

    # Set any std == 0 to 1 to avoid division by zero
    std[std == 0] = 1.0

    standardized = (data_matrix - mean) / std
    return standardized


def prepare_modeling_data(
    subject_base_dir: str,
    features_file_path: str,
    features_dim: int,
    no_voxels: int,
    bold_test: list[str],
    bold_excluded: list[str] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Prepares training and testing data (BOLD signals and corresponding features)
    for a neural encoding or decoding model, adhering to a BIDS-like directory structure.
    Parameters
    ----------
    subject_base_dir : str
        Path to the subject's base directory (e.g., `base_dir/sub-XX/`)
        containing the `ses-XX/func/` subdirectories with BOLD fMRI `.nii` data files.
    features_file_path : str
        Path to the HDF5 (.h5) file containing the encoded stimuli features.
    features_dim : int
        The dimensionality of the encoded stimuli features (e.g., 768 for CLIP embeddings).
    no_voxels : int
        The total number of voxels in the BOLD data (second dimension of BOLD array).
    bold_test : list[str]
        A list of string identifiers corresponding to the BOLD signals
        (e.g., story names, task names) that should be used for the test set.
        These identifiers should match parts of the BOLD filenames and HDF5 keys.
    bold_excluded : list[str], optional
        A list of string identifiers for BOLD signals to be excluded
        from both training and testing. Defaults to an empty list.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]
        - X_train (np.ndarray): Concatenated and standardized feature array for training.
                                Shape: (Total_TRs_train, features_dim).
        - Y_train (np.ndarray): Concatenated and standardized BOLD signals for training.
                                Shape: (Total_TRs_train, no_voxels).
        - X_test (np.ndarray): Concatenated and standardized feature array for testing.
                                Shape: (Total_TRs_test, features_dim).
        - Y_test (np.ndarray): Concatenated and standardized (and averaged for repetitions)
                                BOLD signals for testing. Shape: (Total_TRs_test, no_voxels).
        - train_run_onsets (list[int]): List of cumulative TR counts indicating the
                                        starting index of each training run after concatenation.
                                        Used for preserving run boundaries during modeling.
    """

    train_bold_paths, test_bold_paths = train_test_split_of_bold_signals(subject_base_dir, bold_test, bold_excluded
    )
    bold_signal_names = get_h5_file_keys(features_file_path)

    X_train = np.empty((0, features_dim))
    Y_train = np.empty((0, no_voxels))
    X_test = np.empty((0, features_dim))
    Y_test = np.empty((0, no_voxels))
    train_run_onsets = [0]

    bold_path_to_name = {}
    for bold_name in bold_signal_names:
        for bold_path in train_bold_paths:
            if bold_name in os.path.basename(bold_path):
                bold_path_to_name[bold_path] = bold_name
                break

    print("----------------------------------------------- Processing Training Stories ----------------------------------------------")
    for i, bold_path in enumerate(train_bold_paths):
        bold_name = bold_path_to_name.get(bold_path)
        print(f"current BOLD file: {bold_name}")
        if not bold_name:
            continue

        stimuli_features = get_stimuli_features_per_bold_signal(features_file_path, bold_name)

        if i < len(train_bold_paths) - 1:
            previous_onset = train_run_onsets[-1]
            new_onset = previous_onset + stimuli_features.shape[0]
            train_run_onsets.append(new_onset)

        X_train = np.concatenate((X_train, stimuli_features), axis=0)
        print(f"The encoded stimuli features shape: {stimuli_features.shape}")

        bold_data = nib.load(bold_path).get_fdata()
        bold_data = standardize(bold_data)
        Y_train = np.concatenate((Y_train, bold_data), axis=0)
        print(f"target (shape): {bold_data.shape}")
        print("-----------------------------------------------------------------------------------------------------")

    print("----------------------------------------------- Processing Test Stories ----------------------------------------------")
    for bold_name in bold_test:
        print(f"current story: {bold_name}")
        test_features = get_stimuli_features_per_bold_signal(features_file_path, bold_name)
        print(f"The encoded stimuli features shape: {test_features.shape}")
        X_test = np.concatenate((X_test, test_features), axis=0)

        accumulated_bold = None
        repetition_count = 0

        for bold_path in test_bold_paths:
            if bold_name in os.path.basename(bold_path):
                bold_data = nib.load(bold_path).get_fdata()
                bold_data = standardize(bold_data)
                if accumulated_bold is None:
                    accumulated_bold = bold_data
                else:
                    accumulated_bold += bold_data
                repetition_count += 1

        if repetition_count > 1:
            averaged_bold = accumulated_bold / repetition_count
            Y_test = np.concatenate((Y_test, averaged_bold), axis=0)
            print(f"target (shape): {averaged_bold.shape}")
        elif repetition_count == 1:
            Y_test = np.concatenate((Y_test, accumulated_bold), axis=0)
            print(f"target (shape): {accumulated_bold.shape}")
        else:
            print(f"No repetitions found for test story: {bold_name}")

        print("-----------------------------------------------------------------------------------------------------")

    return X_train, Y_train, X_test, Y_test, train_run_onsets

def get_subjects_results(results_dir: str) -> dict[str, np.ndarray]:
    """
    Loads R-squared scores for multiple subjects from a results directory.

    This function expects the `results_dir` to contain subdirectories,
    each named starting with "sub" (e.g., "sub-01", "sub-02"), and
    each of these subdirectories to contain a "scores.npy" file.

    Parameters
    ----------
    results_dir : str
        The path to the main directory containing subject-specific result
        subdirectories.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where keys are subject directory names (e.g., "sub-01")
        and values are NumPy arrays containing the R-squared scores loaded
        from their respective "scores.npy" files.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    subjects_scores = {}
    print(f"Loading subject results from: {results_dir}")

    for inner_dir in os.listdir(results_dir):
        subject_dir_path = os.path.join(results_dir, inner_dir)
        if os.path.isdir(subject_dir_path) and inner_dir.lower().startswith("sub"):
            scores_file_path = os.path.join(subject_dir_path, "scores.npy")
            if os.path.exists(scores_file_path):
                try:
                    subjects_scores[inner_dir] = np.load(scores_file_path)
                    print(f"  Loaded scores for {inner_dir}")
                except Exception as e:
                    print(f"  Warning: Could not load scores from {scores_file_path}: {e}. Skipping {inner_dir}.")
            else:
                print(f"  Warning: 'scores.npy' not found in {subject_dir_path}. Skipping {inner_dir}.")
    
    if not subjects_scores:
        print(f"No subject score files found in {results_dir} matching 'sub-XX/scores.npy' pattern.")

    return subjects_scores

def render_scores(scores: np.ndarray, threshold: str = "90%", title: str = "Rendered R-squared Scores"):
        """
        Renders voxel-wise scores (e.g., R-squared) onto an inflated brain surface
        using `nilearn.plotting.view_surf`.

        This function assumes `hcp_utils` is configured correctly to provide
        mesh data (`hcp.mesh.inflated`, `hcp.mesh.sulc`) and cortex data.

        Parameters
        ----------
        scores : np.ndarray
            A NumPy array of scores for each voxel/vertex.
            Shape: `(n_voxels,)`.
        threshold : str or float, optional
            Threshold for displaying values on the surface map. Can be a percentage
            string (e.g., "90%") or a float (e.g., 0.1). Defaults to "90%".
        title : str, optional
            Title for the rendered brain plot. Defaults to "Rendered R-squared Scores".

        Returns
        -------
        nilearn.plotting.html_view.HTMLView
            An interactive HTML viewer object of the brain surface plot.
        """
        data = scores.copy()
        if threshold is str:
            threshold= int(threshold.replace("%", ''))
            threshold = np.percentile(data, threshold)
        else:
            threshold = threshold
        view= plotting.view_surf(
            hcp.mesh.inflated,
            hcp.cortex_data(data),
            title= title,
            cmap='Reds',
            symmetric_cmap=False,
            vmin= threshold,
            bg_map=hcp.mesh.sulc
            )
        return view

