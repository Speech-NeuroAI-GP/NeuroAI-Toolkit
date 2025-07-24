import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import signal
import subprocess
import glob
import re 
import shutil 
import platform
import sys
# --- Configuration for wb_command ---
# This will be dynamically determined
WB_COMMAND_PATH = None

def _check_wb_command_exists():
    """
    Checks if the wb_command executable is available.
    It first tries to find it in the system's PATH.
    If not found, it checks if the WB_COMMAND_PATH environment variable is set.
    """
    global WB_COMMAND_PATH

    system = platform.system()
    if system != "Linux":
        raise EnvironmentError(
            f"Connectome Workbench support is only available on Linux.\n"
            f"You're running on {system}, which is currently not supported."
        )

    # Check PATH first
    WB_COMMAND_PATH = shutil.which("wb_command")

    if WB_COMMAND_PATH is None:
        # Check env var
        env_path = os.environ.get("WB_COMMAND_PATH")
        if env_path and os.path.exists(env_path) and os.path.isfile(env_path):
            WB_COMMAND_PATH = env_path
        else:
            raise FileNotFoundError(
                "Connectome Workbench (wb_command) not found.\n"
                "Please install Workbench manually and ensure:\n"
                "- 'wb_command' is in your system's PATH, or\n"
                "- Set WB_COMMAND_PATH to its full path\n"
                "Guide: https://humanconnectome.org/software/connectome-workbench"
            )

    if not os.access(WB_COMMAND_PATH, os.X_OK):
        raise PermissionError(
            f"Connectome Workbench at {WB_COMMAND_PATH} is not executable.\n"
            f"Run: chmod +x {WB_COMMAND_PATH}"
        )
    print(f"Using wb_command from: {WB_COMMAND_PATH}")
    return True


def select_confounds(confound_file: str, selected_confounds: list[str]) -> pd.DataFrame:
    """
    Reads a BIDS fMRIPrep _desc-confounds_timeseries.tsv file and selects
    specified confound columns. Missing values are filled with 0.

    Parameters
    ----------
    confound_file : str
        Path to the fMRIPrep confounds TSV file.
    selected_confounds : list[str]
        A list of column names (confound regressors) to select from the TSV file.
        Common choices include:
        - 'csf', 'white_matter', 'global_signal' (for aCompCor components)
        - 'trans_x', 'trans_y', 'trans_z' (translation motion parameters)
        - 'rot_x', 'rot_y', 'rot_z' (rotation motion parameters)
        - 'dvars', 'framewise_displacement' (for scrubbing/outlier regression)
        - Derivatives like 'trans_x_derivative1', 'rot_x_derivative1', etc.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the selected confound regressors, with
        missing values (NaNs) filled with 0.
    """
    if not os.path.exists(confound_file):
        raise FileNotFoundError(f"Confounds file not found: {confound_file}")

    print(f"Loading confounds from: {confound_file}")
    confounds_df = pd.read_csv(confound_file, sep='\t')

    # Filter selected_confounds to only include those present in the DataFrame
    available_confounds = [col for col in selected_confounds if col in confounds_df.columns]
    
    if not available_confounds:
        raise ValueError(
            f"None of the requested confounds {selected_confounds} were found "
            f"in {confound_file}. Available columns: {list(confounds_df.columns)}"
        )
    
    # Select columns and fill missing values with 0
    confounds = confounds_df[available_confounds].fillna(0)
    print(f"Selected confounds: {available_confounds}")
    print(f"Shape of selected confounds: {confounds.shape}")

    return confounds


def clean_cifti_data(func_img: nib.Cifti2Image, confounds: np.ndarray) -> nib.Cifti2Image:
    """
    Performs confound regression on CIFTI-2 data using `nilearn.signal.clean`.

    This function applies a linear regression to remove the variance explained
    by the provided confound regressors from the fMRI BOLD signal.

    Parameters
    ----------
    func_img : nib.Cifti2Image
        The input CIFTI-2 image object (e.g., loaded with `nibabel.load`).
        Expected to have data of shape (TRs, Voxels).
    confounds : np.ndarray
        A 2D NumPy array of shape (TRs, N_confounds) containing the confound
        regressors to be removed. The number of rows must match the number
        of TRs in `func_img.get_fdata()`.

    Returns
    -------
    nib.Cifti2Image
        A new CIFTI-2 image object containing the cleaned (regressed) BOLD data.
        The header and nifti_header are preserved from the original image.
    """
    data = func_img.get_fdata() # Shape (TRs, Voxels)

    if data.shape[0] != confounds.shape[0]:
        raise ValueError(
            f"Mismatch in number of time points (TRs): BOLD data has {data.shape[0]} TRs, "
            f"but confounds have {confounds.shape[0]} TRs. Cannot perform regression."
        )

    print(f"Cleaning BOLD data of shape {data.shape} using confounds...")
    cleaned_data = signal.clean(data, confounds=confounds,
                                standardize=False, detrend=False)
    print(f"Cleaned data shape: {cleaned_data.shape}")

    return nib.Cifti2Image(cleaned_data, header=func_img.header,
                           nifti_header=func_img.nifti_header)


def smooth_cifti_image(input_file: str, output_file: str,
                       left_surface: str, right_surface: str, smoothing_mm: int ) -> bool:
    """
    Smooths a CIFTI-2 image using the Connectome Workbench `wb_command`.
    Smoothing is performed on the cortical surface (left and right hemispheres)
    and subcortical gray matter volumes.

    Parameters
    ----------
    input_file : str
        Path to the input CIFTI-2 `.dtseries.nii` file to be smoothed.
    output_file : str
        Path where the smoothed CIFTI-2 `.dtseries.nii` file will be saved.
    left_surface : str
        Path to the GIFTI surface file for the left hemisphere
        (e.g., `sub-01_ses-01_task-rest_space-fsLR_den-91k_bold.L.surf.gii`).
        This is typically obtained from fMRIPrep or HCP pipeline.
    right_surface : str
        Path to the GIFTI surface file for the right hemisphere.
    smoothing_mm : int, optional
        The FWHM (Full Width at Half Maximum) smoothing kernel size in millimeters.

    Returns
    -------
    bool
        True if smoothing was successful, False otherwise.

    """
    _check_wb_command_exists() # Ensure wb_command is callable

    if not all(os.path.exists(f) for f in [input_file, left_surface, right_surface]):
        raise FileNotFoundError(
            f"Missing input files for smoothing. Check existence of:\n"
            f"  Input CIFTI: {input_file}\n"
            f"  Left Surface: {left_surface}\n"
            f"  Right Surface: {right_surface}"
        )
    wb_command_full_path = WB_COMMAND_PATH # Use the dynamically determined path
    wb_command_dir = os.path.dirname(wb_command_full_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Construct the wb_command.
    wb_command = (
        f"{wb_command_full_path} -cifti-smoothing {input_file} {smoothing_mm} {smoothing_mm} "
        f"COLUMN {output_file} -left-surface {left_surface} -right-surface {right_surface}"
    )

    print(f"Executing smoothing command:\n{wb_command}")

    try:
        env = os.environ.copy()
        # Dynamically set LD_LIBRARY_PATH relative to wb_command's location
        # This assumes a typical Connectome Workbench installation structure.
        workbench_lib_path = os.path.join(wb_command_dir, '..', 'lib_linux64')
        env['LD_LIBRARY_PATH'] = f"{workbench_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

        result = subprocess.run(wb_command,
                                shell=True,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                env=env)

        print(f"Smoothing successful for {input_file}")
        if result.stdout:
            print("wb_command stdout:\n", result.stdout)
        if result.stderr:
            print("wb_command stderr:\n", result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error in smoothing command execution: {e}")
        print(f"Command returned non-zero exit code {e.returncode}.")
        if e.stdout:
            print("wb_command stdout:\n", e.stdout)
        if e.stderr:
            print("wb_command stderr:\n", e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during smoothing: {str(e)}")
        return False

def _parse_bids_filename(filepath: str) -> tuple[str, str, str]:
    """
    Parses a BIDS-compliant filename to extract subject, session, and task/run.
    Assumes filename format like: sub-<id>_ses-<id>_task-<id>[_run-<id>]..._bold.dtseries.nii

    Parameters
    ----------
    filepath : str
        The full path to the BIDS file.

    Returns
    -------
    tuple[str, str, str]
        (subject_id, session_id, task_id_with_run)
        `task_id_with_run` will be 'task-inamoment' or 'task-coco_run-01' etc.
        Returns empty strings if not found.
    """
    filename = os.path.basename(filepath)
    subject_match = re.search(r'sub-([a-zA-Z0-9]+)', filename)
    session_match = re.search(r'ses-([a-zA-Z0-9]+)', filename)
    task_match = re.search(r'task-([a-zA-Z0-9]+(?:_run-[a-zA-Z0-9]+)?)', filename) # Catches task-X and task-X_run-Y

    sub_id = f"sub-{subject_match.group(1)}" if subject_match else ""
    ses_id = f"ses-{session_match.group(1)}" if session_match else ""
    task_id = task_match.group(0) if task_match else "" # Keep the 'task-' prefix

    return sub_id, ses_id, task_id


def process_functional_run(
    bold_file_path: str,
    confound_file_path: str,
    output_base_dir: str,
    left_surface_file: str,
    right_surface_file: str,
    selected_confounds: list[str] = None,
    smoothing_kernel_size: int = None
) -> None:
    """
    Processes a single functional run by applying optional confound regression and
    optional spatial smoothing to a CIFTI-2 BOLD file.

    Parameters
    ----------
    bold_file_path : str
        Full path to the input CIFTI-2 BOLD file (e.g., `sub-01_ses-01_task-inamoment_bold.dtseries.nii`).
    confound_file_path : str
        Full path to the corresponding fMRIPrep confounds TSV file. This file is
        required if `selected_confounds` is not None.
    output_base_dir : str
        The root directory where cleaned and smoothed data will be saved.
        Output will be organized as:
        `output_base_dir/sub-<id>/ses-<id>/func/cleaned_data/`
    left_surface_file : str
        Path to the GIFTI surface file for the left hemisphere, required for smoothing.
    right_surface_file : str
        Path to the GIFTI surface file for the right hemisphere, required for smoothing.
    selected_confounds : list[str], optional
        List of confound regressors to use for cleaning. If None, no confound
        regression will be performed. Defaults to None.
    smoothing_kernel_size : int, optional
        FWHM smoothing kernel size in millimeters. If None, no smoothing
        will be performed. Defaults to None.
    """
    
    subject_id, session_id, task_id = _parse_bids_filename(bold_file_path)
    if not all([subject_id, session_id, task_id]):
        print(f"Could not parse BIDS info from: {bold_file_path}. Skipping.")
        return

    print(f"\nProcessing {subject_id} {session_id} {task_id}")

    # Define the single output directory
    final_output_dir = os.path.join(output_base_dir, subject_id, session_id, 'func')
    os.makedirs(final_output_dir, exist_ok=True)

    base_filename = os.path.basename(bold_file_path) 
    output_file_path = os.path.join(final_output_dir, base_filename)

    # Determine the input file for the *next* step (either original or regressed)
    current_input_cifti_obj = None # Will store nibabel Cifti2Image object
    
    # --- Step 1: Confound Regression (if requested) ---
    if selected_confounds:
        print("Applying confound regression...")
        try:
            func_img = nib.load(bold_file_path)
            confounds_df = select_confounds(confound_file_path, selected_confounds)
            current_input_cifti_obj = clean_cifti_data(func_img, confounds_df.values)
            print("Confound regression complete in memory.")
        except FileNotFoundError as e:
            print(f"Error: {e}. Confound file is mandatory for regression. Skipping this run.")
            return
        except Exception as e:
            print(f"Error during confound regression for {bold_file_path}: {str(e)}. Skipping this run.")
            return
    else:
        print("No confounds selected, skipping confound regression. Using original BOLD file as input for next step.")
        # If no regression, the original file is the input for smoothing 
        current_input_cifti_obj = nib.load(bold_file_path) # Load the original image

    # We need to save the Cifti2Image object to a temporary file if smoothing is requested,
    # as wb_command operates on disk files.
    temp_input_file_for_smoothing = None
    if smoothing_kernel_size is not None and smoothing_kernel_size > 0 :
        if selected_confounds is not None:        
            temp_dir = os.path.join(os.path.dirname(final_output_dir), "intermediate_regressed_only_data")
            os.makedirs(temp_dir, exist_ok=True)
            temp_input_file_for_smoothing = os.path.join(temp_dir, f"temp_{base_filename}")
        
            # Save the current Cifti2Image object to the temporary file
            try:
                current_input_cifti_obj.to_filename(temp_input_file_for_smoothing)
                print(f"Saved intermediate regressed only data to temporary file for smoothing: {temp_input_file_for_smoothing}")
            except Exception as e:
                print(f"Error saving intermediate CIFTI for smoothing: {e}. Skipping this run.")
                if os.path.exists(temp_input_file_for_smoothing): os.remove(temp_input_file_for_smoothing)
                return
        else:
            temp_input_file_for_smoothing= bold_file_path
        # --- Step 2: Spatial Smoothing (if requested) ---

        print(f"Smoothing data with {smoothing_kernel_size}mm FWHM kernel...")
        try:
            success = smooth_cifti_image(
                input_file=temp_input_file_for_smoothing, # Use the temporary file (either original or regressed)
                output_file=output_file_path,           # Save directly to final output path
                left_surface=left_surface_file,
                right_surface=right_surface_file,
                smoothing_mm=smoothing_kernel_size
            )
            if not success:
                print(f"Smoothing failed for {bold_file_path}. Skipping.")
                # Clean up temporary file
                if os.path.exists(temp_input_file_for_smoothing): os.remove(temp_input_file_for_smoothing)
                return
            print(f"Saved smoothed data to: {output_file_path}")
        except Exception as e:
            print(f"Error during smoothing for {bold_file_path}: {str(e)}. Skipping this run.")
            if os.path.exists(temp_input_file_for_smoothing): os.remove(temp_input_file_for_smoothing)
            return
        finally:
            # Clean up temporary file after smoothing
            if os.path.exists(temp_input_file_for_smoothing):
                os.remove(temp_input_file_for_smoothing)
                print(f"Removed temporary file: {temp_input_file_for_smoothing}")
    else:
        print("No smoothing kernel size specified, skipping spatial smoothing.")
        # If no smoothing, save the (potentially regressed) Cifti2Image object directly
        if selected_confounds: # If regression was done
            try:
                current_input_cifti_obj.to_filename(output_file_path)
                print(f"Saved regressed data (no smoothing) to: {output_file_path}")
            except Exception as e:
                print(f"Error saving regressed data to final output: {e}. Skipping this run.")
                return

    print(f"Completed processing for {subject_id} {session_id} {task_id}")
    print("-" * 50)


def run_cleaning_pipeline(
    bids_fmriprep_root_dir: str,
    output_base_dir: str,
    left_surface_template: str,
    right_surface_template: str,
    selected_confounds: list[str] = None,
    smoothing_kernel_size: int = None,
    target_subject_ids: list[str] = None,
    target_session_ids: list[str] = None,
    target_task_ids: list[str] = None,
) -> None:
    """
    Orchestrates the fMRI BOLD data cleaning pipeline across subjects and runs.

    This function iterates through a BIDS-fMRIPrep processed dataset,
    identifies functional runs, and applies confound regression and/or
    spatial smoothing to CIFTI-2 BOLD data.

    Parameters
    ----------
    bids_fmriprep_root_dir : str
        Path to the root directory of your fMRIPrep processed BIDS dataset.
        This directory should contain `sub-*` folders, each with `func/`
        subfolders holding `*_bold.dtseries.nii` and `*_desc-confounds_timeseries.tsv` files.
    output_base_dir : str
        The root directory where all cleaned and/or smoothed data will be saved.
        The output structure will mirror BIDS within this directory
        (e.g., `output_base_dir/sub-XYZ/ses-ABC/func/...`).
    left_surface_template : str
        A path to the left hemisphere GIFTI surface file.
        This file is crucial for `wb_command` smoothing.
    right_surface_template : str
        A path to the right hemisphere GIFTI surface file,
        similar to `left_surface_template`.
    selected_confounds : list[str], optional
        List of confound regressors to use for cleaning (e.g., ['trans_x', 'trans_y', 'rot_z']).
        If None, no confound regression will be performed. Defaults to None.
    smoothing_kernel_size : int, optional
        FWHM smoothing kernel size in millimeters. If None, no smoothing
        will be performed. Defaults to None.
    target_subject_ids : list[str], optional
        A list of specific subject IDs (e.g., ['sub-01', 'sub-02']) to process.
        If None, all subjects found will be processed.
    target_session_ids : list[str], optional
        A list of specific session IDs (e.g., ['ses-01']) to process.
        If None, all sessions found will be processed for selected subjects.
    target_task_ids : list[str], optional
        A list of specific task/run IDs (e.g., ['task-inamoment', 'task-coco_run-01']) to process.
        If None, all tasks found will be processed for selected subjects/sessions.
    """

    # Check for wb_command if smoothing is requested
    if smoothing_kernel_size is not None and smoothing_kernel_size > 0:
        _check_wb_command_exists() # This will raise an error if not found

    print(f"Starting fMRI BOLD data cleaning pipeline.")
    print(f"  Input BIDS fMRIPrep root: {bids_fmriprep_root_dir}")
    print(f"  Output base directory: {output_base_dir}")
    print(f"  Selected confounds: {selected_confounds if selected_confounds else 'None'}")
    print(f"  Smoothing kernel (FWHM): {smoothing_kernel_size if smoothing_kernel_size is not None else 'None'}mm")
    print(f"  Target Subjects: {target_subject_ids if target_subject_ids else 'All'}")
    print(f"  Target Sessions: {target_session_ids if target_session_ids else 'All'}")
    print(f"  Target Tasks: {target_task_ids if target_task_ids else 'All'}")
    print("-" * 80)

    # Find all BOLD files
    bold_files = glob.glob(os.path.join(bids_fmriprep_root_dir, 'sub-*', 'ses-*', 'func', '*_bold.dtseries.nii'))
    
    if not bold_files:
        raise FileNotFoundError(f"No *_bold.dtseries.nii files found in '{bids_fmriprep_root_dir}'. "
                                 "Ensure your data follows BIDS fMRIPrep output structure.")
    
    # Pre-check for confound files only if regression is requested
    confound_map = {}
    if selected_confounds:
        confound_files = glob.glob(os.path.join(bids_fmriprep_root_dir, 'sub-*', 'ses-*', 'func', '*_desc-confounds_timeseries.tsv'))
        if not confound_files:
             raise FileNotFoundError(f"No *_desc-confounds_timeseries.tsv files found in '{bids_fmriprep_root_dir}'. "
                                 "These are mandatory for confound regression when `selected_confounds` is provided.")
        # Create a mapping for quick lookup of confound files
        # Key: (subject_id, session_id, task_id_with_run)
        for c_file in confound_files:
            sub, ses, task = _parse_bids_filename(c_file)
            if sub and ses and task:
                confound_map[(sub, ses, task)] = c_file

    processed_runs_count = 0

    for bold_file_path in sorted(bold_files): # Sort for consistent processing order
        sub_id, ses_id, task_id = _parse_bids_filename(bold_file_path)

        # Apply filtering based on target_subject_ids, target_session_ids, target_task_ids
        if target_subject_ids and sub_id not in target_subject_ids:
            continue
        if target_session_ids and ses_id not in target_session_ids:
            continue
        if target_task_ids and task_id not in target_task_ids:
            continue

        # Find the corresponding confound file path if regression is active
        current_confound_file_path = None
        if selected_confounds:
            key = (sub_id, ses_id, task_id)
            if key not in confound_map:
                print(f"Warning: Corresponding confound file not found for {bold_file_path}. Skipping this run.")
                continue
            current_confound_file_path = confound_map[key]

        # Dynamically construct surface paths based on template
        if smoothing_kernel_size:
            if not os.path.exists(left_surface_template):
                print(f"Error: Left surface file not found: {left_surface_template}.")
                return
            if not os.path.exists(right_surface_template):
                print(f"Error: Right surface file not found: {right_surface_template}.")
                return

        # Process the individual functional run
        try:
            process_functional_run(
                bold_file_path=bold_file_path,
                confound_file_path=current_confound_file_path, # Will be None if no confounds selected
                output_base_dir=output_base_dir,
                left_surface_file=left_surface_template,
                right_surface_file=right_surface_template,
                selected_confounds=selected_confounds,
                smoothing_kernel_size=smoothing_kernel_size
            )
            processed_runs_count += 1
        except Exception as e:
            print(f"CRITICAL ERROR processing {bold_file_path}: {e}. This run was not completed.")
            continue # Continue to next file even if one fails

    print("\n" + "=" * 80)
    print(f"Cleaning pipeline finished.")
    print(f"Total functional runs processed: {processed_runs_count}")
    print("=" * 80)    
