from neursen_reveal.utils import *
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.utils import generate_leave_one_run_out
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from sklearn.pipeline import make_pipeline
from himalaya.viz import plot_alphas_diagnostic
from sklearn import set_config
from voxelwise_tutorials.utils import explainable_variance
import scipy.stats
import nilearn.plotting as plotting
import hcp_utils as hcp
import random
import torch


class VoxelWiseModeler:
    """
    A class to encapsulate the workflow for voxel-wise fMRI encoding modeling
    using Kernel Ridge Regression with delayed features.

    This class handles feature scaling, applying delays, cross-validated model
    fitting, scoring, and various visualization/reporting utilities.

    Attributes
    ----------
    delays : list
        List of time-step delays applied to features.
    alphas : np.ndarray
        Regularization strengths (alpha values) used in Kernel Ridge Regression.
    model_path : str
        File path where the trained model pipeline will be saved.
    backend : himalaya.backend.BaseBackend
        The computational backend (e.g., 'torch_cuda' for GPU, 'numpy' for CPU).
    feature_scaler : sklearn.preprocessing.StandardScaler
        Scaler for standardizing input features.
    delayer : voxelwise_tutorials.delayer.Delayer
        Delayer object to create time-lagged versions of the features.
    kernel_ridge_cv : himalaya.kernel_ridge.KernelRidgeCV
        Kernel Ridge Regression model with built-in cross-validation for alpha selection.
    pipeline : sklearn.pipeline.Pipeline or None
        The complete scikit-learn pipeline (scaler -> delayer -> KRCV) after fitting.
    """
    def __init__(
        self,
        delays: list = [1, 2, 3, 4],
        alphas: np.ndarray = np.logspace(1, 20, 20),
        backend_name: str = "torch_cuda",
        model_path: str = "voxel_model.joblib"
    ):
        """
        Initializes the VoxelWiseModeler with specified parameters.

        Parameters
        ----------
        delays : list, optional
            List of time-step delays (in TRs) to apply to features.
            E.g., `[1, 2, 3, 4]` means features from 1, 2, 3, and 4 TRs ago will be used.
            Defaults to `[1, 2, 3, 4]`.
        alphas : np.ndarray, optional
            Regularization strengths (hyperparameters) for Kernel Ridge Regression
            to be explored during cross-validation. These are typically on a log scale.
            Defaults to `np.logspace(1, 20, 20)`.
        backend_name : str, optional
            The computational backend to use. Options include 'torch_cuda' (for GPU),
            'numpy' (for CPU). Defaults to 'torch_cuda'.
        model_path : str, optional
            The file path where the trained model's pipeline will be saved
            using `joblib`. Defaults to "voxel_model.joblib".
        """
        self.delays = delays
        self.alphas = alphas
        self.model_path = model_path

        # Set backend (e.g., for GPU support)
        self.backend = set_backend(backend_name, on_error="warn")

        # Initialize components
        self.feature_scaler = StandardScaler()
        self.delayer = Delayer(delays=self.delays)
        self.kernel_ridge_cv = KernelRidgeCV(
            alphas=self.alphas,
            solver_params={
                'n_targets_batch': 500,
                'n_alphas_batch': 5,
                'n_targets_batch_refit': 100
            }
        )
        self.pipeline = None  # Will hold the trained pipeline


    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, train_run_onsets: list[int],
            random_state: int = None, n_runs_out: int = 1):
        """
        Fits the voxel-wise encoding model to the training data.

        The model pipeline includes feature scaling, time-lagging (delaying) features,
        and Kernel Ridge Regression with cross-validation for optimal alpha selection.
        Cross-validation is performed using a leave-one-run-out strategy.

        Parameters
        ----------
        X_train : np.ndarray
            Training features (stimuli embeddings or other predictors).
            Expected shape: `(n_timepoints_train, n_features_original)`.
        Y_train : np.ndarray
            Training BOLD fMRI signals.
            Expected shape: `(n_timepoints_train, n_voxels)`.
        train_run_onsets : list[int]
            A list of integers indicating the starting TR index for each run
            in the concatenated `X_train` and `Y_train` data. Used to define
            cross-validation folds (e.g., leave-one-run-out).
        random_state : int, optional
            Seed for random number generators to ensure reproducibility of
            cross-validation folds. Defaults to None (no fixed seed).
        n_runs_out : int, optional
            Number of runs to leave out for validation in each cross-validation fold.
            Defaults to 1 (Leave-One-Run-Out).
        """

        if random_state is not None: 
            np.random.seed(random_state)
            random.seed(random_state)
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

            
        X_train_selected = X_train.astype(np.float32)
        
        # Setup cross-validation
        n_samples_train = X_train_selected.shape[0]
        cv = generate_leave_one_run_out(n_samples_train, train_run_onsets, random_state = random_state, n_runs_out = n_runs_out)
        cv = check_cv(cv)

        # Create pipeline
        self.kernel_ridge_cv.cv = cv
        self.pipeline = make_pipeline(
            self.feature_scaler,
            self.delayer,
            self.kernel_ridge_cv
        )

        self.pipeline.fit(X_train_selected, Y_train)

    def score(self, X_test: np.ndarray, Y_test: np.ndarray, scores_file: str = None) -> np.ndarray:
        """
        Scores the trained model on test data and returns R-squared scores for each voxel.

        Parameters
        ----------
        X_test : np.ndarray
            Test features. Expected shape: `(n_timepoints_test, n_features_original)`.
        Y_test : np.ndarray
            Test BOLD fMRI signals. Expected shape: `(n_timepoints_test, n_voxels)`.
        scores_file : str, optional
            Optional file path to save the computed R-squared scores as a .npy file.
            Defaults to None (scores are not saved).

        Returns
        -------
        np.ndarray
            A NumPy array of R-squared scores for each voxel.
            Shape: `(n_voxels,)`.

        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before scoring. Call `fit()` first.")       
        X_test_selected = X_test.astype(np.float32)
        scores = self.pipeline.score(X_test_selected, Y_test)
        scores = self.backend.to_numpy(scores)
        if scores_file:
            np.save(scores_file, scores)
        return scores

    def plot_best_alphas_distribution(self):
        """
        Plots a histogram of the best alpha values (regularization strengths)
        chosen for each voxel during the cross-validation fitting process.

        This helps in diagnosing whether the chosen alpha range is appropriate
        and if regularization is consistently applied across voxels.

        Raises
        ------
        RuntimeError
            If the model has not been fitted (`fit` method not called previously).
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted first to plot best alphas. Call `fit()` first.")
        best_alphas = self.backend.to_numpy(self.pipeline[-1].best_alphas_)
        plot_alphas_diagnostic(best_alphas=best_alphas, alphas=self.alphas)


    def report_scores_stats(self, scores: np.ndarray):
        """
        Prints statistical summary of R-squared scores and displays their histogram.

        Parameters
        ----------
        scores : np.ndarray
            A NumPy array of R-squared scores for each voxel.
        """
        highly_correlated_voxels_indices = (scores > 0)
        print(f"Number of correlated voxels in (have R-squared scores > 0): {sum(highly_correlated_voxels_indices)}")
        print(f"The range of R-Squared scores [Min: {min(scores)}, Max: {max(scores)}]")
        plt.hist(scores, bins=np.linspace(0, 1, 100), log=True, histtype='step')
        plt.xlabel(f"$R^{2}$ Score")
        plt.ylabel("Number of voxels")
        plt.title(f'Histogram of $R^{2}$ scores')
        plt.grid('on')
        plt.show()

    def stack_runs(self, bold_runs_paths: list[str]) -> np.ndarray:
        """
        Loads and stacks multiple BOLD fMRI runs (trials) into a single 3D NumPy array.
        Each run is standardized independently before stacking.

        Parameters
        ----------
        bold_runs_paths : list[str]
            A list of full file paths to the .nii BOLD fMRI files for different runs
            of the same stimulus (e.g., repetitions of a test story).

        Returns
        -------
        np.ndarray
            A 3D NumPy array of stacked BOLD data.
            Shape: `(n_runs, n_timepoints_per_run, n_voxels)`.

        Raises
        ------
        ValueError
            If `bold_runs_paths` is empty.
        """
        if not bold_runs_paths:
            raise ValueError("`bold_runs_paths` cannot be an empty list.")
        first_bold = nib.load(bold_runs_paths[0]).get_fdata()
        TRs, voxels = first_bold.shape
        Y_test_trials = np.empty((len(bold_runs_paths), TRs, voxels))

        for i, trial in enumerate(bold_runs_paths):
            bold_signals = nib.load(trial).get_fdata()
            bold_signals = standardize(bold_signals)
            Y_test_trials[i] = bold_signals

        return Y_test_trials

    def compute_explainable_variance(self, stacked_runs: np.ndarray, ev_file: str = None) -> np.ndarray:
        """
        Computes the explainable variance (EV) for each voxel from repeated BOLD runs.
        EV quantifies the signal-to-noise ratio in the fMRI data.

        NaN values in the computed EV are replaced with 0.

        Parameters
        ----------
        stacked_runs : np.ndarray
            A 3D NumPy array of BOLD data from multiple repetitions of the same stimulus.
            Shape: `(n_repetitions, n_timepoints, n_voxels)`.
        ev_file : str, optional
            Optional file path to save the computed explainable variance as a .npy file.
            Defaults to None (EV is not saved).

        Returns
        -------
        np.ndarray
            A NumPy array of explainable variance values for each voxel.
            Shape: `(n_voxels,)`.
        """
        ev = explainable_variance(stacked_runs)
        if np.any(np.isnan(ev)):
            print("There are NaNs in the computed explainable variance!")
            ev = np.nan_to_num(ev, nan=0)
        if ev_file:
            np.save(ev_file, ev)
        return ev

    def plot_ev_distribution(self, ev: np.ndarray):
        """
        Plots a histogram of the computed explainable variance (EV) values across voxels.

        Parameters
        ----------
        ev : np.ndarray
            A NumPy array of explainable variance values for each voxel.
        """
        if not isinstance(ev, np.ndarray):
            raise TypeError("`ev` must be a NumPy array.")
        plt.hist(ev, bins=np.linspace(0, 1, 100), log=True, histtype='step')
        plt.xlabel("Explainable variance")
        plt.ylabel("Number of voxels")
        plt.title('Histogram of explainable variance')
        plt.grid('on')
        plt.show()

    def plot_voxel_with_max_ev(self, stacked_runs: np.ndarray, ev: np.ndarray, tr_duration: float = 2.0):
        """
        Plots the time-series of the voxel with the highest explainable variance.
        Displays individual repetitions and their average.

        Parameters
        ----------
        stacked_runs : np.ndarray
            A 3D NumPy array of BOLD data from multiple repetitions.
            Shape: `(n_repetitions, n_timepoints, n_voxels)`.
        ev : np.ndarray
            A NumPy array of explainable variance values for each voxel.
        tr_duration : float, optional
            Duration of one repetition time (TR) in seconds. Defaults to 2.0 seconds.
        """
        if not isinstance(stacked_runs, np.ndarray) or not isinstance(ev, np.ndarray):
            raise TypeError("`stacked_runs` and `ev` must be NumPy arrays.")
        if stacked_runs.shape[2] != len(ev):
            raise ValueError("Number of voxels in `stacked_runs` and `ev` must match.")
        if ev.size == 0:
            print("No EV values available to plot max voxel.")
            return
        max_voxel = np.argmax(ev) # voxel of max EV

        time = np.arange(stacked_runs.shape[1]) * 2  # one time point every 2 seconds (TR)
        plt.figure(figsize=(10, 3))
        plt.plot(time, stacked_runs[:, :, max_voxel].T, color='C0', alpha=0.5) # plot each repetition
        plt.plot(time, stacked_runs[:, :, max_voxel].mean(0), color='C1', label='average')

        plt.xlabel("Time (sec)")
        plt.title("Voxel with largest explainable variance (%.2f)" % ev[max_voxel])
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_voxel_with_min_ev(self, stacked_runs: np.ndarray, ev: np.ndarray, tr_duration: float = 2.0):
        """
        Plots the time-series of the voxel with the lowest explainable variance.
        Displays individual repetitions and their average.

        Parameters
        ----------
        stacked_runs : np.ndarray
            A 3D NumPy array of BOLD data from multiple repetitions.
            Shape: `(n_repetitions, n_timepoints, n_voxels)`.
        ev : np.ndarray
            A NumPy array of explainable variance values for each voxel.
        tr_duration : float, optional
            Duration of one repetition time (TR) in seconds. Defaults to 2.0 seconds.
        """
        if not isinstance(stacked_runs, np.ndarray) or not isinstance(ev, np.ndarray):
            raise TypeError("`stacked_runs` and `ev` must be NumPy arrays.")
        if stacked_runs.shape[2] != len(ev):
            raise ValueError("Number of voxels in `stacked_runs` and `ev` must match.")
        if ev.size == 0:
            print("No EV values available to plot min voxel.")
            return
        min_voxel = np.argmin(ev) # voxel with min EV
    
        time = np.arange(stacked_runs.shape[1]) * 2  # one time point every 2 seconds (TR)
        plt.figure(figsize=(10, 3))
        plt.plot(time, stacked_runs[:, :, min_voxel].T, color='C0', alpha=0.5) # plot each repetition
        plt.plot(time, stacked_runs[:, :, min_voxel].mean(0), color='C1', label='average')
        plt.xlabel("Time (sec)")
        plt.title("Voxel with lowest explainable variance (%.2f)" % ev[min_voxel])
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.show()

    

    def save_best_alphas(self, saving_path: str):
        """
        Saves the best alpha values chosen by cross-validation for each voxel
        and the full range of `alphas` used during fitting to disk.

        Parameters
        ----------
        saving_path : str
            The directory path where the 'best_alphas.npy' and 'alphas.npy'
            files will be saved.

        Raises
        ------
        RuntimeError
            If the model has not been fitted (`fit` method not called previously).
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted first before saving best alphas. Call `fit()` first.")
        best_alphas = self.backend.to_numpy(self.pipeline[-1].best_alphas_)
        os.makedirs(saving_path, exist_ok=True) # Ensure the directory exists
        
        try:
            np.save(os.path.join(saving_path, "best_alphas.npy"), best_alphas)
            np.save(os.path.join(saving_path, "alphas.npy"), self.alphas)
        except Exception as e:
            print(f"Failed to save best alphas: {e}")

    def save_pipeline(self):
        """
        Saves the entire trained scikit-learn pipeline (including the scaler,
        delayer, and KernelRidgeCV model) to the `model_path` specified
        during initialization using `joblib`.

        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before saving.")
        try:
            joblib.dump(self.pipeline, self.model_path)
            print(f"Model successfully saved to {self.model_path}")
        except Exception as e:
            print(f"Failed to save the model: {e}")

    def load_pipeline(self, model_path: str):
        """
        Loads a previously saved model pipeline from the specified path.

        Parameters
        ----------
        model_path : str
            The file path from which to load the saved model pipeline.

        Raises
        ------
        FileNotFoundError
            If no model file is found at the specified `model_path`.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
        try:
            self.pipeline = joblib.load(model_path)
            print(f"Model successfully loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load the model: {e}")
            self.pipeline = None


# -------------------------------------------------------------

def generate_scores_for_specific_model_across_subjects(
    Bold_dataset_dir: str,
    features_file_path: str,
    saving_dir: str,
    bold_test: list,
    save_EV: bool = False,
    verbose: bool = True,
    random_seed: int = None,
    delayers: list = [1, 2, 3, 4],
    n_runs_out: int = 1
):
  """
    Orchestrates the entire voxel-wise modeling process for multiple subjects,
    adhering to a BIDS-like directory structure (`sub-XX/ses-XX/func/.nii`).

    This function iterates through subjects, prepares their BOLD and feature data,
    trains a `VoxelWiseModeler`, computes and saves R-squared scores,
    and optionally computes and saves explainable variance (EV).

    Parameters
    ----------
    Bold_dataset_dir : str
        Path to the root directory containing subject BOLD data following BIDS.
        Expected structure: `Bold_dataset_dir/sub-XX/ses-XX/func/*.nii`.
    features_file_path : str
        Full path to the HDF5 (.h5) file containing extracted stimuli features.
    saving_dir : str
        Root directory where results (scores, best_alphas, EV) will be saved.
        Results will be organized by subject.
    bold_test : list, optional
        A list of string identifiers 
        that identify the BOLD signals to be used for testing.
    save_EV : bool, optional
        If True, compute and save explainable variance (EV) for test runs.
        Defaults to False.
    verbose : bool, optional
        If True, print detailed progress messages during execution. Defaults to True.
    random_seed : int, optional
        Seed for reproducibility of model fitting and cross-validation.
        Defaults to None (no fixed seed).
    delayers : list, optional
        List of time-step delays to use for the `VoxelWiseModeler`.
        Defaults to `[1, 2, 3, 4]`.
    n_runs_out : int, optional
        Number of runs to leave out for validation in each cross-validation fold
        during model fitting. Defaults to 1 (Leave-One-Run-Out).
    """
  subjects_list = [name for name in os.listdir(Bold_dataset_dir) 
                   if os.path.isdir(os.path.join(Bold_dataset_dir, name)) and name.lower().startswith('sub')]
  if not subjects_list:
        print(f"Error: No subject directories found in {Bold_dataset_dir} starting with 'sub'. Exiting.")
        return

    # Get global data properties from the first subject to ensure consistency
  try:
        sample_subject_base_dir = os.path.join(Bold_dataset_dir, subjects_list[0])
        no_voxels = get_voxels_count(sample_subject_base_dir) # Updated call
  except FileNotFoundError as e:
        print(f"Error: Could not determine voxel count from {sample_subject_base_dir}. {e}. Exiting.")
        return # Exit if essential info can't be obtained
    
  try:
        features_dim = get_feature_space_dim(features_file_path)
  except (FileNotFoundError, ValueError) as e:
        print(f"Error: Could not determine feature dimension from {features_file_path}. {e}. Exiting.")
        return

  if verbose:
    print(f"List of subjects: {subjects_list}")
    print(f"Features file: {features_file_path.split('/')[-1]}")
    print("-----------------------------------------")
  for subject in subjects_list:
    if verbose:
      print(f"Current subject: {subject}")
    subject_data_path = os.path.join(Bold_dataset_dir, subject)
    if not os.path.isdir(subject_data_path):
            print(f"Warning: Subject directory not found at {subject_data_path}. Skipping subject.")
            continue
    if verbose:
      print("----------------------Start Dataset Preparation----------------------")
    try:
        # prepare_modeling_data now takes subject_base_dir
        X_train, Y_train, X_test, Y_test, train_run_onsets = prepare_modeling_data(
                subject_base_dir=subject_data_path, 
                features_file_path=features_file_path,
                features_dim=features_dim,
                no_voxels=no_voxels,
                bold_test=bold_test
            )
    except Exception as e:
            print(f"Error preparing data for {subject}: {e}. Skipping subject.")
            continue
    if verbose:
      print(f"The target train shape (all_stories_TRs, no_voxels): {Y_train.shape}")
      print(f"The features train shape (all_stories_TRs, no_features): {X_train.shape}")
      print(f"the target test shape (last_story_TRs, no_voxels) {Y_test.shape}")
      print(f"the features test shape (last_story_TRs, no_features): {X_test.shape}")

    vm_model = VoxelWiseModeler(delays = delayers)
    if verbose:
      print("----------------------Start Model Fitting----------------------")
    vm_model.fit(X_train, Y_train, train_run_onsets, random_state=random_seed, n_runs_out=n_runs_out)
    if save_EV:
      if verbose:
        print("----------------------Computing Explainable Variance----------------------")
      sub_dir =  os.path.join(saving_dir, "EV")
      os.makedirs(sub_dir, exist_ok=True)
      saving_path = os.path.join(sub_dir, f"{subject}.npy")

      _, test_subset = train_test_split_of_bold_signals(
                subject_base_dir=subject_data_path, 
                bold_test=bold_test 
            )
      Y_test_trials = vm_model.stack_runs(test_subset)
      if Y_test_trials.shape[0] > 0:
        ev = vm_model.compute_explainable_variance(Y_test_trials, ev_file = saving_path)
        if verbose:
            vm_model.plot_ev_distribution(ev)
            if ev.size > 0:
                vm_model.plot_voxel_with_max_ev(Y_test_trials, ev)
                vm_model.plot_voxel_with_min_ev(Y_test_trials, ev)
      else:
            print(f"No valid BOLD data stacked for EV computation for {subject}.")

    if verbose:
      print("----------------------Start Model Scoring----------------------")
    scores_sub_dir =   os.path.join(saving_dir, subject)
    os.makedirs(scores_sub_dir, exist_ok=True)
    scores_path = os.path.join(scores_sub_dir, f"scores.npy")
    curr_scores = vm_model.score(X_test, Y_test, scores_file = scores_path)
    vm_model.save_best_alphas(scores_sub_dir)
    if verbose:
      print("----------------------Reporting Scores Stats----------------------")
      vm_model.report_scores_stats(curr_scores)
      
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
        if isinstance(threshold, str):
            percentile = int(threshold.replace("%", ''))
            threshold = np.percentile(data, percentile)
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



def render_averaged_scores(data_dir: str, threshold: str = "90%", title: str = "Averaged Rendered R-squared Scores"):
  """
    Calculates the average R-squared scores across all subjects for a given model
    and renders them onto an inflated brain surface.

    This function expects `data_dir` to contain subdirectories, each with a
    `scores.npy` file for a different subject.

    Parameters
    ----------
    data_dir : str
        The path to the directory containing subject-specific score subdirectories.
        Expected structure: `data_dir/sub-XX/scores.npy`.
    threshold : str or float, optional
        Threshold for displaying values on the surface map. Can be a percentage
        string (e.g., "90%") or a float (e.g., 0.1). Defaults to "90%".
    title : str, optional
        Title for the rendered brain plot. Defaults to "Averaged Rendered R-squared Scores".

    Returns
    -------
    nilearn.plotting.html_view.HTMLView
        An interactive HTML viewer object of the averaged brain surface plot.
        Returns None if rendering fails.

  """
  scores_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
  if not scores_dirs:
        raise FileNotFoundError(f"No subject score directories found in: {data_dir}")
  averaged_scores = None

  for score_dir in scores_dirs:
    curr_score = np.load(os.path.join(score_dir, "scores.npy"))
    if averaged_scores is None:
      averaged_scores = curr_score
    else:
      averaged_scores += curr_score

  averaged_scores /= len(scores_dirs)
  np.save(os.path.join(data_dir, f"{data_dir.split('/')[-1]}_averaged"), averaged_scores)

  return render_scores(averaged_scores, threshold = threshold, title = title) 


