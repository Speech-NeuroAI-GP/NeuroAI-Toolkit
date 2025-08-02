

# BSD 3-Clause License

# Copyright (c) 2020, the voxelwise tutorials developers All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.



# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class Delayer(TransformerMixin, BaseEstimator):
    """Scikit-learn Transformer to add delays to features.

    This assumes that the samples are ordered in time.
    Adding a delay of 0 corresponds to leaving the features unchanged.
    Adding a delay of 1 corresponds to using features from the previous sample.

    Adding multiple delays can be used to take into account the slow
    hemodynamic response, with for example `delays=[1, 2, 3, 4]`.

    Parameters
    ----------
    delays : array-like or None
        Indices of the delays applied to each feature. If multiple values are
        given, each feature is duplicated for each delay.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during the fit.

    Example
    -------
    >>> from sklearn.pipeline import make_pipeline
    >>> from voxelwise_tutorials.delayer import Delayer
    >>> from himalaya.kernel_ridge import KernelRidgeCV
    >>> pipeline = make_pipeline(Delayer(delays=[1, 2, 3, 4]), KernelRidgeCV())
    """

    def __init__(self, delays=None):
        self.delays = delays

    def fit(self, X, y=None):
        """Fit the delayer.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values. Ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        X = validate_data(self, X, dtype='numeric')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transform the input data X, copying features with different delays.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Xt : array of shape (n_samples, n_features * n_delays)
            Transformed data.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=True)

        n_samples, n_features = X.shape

        if self.delays is None:
            return X

        X_delayed = np.zeros((n_samples, n_features * len(self.delays)),
                             dtype=X.dtype)
        for idx, delay in enumerate(self.delays):
            beg, end = idx * n_features, (idx + 1) * n_features
            if delay == 0:
                X_delayed[:, beg:end] = X
            elif delay > 0:
                X_delayed[delay:, beg:end] = X[:-delay]
            elif delay < 0:
                X_delayed[:-abs(delay), beg:end] = X[abs(delay):]

        return X_delayed

    def reshape_by_delays(self, Xt, axis=1):
        """Reshape an array, splitting and stacking across delays.

        Parameters
        ----------
        Xt : array of shape (n_samples, n_features * n_delays)
            Transformed array.
        axis : int, default=1
            Axis to split.

        Returns
        -------
        Xt_split :array of shape (n_delays, n_samples, n_features)
            Reshaped array, splitting across delays.
        """
        delays = self.delays or [0]  # deals with None
        return np.stack(np.split(Xt, len(delays), axis=axis))
    
########################################utils ############################################
import numpy as np
import scipy.stats
from sklearn.utils.validation import check_random_state


def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):
    """Generate a leave-one-run-out split for cross-validation.

    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.
    random_state : None | int | instance of RandomState
        Random state for the shuffling operation.
    n_runs_out : int
        Number of runs to leave out in the validation set. Default to one.

    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
    """
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)
    # With permutations, we are sure that all runs are used as validation runs.
    # However here for n_runs_out > 1, a run can be chosen twice as validation
    # in the same split.
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val
def explainable_variance(data, bias_correction=True, do_zscore=True):
    """Compute explainable variance for a set of voxels.

    Parameters
    ----------
    data : array of shape (n_repeats, n_times, n_voxels)
        fMRI responses of the repeated test set.
    bias_correction: bool
        Perform bias correction based on the number of repetitions.
    do_zscore: bool
        z-score the data in time. Only set to False if your data time courses
        are already z-scored.

    Returns
    -------
    ev : array of shape (n_voxels, )
        Explainable variance per voxel.
    """
    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)

    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
    ev = var_mean / mean_var

    if bias_correction:
        n_repeats = data.shape[0]
        ev = ev - (1 - ev) / (n_repeats - 1)
    return ev

