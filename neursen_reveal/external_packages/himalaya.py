# BSD 3-Clause License

# Copyright (c) 2020, the himalaya developers All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.



# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
try:
    from scipy.sparse import issparse
except ImportError:

    def issparse(X):
        return False
try:
    from numpy.core.numeric import ComplexWarning
except ImportError:
    from numpy.exceptions import ComplexWarning
import warnings
import numpy as np

import types
import importlib
import warnings
from functools import wraps
from sklearn import get_config
import os
import numbers
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
import itertools
from sklearn.model_selection import check_cv as check_cv_sklearn
import sys
import time
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin

class ProgressBar():
    """Generate a command-line progress bar.

    Parameters
    ----------
    max_value : int
        Maximum value of process (e.g. number of samples to process, bytes to
        download, etc.).
    initial_value : int
        Initial value of process, useful when resuming process from a specific
        value, defaults to 0.
    title : str
        Message to include at end of progress bar.
    max_chars : int
        Number of characters to use for progress bar (be sure to save some room
        for the message and % complete as well).
    progress_character : char
        Character in the progress bar that indicates the portion completed.
    spinner : bool
        Show a spinner.  Useful for long-running processes that may not
        increment the progress bar very often.  This provides the user with
        feedback that the progress has not stalled.

    Examples
    --------
    >>> import time
    >>> from himalaya.progress_bar import ProgressBar
    >>> for ii in ProgressBar(title="La barre", max_value=10)(range(10)):
    >>>     time.sleep(0.5)
    """

    spinner_symbols = ['|', '/', '-', '\\']
    template = '\r[{0}{1}] {2:0.0f}% {3} {4:.02f} sec | {5} | '

    def __init__(self, title='', max_value=None, initial_value=0, max_chars=40,
                 progress_character='.', spinner=False, verbose_bool=True):
        self.cur_value = initial_value
        self.max_value = max_value
        self.title = title
        self.max_chars = max_chars
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)
        self._do_print = verbose_bool
        self.start = time.time()

        self.closed = False
        self.update(initial_value)

    def update(self, cur_value, title=None):
        """Update progressbar with current value of process.

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            (cur_value / max_value) * 100
        title : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        # Ensure floating-point division so we can get fractions of a percent
        # for the progressbar.
        self.cur_value = cur_value
        max_value = self.max_value or 1
        progress = min(float(self.cur_value) / max_value, 1.)
        num_chars = int(progress * self.max_chars)
        num_left = self.max_chars - num_chars

        # Update the message
        if title is not None:
            self.title = title

        # time from start
        duration = time.time() - self.start

        # The \r tells the cursor to return to the beginning of the line rather
        # than starting a new line.  This allows us to have a progressbar-style
        # display in the console window.
        bar = self.template.format(self.progress_character * num_chars,
                                   ' ' * num_left, progress * 100,
                                   self.spinner_symbols[self.spinner_index],
                                   duration, self.title)
        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        if self._do_print:
            sys.stdout.write(bar)
            sys.stdout.flush()
        # Increment the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

        if progress == 1:
            self.close()

    def update_with_increment_value(self, increment_value, title=None):
        """Update progressbar with the value of the increment instead of the
        current value of process as in update().

        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            (self.cur_value + increment_value / max_value) * 100
        title : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        self.cur_value += increment_value
        self.update(self.cur_value, title)

    def close(self):
        """Close the progress bar."""
        if not self.closed:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self.closed = True

    def __call__(self, sequence):
        sequence = iter(sequence)
        while True:
            try:
                yield next(sequence)
                self.update_with_increment_value(1)
            except StopIteration:
                return
def bar(iterable, title='', use_it=True):
    """Simple API for progress_bar.

    Parameters
    ----------
    iterable : iterable
        Iterable that will make the progress bar progress.
    title : str
        Message to include at end of progress bar.
    use_it : bool
        If False, return the iterable unchanged, and does not show a progress
        bar.

    Examples
    --------
    >>> import time
    >>> from himalaya.progress_bar import bar
    >>> for ii in bar(range(10)):
    >>>     time.sleep(0.5)
    """
    if use_it:
        return ProgressBar(title=title, max_value=len(iterable))(iterable)
    else:
        return iterable


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


ALL_BACKENDS = [
    "numpy",
    "cupy",
    "torch",
    "torch_cuda",
]

CURRENT_BACKEND = "numpy"

MATCHING_CPU_BACKEND = {
    "numpy": "numpy",
    "cupy": "numpy",
    "torch": "torch",
    "torch_cuda": "torch",
}
def set_backend(backend, on_error="raise"):
    """Set the backend using a global variable, and return the backend module.

    Parameters
    ----------
    backend : str or module
        Name or module of the backend.
    on_error : str in {"raise", "warn"}
        Define what is done if the backend fails to be loaded.
        If "warn", this function only warns, and keeps the previous backend.
        If "raise", this function raises on errors.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    global CURRENT_BACKEND

    try:
        if isinstance(backend, types.ModuleType):  # get name from module
            backend = backend.name

        if backend not in ALL_BACKENDS:
            raise ValueError("Unknown backend=%r" % (backend, ))

        module = importlib.import_module(__package__ + "." + backend)
        CURRENT_BACKEND = backend
    except Exception as error:
        if on_error == "raise":
            raise error
        elif on_error == "warn":
            warnings.warn(f"Setting backend to {backend} failed: {str(error)}."
                          f"Falling back to {CURRENT_BACKEND} backend.")
            module = get_backend()
        else:
            raise ValueError('Unknown value on_error=%r' % (on_error, ))

    return module

def get_backend():
    """Get the current backend module.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    module = importlib.import_module(__package__ + "." + CURRENT_BACKEND)
    return module
def _dtype_to_str(dtype):
    """Cast dtype to string, such as "float32", or "float64"."""
    if isinstance(dtype, str):
        return dtype
    elif hasattr(dtype, "name"):  # works for numpy and cupy
        return dtype.name
    elif "torch." in str(dtype):  # works for torch
        return str(dtype)[6:]
    elif dtype is None:
        return None
    else:
        raise NotImplementedError()
    
def force_cpu_backend(func):
    """Decorator to force the use of a CPU backend."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # skip if the object does not force cpu use
        if not hasattr(args[0], "force_cpu") or not args[0].force_cpu:
            return func(*args, **kwargs)

        # set corresponding cpu backend
        original_backend = get_backend().name
        temp_backend = MATCHING_CPU_BACKEND[original_backend]
        set_backend(temp_backend)

        # run function
        result = func(*args, **kwargs)

        # set back original backend
        set_backend(original_backend)
        return result

    return wrapper


    
def _get_string_dtype(array):
    """Get array's dtype, as a string.
    Returns None, if array has no attribute dtype.
    """
    dtype = getattr(array, "dtype", None)
    if dtype is None:
        return None

    return _dtype_to_str(dtype)

def _assert_all_finite(X, force_all_finite, numpy=False, batch_size=2 ** 24):
    """Check infinity and NaNs in X.

    Parameters
    ----------
    X : array
        Input data.
    force_all_finite : {True, False, 'allow-nan'}
        If True, raise an error for infinity and NaNs.
        If False, does not raise any error.
        If 'allow_nan', raise an error for infinity but not for NaNs.
    numpy : bool
        If True, use numpy for the check, else use the current backend.
    batch_size : int
        Batch size to avoid a full memory copy of X.

    Return
    ------
    None
    """
    if numpy:
        backend = np
    else:
        backend = get_backend()

    if get_config()['assume_finite']:
        return

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))
    if not force_all_finite:
        return

    X_dtype = _get_string_dtype(X)
    msg = ("Input contains NaN, infinity or a value too large for "
           "dtype(%r)." % X_dtype)

    for start in range(0, np.prod(X.shape), batch_size):
        batch = slice(start, start + batch_size)
        X_batch = X.flatten()[batch]

        if backend.any(backend.isinf(X_batch)):
            raise ValueError(msg)
        if force_all_finite != 'allow_nan' and backend.any(
                backend.isnan(X_batch)):
            raise ValueError(msg)

def check_array(array, accept_sparse=False, dtype=["float32", "float64"],
                copy=False, force_all_finite=True, ndim=2,
                ensure_min_samples=1, ensure_min_features=1, device="default"):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    dtype : str, list of str or None
        Data type of result. If None, the dtype of the input is preserved.
        If dtype is a list of str, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

    ndim : int, list of int, or None (default=2)
        If not None, list the accepted number of dimensions of the array.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    device : misc (default="default")
        Parameter passed to ``backend.asarray``. If "default", does not pass
        any parameter, to get the function's default. If "cpu", force the
        array to be in CPU memory.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    backend = get_backend()

    ########################
    # decide to change dtype

    dtype_input = _get_string_dtype(array)

    if dtype is None:
        dtype = None
    elif isinstance(dtype, str):
        if dtype not in ("float32", "float64"):
            raise ValueError("Unknown dtype=%r" % dtype)
    elif isinstance(dtype, (list, tuple)):
        if dtype_input in dtype:
            dtype = dtype_input
        else:
            dtype = dtype[0]
    else:
        raise ValueError("Unknown dtype=%r" % dtype)

    if not isinstance(device, str) or device != "default":
        kwargs = dict(device=device)
    else:
        kwargs = dict()

    if array is None:
        raise ValueError(
            "Expected array-like (array or non-string sequence), got None")

    #############
    # sparse case
    if issparse(array):
        changed_format = False

        if isinstance(accept_sparse, str):
            accept_sparse = [accept_sparse]
        if accept_sparse is False:
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required. Use X.toarray() to '
                            'convert to a dense numpy array.')
        elif isinstance(accept_sparse, (list, tuple)):
            if len(accept_sparse) == 0:
                raise ValueError("When providing 'accept_sparse' "
                                 "as a tuple or list, it must contain at "
                                 "least one string value.")
            # ensure correct sparse format
            if array.format not in accept_sparse:
                array = array.asformat(accept_sparse[0])
                changed_format = True
        elif accept_sparse is not True:
            raise ValueError("Parameter 'accept_sparse' should be a string, "
                             "boolean or list of strings. You provided "
                             "'accept_sparse={}'.".format(accept_sparse))

        if dtype != dtype_input:
            array = array.astype(dtype)
        elif copy and not changed_format:
            array = array.copy()

        _assert_all_finite(array.data, force_all_finite, numpy=True)

    ############
    # dense case
    else:

        if hasattr(array, 'dtype') and "complex" in str(array.dtype):
            raise ValueError("Complex data not supported.")
        # convert ComplexWarning into error
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                ################
                array = backend.asarray(array, dtype=dtype, **kwargs)
                ################
            except ComplexWarning:
                raise ValueError("Complex data not supported.")

        if copy:
            array = backend.copy(array)

        if ndim is not None and array.ndim not in np.atleast_1d(ndim):
            raise ValueError("Found array with ndim %d, expected %s. "
                             "Reshape your data or change the input." %
                             (array.ndim, ndim))

        # copy misaligned arrays, as it can lead to segmentation faults
        if hasattr(array, "data_ptr"):
            if array.data_ptr() % 8 != 0:
                array = backend.copy(array)
        if hasattr(array, "__array_interface__"):
            if array.__array_interface__['data'][0] % 8 != 0:
                array = backend.copy(array)

        numpy = backend.name == 'cupy' and device == "cpu"
        _assert_all_finite(array, force_all_finite, numpy=numpy)

    #####################
    # check minimum sizes
    if ensure_min_samples > 0:
        n_samples = array.shape[0]
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required." %
                (n_samples, tuple(array.shape), ensure_min_samples))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required." %
                (n_features, tuple(array.shape), ensure_min_features))

    return array
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float64 is returned.
    """
    X_dtype = _get_string_dtype(X)

    if Y is None:
        Y_dtype = X_dtype
    else:
        Y_dtype = _get_string_dtype(Y)

    if X_dtype == Y_dtype == "float32":
        dtype = "float32"
    else:
        dtype = "float64"

    return dtype

def check_pairwise_arrays(X, Y, precomputed=False, dtype=None,
                          accept_sparse='csr', force_all_finite=True):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_X, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_Y, n_features)

    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples_X, n_features)
        An array equal to X.

    Y : {array-like, sparse matrix}, shape (n_samples_Y, n_features)
        An array equal to Y if Y was not None.
        If Y was None, Y will be a pointer to X.

    """
    if dtype is None:
        dtype = _return_float_dtype(X, Y)

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                            force_all_finite=force_all_finite)
    else:
        X = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                        force_all_finite=force_all_finite)
        Y = check_array(Y, accept_sparse=accept_sparse, dtype=dtype,
                        force_all_finite=force_all_finite)

    if precomputed:
        pass
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" %
                         (X.shape[1], Y.shape[1]))

    return X, Y
class KernelCenterer(TransformerMixin, BaseEstimator):
    """Center a kernel matrix.

    Adapt sklearn.preprocessing.KernelCenterer to use other backends.

    Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
    function mapping x to a Hilbert space. KernelCenterer centers (i.e.,
    normalize to have zero mean) the data without explicitly computing phi(x).
    It is equivalent to centering phi(x) with
    sklearn.preprocessing.StandardScaler(with_std=False).

    Parameters
    ----------
    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    Attributes
    ----------
    K_fit_rows_ : array of shape (n_samples,)
        Average of each column of kernel matrix.

    K_fit_all_ : float
        Average of kernel matrix.

    Examples
    --------
    >>> from himalaya.kernel_ridge import KernelCenterer
    >>> from himalaya.kernel_ridge import pairwise_kernels
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> K = pairwise_kernels(X, metric='linear')
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelCenterer().fit(K)
    >>> transformer
    KernelCenterer()
    >>> transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    """

    def __init__(self, force_cpu=False):
        self.force_cpu = force_cpu

    @force_cpu_backend
    def fit(self, K, y=None):
        """Fit KernelCenterer

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        backend = get_backend()
        K = check_array(K, ndim=2)

        if K.shape[0] != K.shape[1]:
            raise ValueError("Kernel matrix must be a square matrix."
                             " Input is a {}x{} matrix.".format(
                                 K.shape[0], K.shape[1]))

        self.K_fit_rows_ = backend.mean_float64(K, axis=0)
        self.K_fit_all_ = backend.mean_float64(self.K_fit_rows_, axis=0)
        return self

    @force_cpu_backend
    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.

        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
        """
        check_is_fitted(self)
        backend = get_backend()
        K = check_array(K, ndim=2, copy=copy)

        K_pred_cols = backend.mean_float64(K, axis=1, keepdims=True)

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K

    def _more_tags(self):
        return {'pairwise': True}
def linear_kernel(X, Y=None):
    """
    Compute the linear kernel between X and Y.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)

    K = X @ Y.T
    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)
    return K


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y::

        K(X, Y) = (gamma <X, Y> + coef0)^degree

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.
    degree : int, default 3
        Degree of the polynomial.
    gamma : float, default None
        if None, defaults to 1.0 / n_features
    coef0 : float, default 1
        Intercept.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = X @ Y.T
    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)
    K *= gamma
    K += coef0
    K = backend.power(K, degree, out=K)
    return K


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """
    Compute the sigmoid kernel between X and Y::

        K(X, Y) = tanh(gamma <X, Y> + coef0)

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    coef0 : float, default 1
        Intercept.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = X @ Y.T
    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)
    K *= gamma
    K += coef0
    K = backend.tanh(K, out=K)
    return K


PAIRWISE_KERNEL_FUNCTIONS = {
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'sigmoid': sigmoid_kernel
}

###########################################kernelridgecv himalaya###################





class _BaseKernelRidge(ABC, MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Base class for kernel ridge estimators"""

    ALL_KERNELS = PAIRWISE_KERNEL_FUNCTIONS

    @property
    @classmethod
    @abstractmethod
    def ALL_SOLVERS(cls):
        ...

    def _call_solver(self, **direct_params):
        """Helper function common to all classes, merging solver parameters."""

        # use self.solver_ if it exists, otherwise use self.solver
        solver = getattr(self, "solver_", self.solver)

        if solver not in self.ALL_SOLVERS:
            raise ValueError("Unknown solver=%r." % solver)

        function = self.ALL_SOLVERS[solver]
        solver_params = self.solver_params or {}

        # check duplicated parameters
        intersection = set(direct_params.keys()).intersection(
            set(solver_params.keys()))
        if intersection:
            raise ValueError(
                'Parameters %s should not be given in solver_params, since '
                'they are either fixed or have a direct parameter in %s.' %
                (intersection, self.__class__.__name__))

        return function(**direct_params, **solver_params)

    def _more_tags(self):
        return {'requires_y': True}

def _pairwise_callable(X, Y, metric, force_all_finite=True, **params):
    """Handle the callable case for pairwise_{distances,kernels}.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y, force_all_finite=force_all_finite)
    out = backend.zeros_like(X, shape=(X.shape[0], Y.shape[0]), dtype='float')

    if X is Y:
        # Only calculate metric for upper triangle
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **params)

        # Make symmetric
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            x = X[i]
            out[i, i] = metric(x, x, **params)

    else:
        # Calculate all cells
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **params)

    return out

def pairwise_kernels(X, Y=None, metric="linear", n_jobs=None, **params):
    """Compute the kernel between arrays X and optional array Y.

    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for metric are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
        'laplacian', 'sigmoid', 'cosine']

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_samples_X) if metric == "precomputed", \
             (n_samples_X, n_features) otherwise
        Array of pairwise kernels between samples, or a feature array.

    Y : array of shape (n_samples_Y, n_features)
        A second feature array only if X metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from X as input and return the corresponding
        kernel value as a single number. This means that callables from
        :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
        matrices, not single samples. Use the string identifying the kernel
        instead.

    **params : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : array [n_samples_X, n_samples_X] or [n_samples_X, n_samples_Y]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.
    """
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X

    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **params)
    else:
        raise ValueError("Unknown metric=%r." % metric)

    return func(X, Y, **params)

def _check_finite(y_pred):
    backend = get_backend()

    is_nan = backend.isnan(y_pred)
    if backend.any(is_nan):
        warnings.warn('nan in y_pred.')
        y_pred[is_nan] = 0

    isinf = backend.isinf(y_pred)
    if backend.any(isinf):
        warnings.warn('inf in y_pred.')
        y_pred[isinf] = 0

    return y_pred

def l2_neg_loss(y_true, y_pred):
    """L2 negative loss, computed for multiple predictions.

    Parameters
    ----------
    y_true : array or Tensor of shape (n_samples, n_targets)
        Ground truth.
    y_pred : array or Tensor of shape (n_predictions, n_samples, n_targets) or\
            (n_samples, n_targets)
        Predictions.

    Returns
    -------
    l2 : array of shape (n_predictions, n_targets) or (n_targets, )
        L2 negative losses.
    """
    backend = get_backend()
    y_true, y_pred = backend.check_arrays(y_true, y_pred)
    y_pred = _check_finite(y_pred)

    # broadcasting if y_pred has more dimensions
    axis = 1 if len(y_pred.shape) > len(y_true.shape) else 0
    n_samples = y_true.shape[0]

    error = ((y_true - y_pred) ** 2).sum(axis)
    l2 = -error / n_samples

    return l2


def r2_score(y_true, y_pred):
    """R2 score, computed for multiple predictions (e.g. multiple alphas).

    Parameters
    ----------
    y_true : array or Tensor of shape (n_samples, n_targets)
        Ground truth.
    y_pred : array or Tensor of shape (n_predictions, n_samples, n_targets) or\
            (n_samples, n_targets)
        Predictions.

    Returns
    -------
    r2 : array of shape (n_predictions, n_targets) or (n_targets, )
        R2 scores.
    """
    backend = get_backend()
    y_true, y_pred = backend.check_arrays(y_true, y_pred)
    y_pred = _check_finite(y_pred)

    # broadcasting if y_pred has more dimensions
    axis = 1 if len(y_pred.shape) > len(y_true.shape) else 0

    error = ((y_true - y_pred) ** 2).sum(axis)
    var = ((y_true - y_true.mean(0)) ** 2).sum(0)
    r2 = 1. - error / var
    if axis == 0:
        r2[var == 0] = 0
    else:
        r2[:, var == 0] = 0

    return r2

def _batch_or_skip(array, batch, axis):
    """Apply a batch on given axis, or skip if the dimension is equal to 1."""
    skip = (array is None or isinstance(array, numbers.Number)
            or array.ndim == 0 or array.shape[axis] == 1)  # noqa
    if skip:
        return array
    else:
        # Not general but works with slices in `batch`.
        if axis == 0:
            return array[batch]
        elif axis == 1:
            return array[:, batch]
        else:
            raise NotImplementedError()
        
def solve_kernel_ridge_eigenvalues(K, Y, alpha=1., method="eigh",
                                   fit_intercept=False,
                                   negative_eigenvalues="zeros",
                                   n_targets_batch=None, random_state=None):
    """Solve kernel ridge regression using eigenvalues decomposition.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    Parameters
    ----------
    K : array of shape (n_samples, n_samples)
        Input kernel.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.
    fit_intercept : boolean
        Whether to fit an intercept. If False, K should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    negative_eigenvalues : str in {"nan", "error", "zeros"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
        - "error" raises an error.
        - "zeros" replaces them with zeros.
        - "nan" returns nans if the regularization does not compensate
        twice the smallest negative value, else it ignores the problem.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    random_state : int, or None
        Random generator seed. Not used.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1, )) * alpha

    K, Y, alpha = backend.check_arrays(K, Y, alpha)

    if K.shape[0] != Y.shape[0]:
        raise ValueError("K and Y must have the same number of samples.")
    if K.shape[0] != K.shape[1]:
        raise ValueError("Kernels must be square.")

    centerer, Y_offset = None, None
    if fit_intercept:
        centerer = KernelCenterer()
        K = centerer.fit_transform(K)
        Y_offset = Y.mean(0)
        Y = Y - Y_offset

    if method == "eigh":
        # diagonalization: K = V @ np.diag(eigenvalues) @ V.T
        eigenvalues, V = backend.eigh(K)
        # match SVD notations: K = U @ np.diag(eigenvalues) @ Vt
        U = V
        Vt = V.T
    elif method == "svd":
        # SVD: K = U @ np.diag(eigenvalues) @ Vt
        U, eigenvalues, Vt = backend.svd(K)
    else:
        raise ValueError("Unknown method=%r." % (method, ))

    inverse = 1 / (alpha[None] + eigenvalues[:, None])

    # negative eigenvalues can emerge from incorrect kernels, or from float32
    if eigenvalues[0] < 0:
        if negative_eigenvalues == "nan":
            if alpha < -eigenvalues[0] * 2:
                return backend.ones_like(Y) * backend.asarray(
                    backend.nan, dtype=Y.dtype)
            else:
                pass

        elif negative_eigenvalues == "zeros":
            eigenvalues[eigenvalues < 0] = 0

        elif negative_eigenvalues == "error":
            raise RuntimeError(
                "Negative eigenvalues. Make sure the kernel is positive "
                "semi-definite, increase the regularization alpha, or use"
                "another solver.")
        else:
            raise ValueError("Unknown negative_eigenvalues=%r." %
                             (negative_eigenvalues, ))

    n_samples, n_targets = Y.shape
    dual_weights = backend.zeros_like(K, shape=(n_samples, n_targets),
                                      device="cpu")
    if n_targets_batch is None:
        n_targets_batch = n_targets

    for start in range(0, n_targets, n_targets_batch):
        batch = slice(start, start + n_targets_batch)

        iUT = _batch_or_skip(inverse, batch, 1)[:, None, :] * U.T[:, :, None]
        iUT = backend.transpose(iUT, (2, 0, 1))
        # iUT.shape = (1 or n_targets_batch, n_samples, n_samples)
        # Vt.T.shape = (n_samples, n_samples)
        # Y.T.shape = (n_targets, n_samples)
        # weights_batch = Vt.T @ iUT @ Y.T (batching over n_targets)
        if Y.shape[0] < Y.shape[1]:
            weights_batch = ((Vt.T @ iUT) @ Y.T[batch, :, None])[:, :, 0].T
        else:
            weights_batch = Vt.T @ (iUT @ Y.T[batch, :, None])[:, :, 0].T

        dual_weights[:, batch] = backend.to_cpu(weights_batch)

    if fit_intercept:
        intercept = backend.to_cpu(Y_offset) - backend.to_cpu(
            centerer.K_fit_rows_) @ dual_weights
        return dual_weights, intercept
    else:
        return dual_weights

KERNEL_RIDGE_SOLVERS = {
    "eigenvalues": solve_kernel_ridge_eigenvalues,
}

def _to_cpu(X):
    backend = get_backend()
    if issparse(X):
        return X
    else:
        return backend.to_cpu(X)


class KernelRidge(_BaseKernelRidge):
    """Kernel ridge regression.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where K is a kernel computed on the input X.
    The default kernel is linear (K = X @ X.T).

    Parameters
    ----------
    alpha : float, or array of shape (n_targets, )
        L2 regularization parameter.

    kernel : str or callable, default="linear"
        Kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'sigmoid',  or 'precomputed'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernel_params : dict or None
        Additional parameters for the kernel function.
        See more details in the docstring of the function:
        ``KernelRidge.ALL_KERNELS[kernel]``

    solver : str
        Algorithm used during the fit, "eigenvalues", "conjugate_gradient",
        "gradient_descent", or "auto".
        If "auto", use "eigenvalues" if ``alpha`` is a float, and
        "conjugate_gradient" is ``alpha`` is an array.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``KernelRidge.ALL_SOLVERS[solver]``

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    warn : bool
        If True, warn if the number of samples is larger than the number of
        features, and if the kernel is linear.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    intercept_ : float or array of shape (n_targets, )
        Intercept. Only present if fit_intercept is True.

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernel == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> model = KernelRidge()
    >>> model.fit(X, Y)
    KernelRidge()
    """
    ALL_SOLVERS = KERNEL_RIDGE_SOLVERS

    def __init__(self, alpha=1, kernel="linear", kernel_params=None,
                 solver="auto", solver_params=None, fit_intercept=False,
                 force_cpu=False, warn=True):
        self.alpha = alpha
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params
        self.fit_intercept = fit_intercept
        self.force_cpu = force_cpu
        self.warn = warn

    @force_cpu_backend
    def fit(self, X, y=None, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape (n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : None, or array of shape (n_samples, )
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, accept_sparse=accept_sparse, ndim=2)
        self.dtype_ = _get_string_dtype(X)
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2])
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        n_samples, n_features = X.shape
        if n_samples > n_features and self.kernel == "linear" and self.warn:
            warnings.warn(
                "Solving linear kernel ridge is slower than solving ridge when"
                f" n_samples > n_features (here {n_samples} > {n_features}). "
                "Using himalaya.ridge.Ridge would be faster. Use warn=False to"
                " silence this warning.", UserWarning)

        K = self._get_kernel(X)

        self.X_fit_ = _to_cpu(X) if self.kernel != "precomputed" else None
        self.n_features_in_ = X.shape[1]
        del X

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        if sample_weight is not None:
            # We need to support sample_weight directly because K might be a
            # precomputed kernel.
            sw = backend.sqrt(sample_weight)[:, None]
            y = y * sw
            K *= sw @ sw.T

        # select solver based on the presence of multiple alphas
        if self.solver == "auto":
            if backend.atleast_1d(self.alpha).shape[0] == 1:
                self.solver_ = "eigenvalues"
            else:
                self.solver_ = "conjugate_gradient"
        else:
            self.solver_ = self.solver

        # ------------------ call the solver
        tmp = self._call_solver(K=K, Y=y, alpha=self.alpha,
                                fit_intercept=self.fit_intercept)
        if self.fit_intercept:
            self.dual_coef_, self.intercept_ = tmp
        else:
            self.dual_coef_ = tmp

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self

    @force_cpu_backend
    def predict(self, X):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernel == "precomputed" this is instead a precomputed
            kernel array of shape (n_samples_test, n_samples_train).

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        backend = get_backend()
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        K = self._get_kernel(X, self.X_fit_)
        del X

        Y_hat = backend.to_cpu(K) @ backend.to_cpu(self.dual_coef_)
        if self.fit_intercept:
            Y_hat += backend.to_cpu(self.intercept_)
        return Y_hat

    @force_cpu_backend
    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernel == "precomputed" this is instead a precomputed
            kernel array of shape (n_samples_test, n_samples_train).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : array of shape (n_targets, )
            R^2 of self.predict(X) versus y.
        """
        y_pred = self.predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)

        if y_true.ndim == 1:
            return r2_score(y_true[:, None], y_pred[:, None])[0]
        else:
            return r2_score(y_true, y_pred)

    def _get_kernel(self, X, Y=None):
        backend = get_backend()
        kernel_params = self.kernel_params or {}
        if Y is not None and not issparse(X):
            Y = backend.asarray_like(Y, ref=X)
        kernel = pairwise_kernels(X, Y, metric=self.kernel, **kernel_params)
        return backend.asarray(kernel)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def get_primal_coef(self, X_fit=None):
        """Returns the primal coefficients, assuming the kernel is linear.

        When the kernel is linear, kernel ridge regression is equivalent to
        ridge rergession, and the ridge regression (primal) coefficients can be
        computed from the kernel ridge regression (dual) coefficients, using
        the training features X_fit.

        Parameters
        ----------
        X_fit : array of shape (n_samples_fit, n_features) or None
            Training features. Used only if self.kernel == "precomputed".
            If you used a Kernelizer, you can use the method `get_X_fit`.

        Returns
        -------
        primal_coef : array of shape (n_features, n_targets)
            Coefficient of the equivalent ridge regression. The coefficients
            are computed on CPU memory, since they can be large.
        """
        check_is_fitted(self)
        backend = get_backend()

        if self.kernel == "linear":
            X_fit_T = self.X_fit_.T
        elif self.kernel == "precomputed":
            if X_fit is None:
                raise ValueError(
                    "get_primal_coef requires the training features `X_fit`. "
                    "If you used a Kernelizer, you can use the method "
                    "`get_X_fit`.")
            X_fit_T = X_fit.T
        else:
            raise ValueError("The primal coefficients can only be computed "
                             "when using a linear kernel.")

        X_fit_T = backend.to_cpu(X_fit_T)
        dual_coef = backend.to_cpu(self.dual_coef_)
        return X_fit_T @ dual_coef
    
def generate_dirichlet_samples(n_samples, n_kernels, concentration=[.1, 1.],
                               random_state=None):
    """Generate samples from a Dirichlet distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_kernels : int
        Number of dimension of the distribution.
    concentration : float, or list of float
        Concentration parameters of the Dirichlet distribution.
        A value of 1 corresponds to uniform sampling over the simplex.
        A value of infinity corresponds to equal weights.
        If a list, samples cycle through the list.
    random_state : int, or None
        Random generator seed. Use an int for deterministic samples.

    Returns
    -------
    gammas : array of shape (n_samples, n_kernels)
        Dirichlet samples.
    """
    random_generator = check_random_state(random_state)

    concentration = np.atleast_1d(concentration)
    n_concentrations = len(concentration)
    n_samples_per_concentration = int(
        np.ceil(n_samples / float(n_concentrations)))

    # generate the gammas
    gammas = []
    for conc in concentration:
        if conc == np.inf:
            gamma = np.full(n_kernels, fill_value=1. / n_kernels)[None]
            gamma = np.tile(gamma, (n_samples_per_concentration, 1))
        else:
            gamma = random_generator.dirichlet([conc] * n_kernels,
                                               n_samples_per_concentration)
        gammas.append(gamma)
    gammas = np.vstack(gammas)

    # reorder the gammas to alternate between concentrations:
    # [a0, a1, a2, a0, a1, a2] instead of [a0, a0, a1, a1, a2, a2]
    gammas = gammas.reshape(n_concentrations, n_samples_per_concentration,
                            n_kernels)
    gammas = np.swapaxes(gammas, 0, 1)
    gammas = gammas.reshape(n_concentrations * n_samples_per_concentration,
                            n_kernels)

    # remove extra gammas
    gammas = gammas[:n_samples]

    # cast to current backend
    backend = get_backend()
    gammas = backend.asarray(gammas)

    return gammas
def _decompose_kernel_ridge(Ktrain, alphas, Ktest=None, n_alphas_batch=None,
                            method="eigh", negative_eigenvalues="zeros"):
    """Precompute resolution matrices for kernel ridge predictions.

    To compute the prediction::

        Ytest_hat = Ktest @ (Ktrain + alphas * Id)^-1 @ Ytrain

    this function precomputes::

        matrices = Ktest @ (Ktrain + alphas * Id)^-1

    or just::

        matrices = (Ktrain + alphas * Id)^-1

    if Ktest is None.

    Parameters
    ----------
    Ktrain : array of shape (n_samples_train, n_samples_train)
        Training kernel for one feature space.
    alphas : float, or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    Ktest : array of shape (n_samples_test, n_samples_train)
        Testing kernel for one feature space.
    n_alphas_batch : int or None
        If not None, returns a generator over batches of alphas.
    method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.
    negative_eigenvalues : str in {"nan", "error", "zeros"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
            - "error" raises an error.
            - "zeros" replaces them with zeros.
            - "nan" returns nans if the regularization does not compensate
                twice the smallest negative value, else it ignores the problem.

    Returns
    -------
    matrices : array of shape (n_alphas, n_samples_train, n_samples_train) or \
        (n_alphas, n_samples_test, n_samples_train) if Ktest is not None
        Precomputed resolution matrices.
    alpha_batch : slice
        Slice of the batch of alphas.
    """
    backend = get_backend()

    use_alpha_batch = n_alphas_batch is not None
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    if method == "eigh":
        # diagonalization: K = V @ np.diag(eigenvalues) @ V.T
        eigenvalues, V = backend.eigh(Ktrain)
        # match SVD notations: K = U @ np.diag(eigenvalues) @ Vt
        U = V
        Vt = V.T
    elif method == "svd":
        # SVD: K = U @ np.diag(eigenvalues) @ Vt
        U, eigenvalues, Vt = backend.svd(Ktrain)
    else:
        raise ValueError("Unknown method=%r." % (method, ))

    if Ktest is not None:
        Ktest_V = backend.matmul(Ktest, Vt.T)

    for start in range(0, len(alphas), n_alphas_batch):
        batch = slice(start, start + n_alphas_batch)

        ev_weighting = (alphas[batch, None] + eigenvalues) ** -1

        # negative eigenvalues can emerge from incorrect kernels,
        # or from float32
        if eigenvalues[0] < 0:
            if negative_eigenvalues == "nan":
                ev_weighting[alphas[batch] < -eigenvalues[0] *
                             2, :] = backend.asarray(backend.nan,
                                                     dtype=ev_weighting.dtype)

            elif negative_eigenvalues == "zeros":
                eigenvalues[eigenvalues < 0] = 0

            elif negative_eigenvalues == "error":
                raise RuntimeError(
                    "Negative eigenvalues. Make sure the kernel is positive "
                    "semi-definite, increase the regularization alpha, or use"
                    "another solver.")
            else:
                raise ValueError("Unknown negative_eigenvalues=%r." %
                                 (negative_eigenvalues, ))

        if Ktest is not None:
            matrices = backend.matmul(Ktest_V, ev_weighting[:, :, None] * U.T)
        else:
            matrices = backend.matmul(Vt.T, ev_weighting[:, :, None] * U.T)

        if use_alpha_batch:
            yield matrices, batch
        else:
            return matrices, batch

        del matrices
def _helper_intercept(Ks, Y):
    """Transform Ks and Y if we fit an intercept."""
    backend = get_backend()

    # unfortunate, but we need to preserve the input. XXX add a copy parameter
    Ks = backend.copy(Ks)
    centerer = KernelCenterer()
    Ks_rows = backend.zeros_like(Ks, shape=(Ks.shape[0], Ks.shape[1]))
    for ii in range(Ks.shape[0]):
        Ks[ii] = centerer.fit_transform(Ks[ii])
        Ks_rows[ii] = centerer.K_fit_rows_

    Y = backend.copy(Y)
    Y_offset = Y.mean(0)
    Y -= Y_offset
    return Ks, Y, Ks_rows, Y_offset
def _select_best_alphas(scores, alphas, local_alpha, conservative):
    """Helper to select the best alphas

    Parameters
    ----------
    scores : array of shape (n_splits, n_alphas, n_targets)
        Cross validation scores.
    alphas :array of shape (n_alphas, )
        Regularization parameters.
    local_alpha : bool
        If True, alphas are selected per target, else shared over all targets.
    conservative : bool
        If True, when selecting the hyperparameter alpha, take the largest one
        that is less than one standard deviation away from the best.
        If False, take the best.

    Returns
    -------
    alphas_argmax : array of shape (n_targets, )
        Indices of the best alphas.
    best_scores_mean : array of shape (n_targets, )
        Scores, averaged over splits, and maximized over alphas.
    """
    backend = get_backend()

    # average scores over splits
    scores_mean = backend.mean_float64(scores, axis=0)
    # add epsilon slope to select larger alphas if scores are equal
    scores_mean += (backend.log(alphas) * 1e-10)[:, None]

    # compute the max over alphas
    axis = 0
    if local_alpha:
        alphas_argmax = backend.argmax(scores_mean, axis)

        if conservative:
            # take a conservative alpha, the largest that beats the best
            # score minus one standard deviation
            scores_std = scores.std(0)
            to_beat = backend.apply_argmax(scores_mean - scores_std,
                                           alphas_argmax, axis)
            does_beat = backend.asarray(scores_mean > to_beat, dtype="float32")
            # add bias toward large alphas (scaled to be << 1)
            does_beat += backend.log(alphas)[:, None] * 1e-4
            alphas_argmax = backend.argmax(does_beat, axis)
    else:
        if conservative:
            raise NotImplementedError()
        else:
            alphas_argmax = backend.argmax(scores_mean.mean(1))
            alphas_argmax = backend.full_like(alphas_argmax,
                                              shape=scores_mean.shape[1],
                                              fill_value=alphas_argmax)
    best_scores_mean = backend.apply_argmax(scores_mean, alphas_argmax, axis)

    return alphas_argmax, best_scores_mean


def solve_multiple_kernel_ridge_random_search(
        Ks, Y, n_iter=100, concentration=[0.1, 1.0], alphas=1.0,
        score_func=l2_neg_loss, cv=5, fit_intercept=False, return_weights=None,
        Xs=None, local_alpha=True, jitter_alphas=False, random_state=None,
        n_targets_batch=None, n_targets_batch_refit=None, n_alphas_batch=None,
        progress_bar=True, Ks_in_cpu=False, conservative=False, Y_in_cpu=False,
        diagonalize_method="eigh", return_alphas=False):
    """Solve multiple kernel ridge regression using random search.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels.
    Y : array of shape (n_samples, n_targets)
        Target data.
    n_iter : int, or array of shape (n_iter, n_kernels)
        Number of kernel weights combination to search.
        If an array is given, the solver uses it as the list of kernel weights
        to try, instead of sampling from a Dirichlet distribution. Examples:
          - `n_iter=np.eye(n_kernels)` implement a winner-take-all strategy
            over kernels.
          - `n_iter=np.ones((1, n_kernels))/n_kernels` solves a (standard)
            kernel ridge regression.
    concentration : float, or list of float
        Concentration parameters of the Dirichlet distribution.
        If a list, iteratively cycle through the list.
        Not used if n_iter is an array.
    alphas : float or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    score_func : callable
        Function used to compute the score of predictions versus Y.
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    fit_intercept : boolean
        Whether to fit an intercept. If False, Ks should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
        Only available if return_weights == 'dual'.
    return_weights : None, 'primal', or 'dual'
        Whether to refit on the entire dataset and return the weights.
    Xs : array of shape (n_kernels, n_samples, n_features) or None
        Necessary if return_weights == 'primal'.
    local_alpha : bool
        If True, alphas are selected per target, else shared over all targets.
    jitter_alphas : bool
        If True, alphas range is slightly jittered for each gamma.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    n_targets_batch_refit : int or None
        Size of the batch for over targets during refit.
        Used for memory reasons. If None, uses all n_targets at once.
    n_alphas_batch : int or None
        Size of the batch for over alphas. Used for memory reasons.
        If None, uses all n_alphas at once.
    progress_bar : bool
        If True, display a progress bar over gammas.
    Ks_in_cpu : bool
        If True, keep Ks in CPU memory to limit GPU memory (slower).
        This feature is not available through the scikit-learn API.
    conservative : bool
        If True, when selecting the hyperparameter alpha, take the largest one
        that is less than one standard deviation away from the best.
        If False, take the best.
    Y_in_cpu : bool
        If True, keep the target values ``Y`` in CPU memory (slower).
    diagonalize_method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.
    return_alphas : bool
        If True, return the best alpha value for each target.

    Returns
    -------
    deltas : array of shape (n_kernels, n_targets)
        Best log kernel weights for each target.
    refit_weights : array or None
        Refit regression weights on the entire dataset, using selected best
        hyperparameters. Refit weights are always stored on CPU memory.
        If return_weights == 'primal', shape is (n_features, n_targets),
        if return_weights == 'dual', shape is (n_samples, n_targets),
        else, None.
    cv_scores : array of shape (n_iter, n_targets)
        Cross-validation scores per iteration, averaged over splits, for the
        best alpha. Cross-validation scores will always be on CPU memory.
    best_alphas : array of shape (n_targets, )
        Best alpha value per target. Only returned if return_alphas is True.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    if isinstance(n_iter, int):
        gammas = generate_dirichlet_samples(n_samples=n_iter,
                                            n_kernels=len(Ks),
                                            concentration=concentration,
                                            random_state=random_state)
        gammas[0] = 1 / len(Ks)
    elif n_iter.ndim == 2:
        gammas = n_iter
        assert gammas.shape[1] == Ks.shape[0]
    else:
        raise ValueError("Unknown parameter n_iter=%r." % (n_iter, ))

    dtype = Ks.dtype
    gammas = backend.asarray(gammas, dtype=dtype)
    device = getattr(gammas, "device", None)
    Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)

    if isinstance(alphas, numbers.Number) or alphas.ndim == 0:
        alphas = backend.ones_like(Y, shape=(1, )) * alphas

    gammas, alphas, Xs = backend.check_arrays(gammas, alphas, Xs)
    Ks = backend.asarray(Ks, dtype=dtype,
                         device="cpu" if Ks_in_cpu else device)

    if fit_intercept:
        if return_weights == 'dual':
            Ks, Y, Ks_rows, Y_offset = _helper_intercept(Ks, Y)
        elif return_weights == 'primal':
            raise NotImplementedError(
                'Computing the intercept with return_weights="primal" is not'
                'implemented. Use fit_intercept=False or'
                'return_weights="dual".')
        else:
            raise ValueError(f'Cannot compute the intercept if return_weights'
                             f'is equal to {return_weights}.')

    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    cv = check_cv(cv, Y)
    n_splits = cv.get_n_splits()
    n_kernels = len(Ks)
    for train, val in cv.split(Y):
        if len(val) == 0 or len(train) == 0:
            raise ValueError("Empty train or validation set. "
                             "Check that `cv` is correctly defined.")

    if jitter_alphas:
        random_generator = check_random_state(random_state)
        given_alphas = backend.copy(alphas)

    best_gammas = backend.full_like(gammas, fill_value=1.0 / n_kernels,
                                    shape=(n_kernels, n_targets))
    best_alphas = backend.ones_like(gammas, shape=n_targets)
    cv_scores = backend.zeros_like(gammas, shape=(len(gammas), n_targets),
                                   device="cpu")
    current_best_scores = backend.full_like(gammas, fill_value=-backend.inf,
                                            shape=n_targets)

    # initialize refit ridge weights
    if return_weights == 'primal':
        if Xs is None:
            raise ValueError("Xs is needed to compute the primal weights.")
        n_features = sum(X.shape[1] for X in Xs)
        refit_weights = backend.zeros_like(gammas,
                                           shape=(n_features, n_targets),
                                           device="cpu")

    elif return_weights == 'dual':
        refit_weights = backend.zeros_like(gammas,
                                           shape=(n_samples, n_targets),
                                           device="cpu")
    elif return_weights is None:
        refit_weights = None
    else:
        raise ValueError("Unknown parameter return_weights=%r." %
                         (return_weights, ))

    # Possibility to entirely skip the fit and return arbitrary weights.
    # This is to help designing integration smoke tests.
    if os.environ.get('HIMALAYA_SKIP_FIT', default=False):
        warnings.warn("Skip fit because HIMALAYA_SKIP_FIT=True.")
        # skip the loop by emptying the gammas candidates
        gammas = gammas[:0]
        # fill with fake weights, to avoid getting only zeros.
        refit_weights += (backend.arange(n_targets)[None] + 1) / n_targets

    ###########################################################################
    # Main loop over hyperparameter candidates (gammas)
    for ii, gamma in enumerate(
            bar(gammas, '%d random sampling with cv' % len(gammas),
                use_it=progress_bar)):

        if Ks_in_cpu:
            K = (backend.to_cpu(gamma[:, None, None]) * Ks).sum(0)
            K = backend.to_gpu(K, device=device)
        else:
            K = (gamma[:, None, None] * Ks).sum(0)

        if jitter_alphas:
            noise = backend.asarray_like(random_generator.rand(), alphas)
            alphas = given_alphas * (10 ** (noise - 0.5))

        scores = backend.zeros_like(gammas,
                                    shape=(n_splits, len(alphas), n_targets))
        for jj, (train, test) in enumerate(cv.split(K)):
            train = backend.to_gpu(train, device=device)
            test = backend.to_gpu(test, device=device)
            Ktrain, Ktest = K[train[:, None], train], K[test[:, None], train]
            if fit_intercept:
                centerer = KernelCenterer()
                Ktrain = centerer.fit_transform(Ktrain)
                Ktest = centerer.transform(Ktest)

            for matrix, alpha_batch in _decompose_kernel_ridge(
                    Ktrain=Ktrain, alphas=alphas, Ktest=Ktest,
                    negative_eigenvalues="nan", n_alphas_batch=n_alphas_batch,
                    method=diagonalize_method):
                # n_alphas_batch, n_samples_test, n_samples_train = \
                # matrix.shape

                for start in range(0, n_targets, n_targets_batch):
                    batch = slice(start, start + n_targets_batch)
                    Ybatch = backend.to_gpu(Y[:, batch], device=device)
                    Ytrain, Ytest = Ybatch[train], Ybatch[test]
                    del Ybatch
                    if fit_intercept:
                        Ytrain_mean = Ytrain.mean(0)
                        Ytrain = Ytrain - Ytrain_mean
                        Ytest = Ytest - Ytrain_mean

                    predictions = backend.matmul(matrix, Ytrain)
                    # n_alphas_batch, n_samples_test, n_targets_batch = \
                    # predictions.shape

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        scores[jj, alpha_batch, batch] = score_func(
                            Ytest, predictions)
                        # n_alphas_batch, n_targets_batch = score.shape

                # make small alphas impossible to select
                too_small_alphas = backend.isnan(matrix[:, 0, 0])
                scores[jj, alpha_batch, :][too_small_alphas] = -1e5

                del matrix, predictions
            del train, test

        # select best alphas
        alphas_argmax, cv_scores_ii = _select_best_alphas(
            scores, alphas, local_alpha, conservative)
        cv_scores[ii, :] = backend.to_cpu(cv_scores_ii)

        # update best_gammas and best_alphas
        epsilon = np.finfo(_dtype_to_str(dtype)).eps
        if local_alpha:
            mask = cv_scores_ii > current_best_scores + epsilon
        else:
            # update based on score across all targets
            update = cv_scores_ii.mean() > current_best_scores.mean() + epsilon
            mask = backend.full_like(cv_scores_ii, fill_value=update,
                                     dtype=bool)
        current_best_scores[mask] = cv_scores_ii[mask]
        best_gammas[:, mask] = gamma[:, None]
        best_alphas[mask] = alphas[alphas_argmax[mask]]

        # compute primal or dual weights on the entire dataset (nocv)
        if return_weights is not None:
            update_indices = backend.flatnonzero(mask)
            if Y_in_cpu:
                update_indices = backend.to_cpu(update_indices)
            if len(update_indices) > 0:

                # refit weights only for alphas used by at least one target
                used_alphas = backend.unique(best_alphas[mask])
                dual_weights = backend.zeros_like(
                    K, shape=(n_samples, len(update_indices)), device="cpu")
                for matrix, alpha_batch in _decompose_kernel_ridge(
                        K, used_alphas, Ktest=None,
                        negative_eigenvalues="zeros",
                        n_alphas_batch=min(len(used_alphas), n_alphas_batch),
                        method=diagonalize_method):

                    for start in range(0, len(update_indices),
                                       n_targets_batch_refit):
                        batch = slice(start, start + n_targets_batch_refit)

                        weights = backend.matmul(
                            matrix,
                            backend.to_gpu(Y[:, update_indices[batch]],
                                           device=device))
                        # used_n_alphas_batch, n_samples, n_targets_batch = \
                        # weights.shape

                        # select alphas corresponding to best cv_score
                        alphas_indices = backend.searchsorted(
                            used_alphas, best_alphas[mask][batch])
                        # mask targets whose selected alphas are outside the
                        # alpha batch
                        mask2 = backend.isin(
                            alphas_indices,
                            backend.arange(len(used_alphas))[alpha_batch])
                        # get indices in alpha_batch
                        alphas_indices = backend.searchsorted(
                            backend.arange(len(used_alphas))[alpha_batch],
                            alphas_indices[mask2])
                        # update corresponding weights
                        mask_target = backend.arange(weights.shape[2])
                        mask_target = backend.to_gpu(mask_target)[mask2]
                        tmp = weights[alphas_indices, :, mask_target]
                        dual_weights[:, batch][:, backend.to_cpu(mask2)] = \
                            backend.to_cpu(tmp).T
                        del weights, alphas_indices, mask2, mask_target
                    del matrix

                if return_weights == 'primal':
                    # multiply by g and not np.sqrt(g), as we then want to use
                    # the primal weights on the unscaled features Xs, and not
                    # on the scaled features (np.sqrt(g) * Xs)
                    X = backend.concatenate([t * g for t, g in zip(Xs, gamma)],
                                            1)
                    primal_weights = backend.to_cpu(X.T) @ dual_weights
                    refit_weights[:, backend.to_cpu(mask)] = primal_weights
                    del X, primal_weights

                elif return_weights == 'dual':
                    refit_weights[:, backend.to_cpu(mask)] = dual_weights

                del dual_weights
            del update_indices
        del K, mask
    # End of main loop
    ###########################################################################

    deltas = backend.log(best_gammas / best_alphas[None, :])
    if return_weights == 'dual':
        refit_weights *= backend.to_cpu(best_alphas)

    if fit_intercept:
        if return_weights == 'dual':
            intercept = backend.to_cpu(Y_offset)
            intercept -= ((backend.to_cpu(Ks_rows) @ refit_weights) *
                          backend.to_cpu(backend.exp(deltas))).sum(0)
        else:
            return NotImplementedError()

    results = [deltas, refit_weights, cv_scores]
    if return_alphas:
        results.append(best_alphas)
    if fit_intercept:
        results.append(intercept)
    return results
def check_cv(cv, y):
    """Check the cross-validation generator object.

    In particular, check if indices in cv exceed number of samples, because
    some backends do not check for out-of-bound errors.

    Parameters
    ----------
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    y : array of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    cv : scikit-learn splitter
        Cross-validation splitter.
    """
    # first call check_cv from sklearn
    cv = check_cv_sklearn(cv)

    for train, test in cv.split(y):
        if max(train) >= y.shape[0] or max(test) >= y.shape[0]:
            # some backends do not check out of bound indices
            raise ValueError("Indices in cv exceed number of samples.")
    return cv
def solve_kernel_ridge_cv_eigenvalues(
        K, Y, alphas=1.0, score_func=l2_neg_loss, cv=5, fit_intercept=False,
        local_alpha=True, n_targets_batch=None, n_targets_batch_refit=None,
        n_alphas_batch=None, conservative=False, Y_in_cpu=False,
        diagonalize_method="eigh"):
    """Solve kernel ridge regression with a grid search over alphas.

    Parameters
    ----------
    K : array of shape (n_samples, n_samples)
        Input kernel.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alphas : float or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    score_func : callable
        Function used to compute the score of predictions versus Y.
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    fit_intercept : boolean
        Whether to fit an intercept. If False, K should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    local_alpha : bool
        If True, alphas are selected per target, else shared over all targets.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    n_targets_batch_refit : int or None
        Size of the batch for over targets during refit.
        Used for memory reasons. If None, uses all n_targets at once.
    n_alphas_batch : int or None
        Size of the batch for over alphas. Used for memory reasons.
        If None, uses all n_alphas at once.
    conservative : bool
        If True, when selecting the hyperparameter alpha, take the largest one
        that is less than one standard deviation away from the best.
        If False, take the best.
    Y_in_cpu : bool
        If True, keep the target values ``Y`` in CPU memory (slower).
    diagonalize_method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.

    Returns
    -------
    best_alphas : array of shape (n_targets, )
        Selected best hyperparameter alphas.
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients refit on the entire dataset, using selected
        best hyperparameters alpha. Always stored on CPU memory.
    cv_scores : array of shape (n_targets, )
        Cross-validation scores averaged over splits, for the best alpha.
    """
    backend = get_backend()

    n_iter = backend.ones_like(K, shape=(1, 1))
    fixed_params = dict(return_weights="dual", Xs=None, progress_bar=False,
                        concentration=None, jitter_alphas=False,
                        random_state=None, n_iter=n_iter)

    copied_params = dict(alphas=alphas, score_func=score_func, cv=cv,
                         local_alpha=local_alpha, fit_intercept=fit_intercept,
                         n_targets_batch=n_targets_batch,
                         n_targets_batch_refit=n_targets_batch_refit,
                         n_alphas_batch=n_alphas_batch,
                         conservative=conservative, Y_in_cpu=Y_in_cpu,
                         diagonalize_method=diagonalize_method)

    tmp = solve_multiple_kernel_ridge_random_search(K[None], Y,
                                                    **copied_params,
                                                    **fixed_params)
    if fit_intercept:
        deltas, dual_weights, cv_scores, intercept = tmp
        best_alphas = backend.exp(-deltas[0])
        dual_weights /= backend.to_cpu(best_alphas)
        return best_alphas, dual_weights, cv_scores, intercept
    else:
        deltas, dual_weights, cv_scores = tmp
        best_alphas = backend.exp(-deltas[0])
        dual_weights /= backend.to_cpu(best_alphas)
        return best_alphas, dual_weights, cv_scores

KERNEL_RIDGE_CV_SOLVERS = dict(eigenvalues=solve_kernel_ridge_cv_eigenvalues,
                        )

class KernelRidgeCV(KernelRidge):
    """Kernel ridge regression with efficient cross-validation over alpha.

    Parameters
    ----------
    alphas : array of shape (n_alphas, )
        List of L2 regularization parameter to try.

    kernel : str or callable, default="linear"
        Kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'sigmoid', or 'precomputed'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernel_params : dict or None
        Additional parameters for the kernel function.
        See more details in the docstring of the function:
        ``KernelRidgeCV.ALL_KERNELS[kernel]``

    solver : str, "eigenvalues"

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``KernelRidgeCV.ALL_SOLVERS[solver]``

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Y_in_cpu : bool
        If True, keep the target values ``y`` in CPU memory (slower).

    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    warn : bool
        If True, warn if the number of samples is larger than the number of
        features, and if the kernel is linear.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    best_alphas_ : array of shape (n_targets, )
        Selected best hyperparameter alphas.

    cv_scores_ : array of shape (n_targets, )
        Cross-validation scores averaged over splits, for the best alpha.
        By default, the scores are computed with l2_neg_loss (in ]-inf, 0]).
        The scoring function can be changed with solver_params["score_func"].

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernel == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    Examples
    --------
    >>> from himalaya.ridge import KernelRidgeCV
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> clf = KernelRidgeCV()
    >>> clf.fit(X, Y)
    KernelRidgeCV()
    """
    ALL_SOLVERS = KERNEL_RIDGE_CV_SOLVERS

    def __init__(self, alphas=[0.1, 1], kernel="linear", kernel_params=None,
                 solver="eigenvalues", solver_params=None, fit_intercept=False,
                 cv=5, Y_in_cpu=False, force_cpu=False, warn=True):
        self.alphas = alphas
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.Y_in_cpu = Y_in_cpu
        self.force_cpu = force_cpu
        self.warn = warn

    @force_cpu_backend
    def fit(self, X, y=None, sample_weight=None):
        """Fit kernel ridge regression model

        Parameters
        ----------
        X : array of shape (n_samples, n_features).
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape (n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : None, or array of shape (n_samples, )
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()
        X = check_array(X, accept_sparse=("csr", "csc"), ndim=2)
        self.dtype_ = _get_string_dtype(X)
        device = "cpu" if self.Y_in_cpu else None
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        alphas = check_array(self.alphas, dtype=self.dtype_, ndim=1)

        n_samples, n_features = X.shape
        if n_samples > n_features and self.kernel == "linear" and self.warn:
            warnings.warn(
                "Solving linear kernel ridge is slower than solving ridge when"
                f" n_samples > n_features (here {n_samples} > {n_features}). "
                "Using himalaya.ridge.RidgeCV would be faster. "
                "Use warn=False to silence this warning.", UserWarning)

        K = self._get_kernel(X)

        self.X_fit_ = _to_cpu(X) if self.kernel != "precomputed" else None
        self.n_features_in_ = X.shape[1]
        del X

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        if sample_weight is not None:
            # We need to support sample_weight directly because K might be a
            # pre-computed kernel.
            sw = backend.sqrt(sample_weight)[:, None]
            y = y * backend.to_cpu(sw) if self.Y_in_cpu else y * sw
            K *= sw @ sw.T

        cv = check_cv(self.cv, y)

        # ------------------ call the solver
        tmp = self._call_solver(K=K, Y=y, cv=cv, alphas=alphas,
                                Y_in_cpu=self.Y_in_cpu,
                                fit_intercept=self.fit_intercept)
        if self.fit_intercept:
            self.best_alphas_, self.dual_coef_, self.cv_scores_ = tmp[:3]
            self.intercept_, = tmp[3:]
        else:
            self.best_alphas_, self.dual_coef_, self.cv_scores_ = tmp
        self.cv_scores_ = self.cv_scores_[0]

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_sample_weights_invariance':
                'zero sample_weight is not equivalent to removing samples, '
                'because of the cross-validation splits.',
            }
        }
        



###############################################################################
###############################################################################
###############################################################################
###############################################################################



