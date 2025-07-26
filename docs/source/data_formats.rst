.. _data_formats:

Data Preparation: Understanding Expected Formats
=================================================

`neursen_reveal` is designed to work efficiently with neuroimaging data, particularly fMRI BOLD signals and AI model features. For optimal compatibility and ease of use, we strongly recommend organizing your fMRI data in accordance with the **Brain Imaging Data Structure (BIDS)** standard.

fMRI BOLD Data (CIFTI Files)
-----------------------------

**Recommended Structure (BIDS-like):** Your fMRI BOLD data should be organized in a subject-centric hierarchy. Each subject folder should contain `ses-` subdirectories each with a `func` subdirectory, which holds CIFTI time series files (`.dtseries.nii`) and associated confound/event files for each functional run.

.. code-block::
   :force:

  data/
    ├── sub-01/
    │   ├── ses-01
    |   |    ├── func/
    |   │    │   ├── sub-01_ses-01_task-1_run-01_bold.dtseries.nii
    |   |    |   ├── sub-01_ses-01_task-1_run-01_events.tsv # Mandatory for stimulus alignment
    |   │    │   ├── sub-01_ses-01_task-2_run-01_bold.dtseries.nii
    |   │    │   └── ...
    |   ├── ses-02
    |   └── ...
    |     
    ├── sub-02/
    │   └── ses-01/
    │       └── ...
    └── ...

**Naming Convention:** Functional CIFTI files are expected to follow the BIDS naming scheme: `sub-<label>_ses-<label>_task-<label>[_run-<index>]_bold.dtseries.nii`. The `task-<label>` is crucial as it links to your stimulus set.

AI Model Feature Files
-----------------------

**Input for Feature Extraction Module (requiring `_events.tsv`):**

If you utilize `neursen_reveal`'s integrated feature extraction module (`neursen_reveal.feature_extraction`), then **for every BOLD run you wish to process, a corresponding `_events.tsv` file is mandatory.** This file defines the timing and identity of stimuli or events presented during that run, enabling accurate alignment of stimuli with fMRI TRs.

Specifically, the `_events.tsv` should contain at least the following columns:

* **`onset`**: The onset time of the stimulus/event in seconds, relative to the start of the functional run.
* **`duration`**: The duration of the stimulus/event in seconds.
* **`stim_file`**: The path to the specific stimulus file, relative to the `stimuli/` root directory (see section below). The `task-<label>` in your BOLD file name should match the base name of the `stim_file`.

Examples of `_events.tsv` content:

.. code-block:: tsv

   onset   duration    stim_file
   0.0       450.0     stories/story_name.wav

**Stimuli Data Structure (for Feature Extraction Module):**

The `neursen_reveal.feature_extraction` module expects a path to your stimulus data.

**Naming Convention:** Individual stimulus files should be named and located logically withon the `stimuli_base_dir` such that the full *relative path* from `stimuli_base_dir` to the file must precisely match the entries in the `stim_file` column of your `_events.tsv` files.

.. code-block::
   :force:

   stimuli/
   ├── object_recognition/
   │   ├── car_image.png
   │   ├── dog_image.png
   │   └── ...
   └── ...

After processing, the feature extraction module will generate a single `.h5` file for your entire dataset based on your specified AI model, typically with keys corresponding to the unique identifiers for the functional runs (e.g., `task_label_run-index`).

Expected Format for Pre-computed Features (`.h5` files):
---------------------------------------------------------

If you are providing your own pre-computed features (i.e., **not** using `neursen_reveal.feature_extraction`), these features **must be provided as a single HDF5 (`.h5`) file**. In this scenario, `_events.tsv` files are not directly consumed by `neursen_reveal` as you are providing the already aligned features.

**Structure of the `.h5` file:**
* **Keys:** Each key in the `.h5` file should correspond to a **unique identifier for a functional run** (e.g., `'story_A_run-01'`). This is critical for correctly aligning AI features with the BOLD brain data during voxel-wise modeling.
* **Values:** The value associated with each key will be a NumPy array representing the AI model features for all stimuli presented within that specific functional run, already aligned to the run's temporal progression. The shape of this array will be (`number_of_TRs_in_run`, `AI_model_feature_dimension`). This means that for each TR in a given functional run, there will be a corresponding row of AI features.

