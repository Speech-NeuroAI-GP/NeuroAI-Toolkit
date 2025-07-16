# 🧠 NeuroAI-Toolkit

![NeuroAI Toolkit Logo/Banner - Placeholder Image](https://raw.githubusercontent.com/your-username/NeuroAI-Toolkit/main/assets/neuroai_banner.png) 


---

## Unlocking Brain-AI Connections: A Comprehensive Toolkit for Neuroimaging-AI Research

The `NeuroAI-Toolkit` is a powerful Python package designed to bridge the gap between cutting-edge Artificial Intelligence models and intricate neuroimaging data. Tailored for researchers exploring the neural underpinnings of perception and cognition, this toolkit provides a streamlined pipeline for advanced analysis, from feature extraction to brain representation similarity and modeling.

Whether you're investigating how the brain processes speech, language, or visual stimuli, `NeuroAI-Toolkit` offers a robust and flexible suite of tools to accelerate your research and uncover meaningful insights.

---

## ✨ Features at a Glance

`NeuroAI-Toolkit` is meticulously designed to support a wide range of analytical needs in NeuroAI research, offering modules for:

* **Deep Feature Extraction:**
    * Effortlessly extract rich, high-dimensional features from state-of-the-art AI models across multiple modalities.
    * **Supported Models:**
        * **Speech:** OpenAI Whisper Encoder (all checkpoints), ECAPA-TDNN
        * **Language:** GPT (various versions via Hugging Face Transformers)
         * **Language:** OpenAI Whisper Decoder (all checkpoints)
        * **Vision:** ResNet50
         * **Vision-Language interaction:** CLIP (all checkpoints)
    * Generates standardized `.h5` feature files for seamless integration into your analyses.

* **fMRI CIFTI Data Cleaning:**
    * Robust preprocessing functionalities specifically optimized for CIFTI (Connectivity Informatics Technology Initiative) brain data.
    * Ensures high-quality, artifact-minimized fMRI signals for downstream analysis.

* **Voxelwise Encoding Models:**
    * Build and evaluate powerful voxelwise models to predict brain activity from AI features.
    * Facilitates granular analysis of representational mappings across the brain.

* **Brain Parcellation & Score Mapping:**
    * Tools to parcellate and visualize brain scores derived from voxelwise modeling, allowing for region-of-interest (ROI) level interpretation.
    * Understand the spatial distribution of representational information.

* **Representational Similarity Analysis (RSA):**
    * Quantify the similarity between AI model representations and brain activity patterns.
    * Provides statistical measures (e.g., distance correlation) and visualization tools for direct comparison of representational geometries.

* **Statistical Significance Testing:**
    * Perform rigorous statistical analyses to assess the significance of findings across subjects and experimental conditions.

---

## 🚀 Getting Started

### Installation

```bash
# Recommended: Create a new virtual environment
python -m venv neuroai-env
source neuroai-env/bin/activate  # On Windows: `neuroai-env\Scripts\activate`

# Install from PyPI (once available)
pip install neuroai-toolkit

# Or, install from source (for latest development version)
git clone [https://github.com/Speech-NeuroAI-GP/NeuroAI-Toolkit.git](https://github.com/Speech-NeuroAI-GP/NeuroAI-Toolkit.git)
cd NeuroAI-Toolkit
pip install -e .
```
---
## Data Preparation: Understanding Expected Formats

`NeuroAI-Toolkit` is designed to work efficiently with neuroimaging data, particularly fMRI BOLD signals and AI model features. For optimal compatibility and ease of use, we strongly recommend organizing your fMRI data in accordance with the **Brain Imaging Data Structure (BIDS)** standard.

### fMRI BOLD Data (CIFTI Files)

 **Recommended Structure (BIDS-like):** Your fMRI BOLD data should be organized in a subject-centric hierarchy. Each subject folder should contain a `func` subdirectory, which holds CIFTI time series files (`.dtseries.nii`) for each functional run.
   
    
    data/
    ├── sub-01/
    │   ├── func/
    │   │   ├── sub-01_task-1_run-01_bold.dtseries.nii
    |   |   ├── sub-01_task-1_run-01_events.tsv # Mandatory for stimulus alignment
    │   │   ├── sub-01_task-2_run-01_bold.dtseries.nii
    │   │   └── ...
    ├── sub-02/
    │   └── func/
    │       └── ...
    └── ...
    
 **Naming Convention:** Functional CIFTI files are expected to follow the BIDS naming scheme: `sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_rec-<label>][_run-<index>]_bold.dtseries.nii`. The `task-<label>` is crucial as it links to your stimulus set.

 **Stimulus Information (`_events.tsv` - Mandatory for Alignment):**
For **every** BOLD run, a corresponding `_events.tsv` file is **mandatory**. This file must define the timing and identity of stimuli or events presented during that run. Specifically, it should contain at least the following columns:

* **`onset`**: The onset time of the stimulus/event in seconds, relative to the start of the functional run.
* **`duration`**: The duration of the stimulus/event in seconds.
* **`stim_file`**: The path to the specific stimulus file, relative to the `stimuli/` root directory (see section below).

#### Examples of `_events.tsv` content:

**1. Passive Listening:**
For datasets where a single stimulus (e.g., a story) spans an entire run, the `_events.tsv` would typically have one entry per run. The `stim_file` would point to the `.wav` file of the story. The `task-<label>` in your BOLD file name should match the base name of the `stim_file`.

`sub-01_ses-5_task-inamoment_events.tsv`
```tsv
onset	duration	stim_file
0.0	     450.0	  stories/story_name.wav
```
    
---
### AI Model Feature Files
 **Input for Feature Extraction Module:**

 If you utilize `NeuroAI-Toolkit`'s integrated feature extraction module, you will provide a path to your stimulus data. This module expects your stimuli to be organized in a way that allows it to identify and process them according to the `stim_file` specified in your `_events.tsv`.
    
 **Structure:** The top-level stimuli/ directory should contain subfolders that mirror the directory structure used in the `stim_file` column of your `_events.tsv` files. Place your actual stimulus files (e.g., .wav, .mp3, .jpg, .mp4) within these folders.

**Naming Convention:** Individual stimulus files within these folders should be named logically such that the full relative path to these files must precisely match the entries in the `stim_file` column of your `_events.tsv` files.

    stimuli/
    ├── object_recognition/
    │   ├── car_image.png
    │   ├── dog_image.png
    │   └── ...
    └── ...

The feature extraction module will then generate `.h5` file for your entire dataset based on your specified AI model, typically with keys corresponding to the unique identifiers for the functional runs (e.g. `task_label`_run-`index` ).

---
###  Expected Format for Pre-computed Features:

If you are providing your own pre-computed AI features (i.e., not using our feature extraction module), they **must be provided as a single `.h5` file** 

**Structure of the `.h5` file:**
*    **Keys:** Each key in the `.h5` file should correspond to a **unique identifier for a functional run** (e.g., `'story_A_run-01'`). This is critical for correctly aligning AI features with brain data.
*  **Values:** The value associated with each key will be a NumPy array representing the AI model features for all stimuli presented within that specific functional run, aligned to the run's temporal progression. The shape of this array will be (`number_of_TRs_in_run`, `AI_model_feature_dimension`). This means that for each TR in a given functional run, there will be a corresponding row of AI features.


## 📄 License

This project is licensed under the GPL v3 License - see the `LICENSE` file for details.


## 📧 Contact

For questions, issues, or collaborations, please open an issue on our GitHub repository or contact:
* nada.omran01@eng-st.cu.edu.eg
* nuran.khattab02@eng-st.cu.edu.eg
* youssefshawki89@gmail.com

## 📧 Authors
<div align="left">
  <a href="https://github.com/Nadaaomran">
    <img src="https://avatars.githubusercontent.com/u/104179154?v=4" width="100px" alt="@Nadaaomran">
  </a>
  <a href="https://github.com/joyou159">
    <img src="https://avatars.githubusercontent.com/u/85418161?v=4" width="100px" alt="@joyou159">
  </a>
  <a href="https://github.com/nouran-19">
    <img src="https://avatars.githubusercontent.com/u/99448829?v=4" width="100px" alt="@nouran-19">
  </a>
</div>


* * *
