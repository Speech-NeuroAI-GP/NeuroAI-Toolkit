.. _installation:

Installation and Setup
=======================

This section guides you through setting up `neursen_reveal` and preparing your environment for use.

Python Package Installation
----------------------------

We recommend using a virtual environment to manage dependencies and avoid conflicts with your system's Python installation.

1.  **Create a new virtual environment (recommended):**

    ```bash
    python -m venv neursen-reveal-env
    ```

2.  **Activate your virtual environment:**

    * **On Linux/macOS:**
        ```bash
        source neursen-reveal-env/bin/activate
        ```
    * **On Windows:**
        ```bash
        neursen-reveal-env\Scripts\activate
        ```

3.  **Install the `neursen_reveal` package:**

    Currently, `neursen_reveal` is primarily installed from its source repository. Once available on PyPI, `pip install neursen-reveal` will be the simplest method.

    * **Install from source (for latest development version):**

    First, clone the repository. 
        
        git clone [https://github.com/Speech-NeuroAI-GP/NeuroAI-Toolkit.git](https://github.com/Speech-NeuroAI-GP/NeuroAI-Toolkit.git)
        

    Then, navigate into the cloned directory and install the package:

        cd NeuroAI-Toolkit
        pip install -e .
        

Important Note on Smoothing (Workbench Dependency)
--------------------------------------------------

`neursen_reveal` offers **surface-based spatial smoothing** functionality as part of its data cleaning pipeline (`fmri_cleaning.run_cleaning_pipeline`). This feature relies on the **Connectome Workbench (`wb_command`)** for its high-quality surface operations.

**Key Considerations:**

* **Linux Only:** Workbench is primarily supported on **Linux operating systems**. Therefore, the smoothing feature within this toolkit is **unavailable for users on Windows or macOS**.
* **Installation for Linux Users:** If you are a Linux user and wish to utilize the smoothing feature, you must have Connectome Workbench installed and accessible in your system's PATH. If you don't already have it, you can easily install it using our provided script:

    ```
    # From the root directory of the cloned NeuroAI-Toolkit repository:

    cd NeuroAI-Toolkit

    # Make the installation script executable

    chmod +x install_workbench.sh

    # Run the installation script
    
    ./install_workbench.sh
    ```
This script will download, install, and configure Workbench for you. After running it, you may need to restart your terminal or re-source your `~/.bashrc` for the `wb_command` to be recognized in new shell sessions. For interactive Python environments like Jupyter/Colab, you might need to manually update the `PATH` environment variable for the current session (as described in the :ref:`example_usage` notebook).
