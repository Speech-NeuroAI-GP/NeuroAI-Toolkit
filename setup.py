from setuptools import setup, find_packages
setup(
    name='neuroai',         # name of the package
    version='0.1.0',                  # version
    packages=find_packages(),         # automatically find folders with __init__.py
    install_requires=[
        'dcor==0.6',
        'h5py==3.12.1',
        'hcp-utils @ git+https://github.com/GasserElbanna/hcp-utils.git@master',
        'himalaya==0.4.6',
        'joblib==1.4.2',
        'librosa==0.10.2.post1',
        'matplotlib==3.10.3',
        'nibabel==5.3.2',
        'nilearn==0.12.0',
        'numpy==2.3.1',
        'pandas==2.3.1',
        'Pillow==11.3.0',
        'scikit_learn==1.7.0',
        'scipy==1.16.0',
        'setuptools==75.6.0',
        'speechbrain==1.0.2',
        'statsmodels==0.14.5',
        'torch==2.5.1',
        'torchvision==0.20.1',
        'transformers==4.45.2',
        'voxelwise_tutorials==0.2.3'],
    author='Nada Omran, Youssef Ahmed, Nouran Khattab',
    description="The `NeuroAI-Toolkit` is a powerful Python package designed to bridge the gap between cutting-edge Artificial Intelligence models and intricate neuroimaging data. Tailored for researchers exploring the neural underpinnings of perception and cognition, this toolkit provides a streamlined pipeline for advanced analysis, from feature extraction to brain representation similarity and modeling. Whether you're investigating how the brain processes speech, language, or visual stimuli, `NeuroAI-Toolkit` offers a robust and flexible suite of tools to accelerate your research and uncover meaningful insights."
)