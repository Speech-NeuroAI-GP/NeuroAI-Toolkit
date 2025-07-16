from setuptools import setup, find_packages
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
setup(
    name='neuroai',         # name of the package
    version='0.1.0',                  # version
    packages=find_packages(),         # automatically find folders with __init__.py
    install_requires=requirements,
    author='Nada Omran, Youssef Ahmed, Nouran Khattab',
    description="The `NeuroAI-Toolkit` is a powerful Python package designed to bridge the gap between cutting-edge Artificial Intelligence models and intricate neuroimaging data. Tailored for researchers exploring the neural underpinnings of perception and cognition, this toolkit provides a streamlined pipeline for advanced analysis, from feature extraction to brain representation similarity and modeling. Whether you're investigating how the brain processes speech, language, or visual stimuli, `NeuroAI-Toolkit` offers a robust and flexible suite of tools to accelerate your research and uncover meaningful insights.",
    url='https://github.com/Speech-NeuroAI-GP/NeuroAI-Toolkit.git'
)