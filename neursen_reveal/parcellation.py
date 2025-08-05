import numpy as np
import hcp_utils as hcp
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import nilearn.plotting as plotting
from typing import Union, List, Dict, Any

def parcellate_brain_by_network(network):
  """
   Generates a binary brain mask for a specified functional brain network
    based on the Yan et al. (2017) parcellation (Kong et al., 2019).

    This function loads the 'yan_kong17' parcellation and creates a mask
    where voxels belonging to the specified functional network are marked with 1,
    and all other voxels are marked with 0.

    Parameters
    ----------
    network_name : str
        The name of the brain network to parcellate by. This name must exactly
        match one of the keys in the internal `network_map` dictionary
        (e.g., 'left_auditory', 'right_language').

    Returns
    -------
    np.ndarray
        A 1D NumPy array representing the binary brain mask.
        Shape: `(n_voxels,)`, where `n_voxels` corresponds to the
        total number of grayordinates (vertices/voxels) in the HCP MNI space
        (e.g., 91282 for resolution=1000).

  """
  yan_kong17 = hcp.load_hcp_parcellation('yan_kong17', resolution=1000)
  left_auditory = [
    'LH_Aud'
  ]
  right_auditory = [
    'RH_Aud'
  ]

  left_language = [
      'LH_Language'
  ]
  right_language = [
      'RH_Language'
  ]
  left_ventral_attention_salience_A = [
    'LH_SalVenAttnA'
  ]
  right_ventral_attention_salience_A = [
      'RH_SalVenAttnA'
  ]
  left_ventral_attention_salience_B = [
      'LH_SalVenAttnB'
  ]
  right_ventral_attention_salience_B = [
      'RH_SalVenAttnB'
  ]

  left_control_A = [
      'LH_ContA'
  ]
  right_control_A = [
      'RH_ContA'
  ]
  left_control_B = [
      'LH_ContB'
  ]
  right_control_B = [
      'RH_ContB'
  ]
  left_control_C = [
      'LH_ContC'
  ]
  right_control_C = [
      'RH_ContC'
  ]
  left_dorsal_attention_A = [
      'LH_DorsAttnA'
  ]
  right_dorsal_attention_A = [
      'RH_DorsAttnA'
  ]
  left_dorsal_attention_B = [
      'LH_DorsAttnB'
  ]
  right_dorsal_attention_B = [
      'RH_DorsAttnB'
  ]

  left_somatomotor_A = [
      'LH_SomMotA'
  ]
  right_somatomotor_A = [
      'RH_SomMotA'
  ]
  left_somatomotor_B = [
      'LH_SomMotB'
  ]
  right_somatomotor_B = [
      'RH_SomMotB'
  ]
  left_visual_A = [
      'LH_VisualA'
  ]
  right_visual_A = [
      'RH_VisualA'
  ]
  left_visual_B = [
      'LH_VisualB'
  ]
  right_visual_B = [
      'RH_VisualB'
  ]
  left_visual_C = [
      'LH_VisualC'
  ]
  right_visual_C = [
      'RH_VisualC'
  ]

  left_default_mode_A = [
      'LH_DefaultA'
  ]
  right_default_mode_A = [
      'RH_DefaultA'
  ]
  left_default_mode_B = [
      'LH_DefaultB'
  ]
  right_default_mode_B = [
      'RH_DefaultB'
  ]
  left_default_mode_C = [
      'LH_DefaultC'
  ]
  right_default_mode_C = [
      'RH_DefaultC'
  ]

  left_subcortical = [
      'thalamus_left',
      'caudate_left',
      'putamen_left',
      'pallidum_left',
      'diencephalon_left',
      'brainStem'
  ]
  right_subcortical = [
      'thalamus_right',
      'caudate_right',
      'putamen_right',
      'pallidum_right',
      'diencephalon_right',
      'brainStem']


  left_cerebellar = [
      'cerebellum_left'
  ]
  right_cerebellar = [
      'cerebellum_right'
  ]
  left_memory_related = [
      'hippocampus_left',  'LH_DefaultA_PHC', 'LH_DefaultC_PHC'
  ]
  right_memory_related = [
      'hippocampus_right', 'RH_DefaultA_PHC', 'RH_DefaultC_PHC'
  ]

  left_limbic = [
      'amygdala_left',  'accumbens_left'
  ]
  right_limbic = [
      'amygdala_right', 'accumbens_right'
  ]
  network_map = {
    'left_auditory': left_auditory,
    'left_language': left_language,
    'left_ventral_attention_salience_A': left_ventral_attention_salience_A,
    'left_ventral_attention_salience_B': left_ventral_attention_salience_B,
    'left_control_A': left_control_A,
    'left_control_B': left_control_B,
    'left_control_C': left_control_C,
    'left_dorsal_attention_A': left_dorsal_attention_A,
    'left_dorsal_attention_B': left_dorsal_attention_B,
    'left_somatomotor_A': left_somatomotor_A,
    'left_somatomotor_B': left_somatomotor_B,
    'left_visual_A': left_visual_A,
    'left_visual_B': left_visual_B,
    'left_visual_C': left_visual_C,
    'left_default_mode_A': left_default_mode_A,
    'left_default_mode_B': left_default_mode_B,
    'left_default_mode_C': left_default_mode_C,
    'left_subcortical': left_subcortical,
    'left_cerebellar': left_cerebellar,
    'left_memory_related': left_memory_related,
    'left_limbic': left_limbic,
    'right_auditory': right_auditory,
    'right_language': right_language,
    'right_ventral_attention_salience_A': right_ventral_attention_salience_A,
    'right_ventral_attention_salience_B': right_ventral_attention_salience_B,
    'right_control_A': right_control_A,
    'right_control_B': right_control_B,
    'right_control_C': right_control_C,
    'right_dorsal_attention_A': right_dorsal_attention_A,
    'right_dorsal_attention_B': right_dorsal_attention_B,
    'right_somatomotor_A': right_somatomotor_A,
    'right_somatomotor_B': right_somatomotor_B,
    'right_visual_A': right_visual_A,
    'right_visual_B': right_visual_B,
    'right_visual_C': right_visual_C,
    'right_default_mode_A': right_default_mode_A,
    'right_default_mode_B': right_default_mode_B,
    'right_default_mode_C': right_default_mode_C,
    'right_subcortical': right_subcortical,
    'right_cerebellar': right_cerebellar,
    'right_memory_related': right_memory_related,
    'right_limbic': right_limbic,
}
  indices = list()
  if network in network_map:
    for key, value in yan_kong17['labels'].items():
      if any(label in value for label in network_map[network]):
        indices.append(key)
    mask= np.isin(yan_kong17.map_all, indices).astype(int)
  else:
    raise ValueError("Invalid network.")
  return mask

def generate_legend(colors: List[Union[str, np.ndarray]], labels: List[str]) -> tuple:
  """
      Generates and displays a custom matplotlib legend based on provided colors and labels.

      This utility function is useful for creating a visual key for plotted brain regions
      where each region (or network) is assigned a specific color.

      Parameters
      ----------
      colors : list[Union[str, np.ndarray]]
          A list of colors for the legend entries. Each element can be a string
          (e.g., 'red', 'blue') or an RGBA numpy array (e.g., `[0.1, 0.2, 0.3, 1.0]`).
          The alpha channel (last element for RGBA) is important for `hcp_utils` colors.
      labels : list[str]
          A list of string labels corresponding to each color. These labels will be
          displayed next to their respective color patches in the legend.
          The order of `labels` must match the order of `colors`.

      Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the Matplotlib Figure and Axes objects of the generated legend.
        These can be displayed or saved as needed.
  """
  legend_elements = [Patch(facecolor=color, edgecolor='black', label=label)
                   for color, label in zip(colors, labels)]

  fig, ax = plt.subplots(figsize=(8, 2))
  ax.axis('off')

  ax.legend(handles=legend_elements, loc='center', ncol=5, frameon=False)

  return fig, ax


def plot_parcellation_of_peak_intensity_regions(
    data: np.ndarray, 
    threshold: Union[float, str] = 0,
    title: str = "Parcellation results"
) -> tuple: # Returns a nilearn.plotting.html_surface.HTMLSurface object
  """
    Visualizes brain regions with activity (or scores) above a specified threshold
    by mapping them onto a cortical surface and coloring them according to their
    functional network membership. It also generates a legend for the networks.

    The function takes a 1D array of data (e.g., voxel-wise RSA scores, t-statistics),
    applies a threshold to identify "peak intensity" regions, and then identifies
    which functional networks (from Yan et al. 2017 parcellation) these regions belong to.
    It then plots these regions on an inflated HCP surface using `nilearn.plotting.view_surf`.

    Parameters
    ----------
    data : np.ndarray
        A 1D NumPy array representing brain data (e.g., voxel-wise scores, activation values).
        Expected shape: `(n_voxels,)`, matching the grayordinate count of the
        'yan_kong17' parcellation (e.g., 91282).
    threshold : Union[float, str], optional
        The threshold value to apply to the `data`. Only values strictly
        greater than this threshold will be considered for plotting.
        - If a `float`, it's used directly as the threshold.
        - If a `str` ending with '%` (e.g., '90%'), it's interpreted as a
          percentile. The threshold will be calculated as the `threshold`-th
          percentile of the input `data`.
        Defaults to 0.
    title : str, optional
            Title for the rendered brain plot.

    Returns
    -------
    Tuple[nilearn.plotting.html_surface.HTMLSurface, matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing:
        - The NiLearn `HTMLSurface` object of the parcellated brain regions.
        - The Matplotlib Figure object of the generated legend.
        - The Matplotlib Axes object of the generated legend.
  """
  yan_kong17 = hcp.load_hcp_parcellation('yan_kong17', resolution=1000)
  data = data.copy()
  auditory = [
    'Aud'
  ]

  language = [
      'Language'
  ]
  ventral_attention_salience_A = [
    'SalVenAttnA'
  ]
  ventral_attention_salience_B = [
      'SalVenAttnB'
  ]

  control_A = [
      'ContA'
  ]
  control_B = [
      'ContB'
  ]
  control_C = [
      'ContC'
  ]
  dorsal_attention_A = [
      'DorsAttnA'
  ]
  dorsal_attention_B = [
      'DorsAttnB'
  ]

  somatomotor_A = [
      'SomMotA'
  ]
  somatomotor_B = [
      'SomMotB'
  ]
  visual_A = [
      'VisualA'
  ]
  visual_B = [
      'VisualB'
  ]
  visual_C = [
      'VisualC'
  ]

  default_mode_A = [
      'DefaultA'
  ]
  default_mode_B = [
      'DefaultB'
  ]
  default_mode_C = [
      'DefaultC'
  ]

  subcortical = [
      'thalamus_left', 'thalamus_right',
      'caudate_left', 'caudate_right',
      'putamen_left', 'putamen_right',
      'pallidum_left', 'pallidum_right',
      'diencephalon_left', 'diencephalon_right',
      'brainStem', 'hippocampus_left', 'hippocampus_right', 'amygdala_left', 'amygdala_right', 'accumbens_left', 'accumbens_right', 'cerebellum_left', 'cerebellum_right'
  ]
  network_map = {
    'auditory': auditory,
    'language': language,
    'ventral_attention_salience_A': ventral_attention_salience_A,
    'ventral_attention_salience_B': ventral_attention_salience_B,
    'control_A': control_A,
    'control_B': control_B,
    'control_C': control_C,
    'dorsal_attention_A': dorsal_attention_A,
    'dorsal_attention_B': dorsal_attention_B,
    'somatomotor_A': somatomotor_A,
    'somatomotor_B': somatomotor_B,
    'visual_A': visual_A,
    'visual_B': visual_B,
    'visual_C': visual_C,
    'default_mode_A': default_mode_A,
    'default_mode_B': default_mode_B,
    'default_mode_C': default_mode_C,
    'subcortical': subcortical,
}
  if isinstance(threshold, str):
    threshold= int(threshold.replace("%", ''))
    threshold = np.percentile(data, threshold)
  else:
    threshold = threshold

  data[data < threshold]=0
  indices= np.where(data > threshold)[0]
  areas={}
  for index in indices:
    area= yan_kong17.labels[yan_kong17.map_all[index]]
    for network in network_map.keys():
      if any(label in area for label in network_map[network]):
        if network in areas:
          areas[network].append(index)
        else:
          areas[network]= [index]
  colors=[]
  for index, key in enumerate(areas.keys()):
    if key == 'subcortical':
      continue
    data[areas[key]]= index +1
    colors.append(yan_kong17.rgba[yan_kong17.map_all[areas[key][0]]])
  legend_fig, legend_ax =generate_legend(colors, areas.keys())
  light_gray = [0.85, 0.85, 0.85, 1.0]
  colors = np.vstack([light_gray, colors])
  custom_cmap = ListedColormap(colors)
  view= plotting.view_surf(
    hcp.mesh.inflated,
    hcp.cortex_data(data),
    cmap=custom_cmap,
    colorbar= True,
    title= title,
    symmetric_cmap=False,
    bg_map=hcp.mesh.sulc)

  return legend_fig, legend_ax, view

