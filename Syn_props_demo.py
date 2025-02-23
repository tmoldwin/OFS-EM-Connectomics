import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from synapse_analysis import synapse_size, synapse_shape, synapse_intensity, synapse_distance, synapse_colocalization, synapse_morphology


# Change to your path here
path_ = 'data/synpase_raw_em/'
synapses = pd.read_csv(f'{path_}synapse_data.csv', index_col=0)
print(len(synapses))
synapses.head()

syn_id = synapses.iloc[0].syn_id
img = np.load(f'{path_}{syn_id}_syn.npy')
pre_mask = np.load(f'{path_}{syn_id}_pre_syn_n_mask.npy')
post_mask = np.load(f'{path_}{syn_id}_post_syn_n_mask.npy')

# Calculate metrics for a single synapse
size_metrics = synapse_size(pre_mask, post_mask)
shape_metrics = synapse_shape(pre_mask, post_mask)
intensity_metrics = synapse_intensity(img, pre_mask, post_mask)
distance = synapse_distance(pre_mask, post_mask)
coloc_metrics = synapse_colocalization(pre_mask, post_mask)
morphology_metrics = synapse_morphology(pre_mask, post_mask)

# Create a dictionary with the metrics
metrics_dict = {
    'Synapse Area': size_metrics[0],
    'Pre-synaptic Area': size_metrics[1],
    'Post-synaptic Area': size_metrics[2],
    'Pre-Post Area Ratio': size_metrics[3],
    'Perimeter': shape_metrics[0],
    'Circularity': shape_metrics[1],
    'Mean Intensity': intensity_metrics[0],
    'Median Intensity': intensity_metrics[1],
    'Std Intensity': intensity_metrics[2],
    'Pre-Post Distance': distance,
    'Jaccard Index': coloc_metrics[0],
    'Pre-synaptic Overlap': coloc_metrics[1],
    'Post-synaptic Overlap': coloc_metrics[2],
    'Major Axis Length': morphology_metrics[0],
    'Minor Axis Length': morphology_metrics[1]
}

# Create a DataFrame from the dictionary
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value'])

# Pretty print the DataFrame
print("Synapse Metrics:")
print(metrics_df.to_string(index=True, header=False, float_format=lambda x: f"{x:.2f}"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
ax1.imshow(img.T, cmap='gray')
ax2.imshow(img.T, cmap='gray')
ax2.imshow(pre_mask.T, alpha=0.5)
ax2.imshow(post_mask.T, alpha=0.5)

plt.tight_layout()