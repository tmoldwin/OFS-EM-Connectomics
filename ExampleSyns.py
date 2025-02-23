import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metric_calc import (synapse_size, synapse_shape, synapse_intensity, 
                        synapse_distance, synapse_colocalization, synapse_morphology,
                        synapse_texture, synapse_intensity_distribution, 
                        synapse_boundary)


# Change to your path here
path_ = 'data/synpase_raw_em/'
synapses = pd.read_csv(f'{path_}synapse_data.csv', index_col=0)
print(len(synapses))
print(synapses.columns)

syn_id = synapses.iloc[0].syn_id
img = np.load(f'{path_}{syn_id}_syn.npy')
pre_mask = np.load(f'{path_}{syn_id}_pre_syn_n_mask.npy')
post_mask = np.load(f'{path_}{syn_id}_post_syn_n_mask.npy')

# Calculate all metrics
metrics = {
    **synapse_size(pre_mask, post_mask),
    **synapse_shape(pre_mask, post_mask),
    **synapse_intensity(img, pre_mask, post_mask),
    **synapse_distance(pre_mask, post_mask),
    **synapse_colocalization(pre_mask, post_mask),
    **synapse_morphology(pre_mask, post_mask),
    **synapse_texture(img, pre_mask, post_mask),
    **synapse_intensity_distribution(img, pre_mask, post_mask),
    **synapse_boundary(pre_mask, post_mask)
}

# Create a DataFrame with prettier display names
display_names = {
    'synapse_area': 'Synapse Area',
    'pre_area': 'Pre-synaptic Area',
    'post_area': 'Post-synaptic Area',
    'pre_post_area_ratio': 'Pre-Post Area Ratio',
    # Add more display names for the new metrics...
}

metrics_df = pd.DataFrame.from_dict(
    {display_names.get(k, k.replace('_', ' ').title()): [v] 
     for k, v in metrics.items()},
    orient='index',
    columns=['Value']
)

# Pretty print the DataFrame
print("Synapse Metrics:")
print(metrics_df.to_string(index=True, header=False, float_format=lambda x: f"{x:.2f}"))

# Plot original visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
ax1.imshow(img.T, cmap='gray')
ax2.imshow(img.T, cmap='gray')
ax2.imshow(pre_mask.T, alpha=0.5)
ax2.imshow(post_mask.T, alpha=0.5)

plt.tight_layout()
plt.show()