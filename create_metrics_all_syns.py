import pandas as pd
import numpy as np
from metric_calc import (synapse_size, synapse_shape, synapse_intensity, 
                        synapse_distance, synapse_colocalization, synapse_morphology,
                        synapse_texture, synapse_intensity_distribution, 
                        synapse_boundary)

def calculate_synapse_metrics(synapses, path_):
    metrics_data = []

    for i, row in synapses.iterrows():
        print(i)
        syn_id = row['syn_id']
        img = np.load(f'{path_}{syn_id}_syn.npy')
        pre_mask = np.load(f'{path_}{syn_id}_pre_syn_n_mask.npy')
        post_mask = np.load(f'{path_}{syn_id}_post_syn_n_mask.npy')

        # Get all metrics as dictionaries
        metrics = {
            **{'syn_id': syn_id},
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
        
        metrics_data.append(metrics)

    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df

# Change to your path here
path_ = 'data/synpase_raw_em/'
synapses = pd.read_csv(f'{path_}synapse_data.csv', index_col=0)

metrics_df = calculate_synapse_metrics(synapses, path_)

# Merge the metrics DataFrame with the original synapses DataFrame
synapses_with_metrics = pd.merge(synapses, metrics_df, on='syn_id')

# Save the updated DataFrame to a new CSV file
synapses_with_metrics.to_csv(f'{path_}synapse_data_with_metrics.csv', index=False) 