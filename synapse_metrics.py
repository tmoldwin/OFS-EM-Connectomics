import pandas as pd
import numpy as np
from synapse_analysis import synapse_size, synapse_shape, synapse_intensity, synapse_distance, synapse_colocalization, synapse_morphology

def calculate_synapse_metrics(synapses, path_):
    metrics_data = []

    for i, row in synapses.iterrows():
        print(i)
        syn_id = row['syn_id']
        img = np.load(f'{path_}{syn_id}_syn.npy')
        pre_mask = np.load(f'{path_}{syn_id}_pre_syn_n_mask.npy')
        post_mask = np.load(f'{path_}{syn_id}_post_syn_n_mask.npy')

        size_metrics = synapse_size(pre_mask, post_mask)
        shape_metrics = synapse_shape(pre_mask, post_mask)
        intensity_metrics = synapse_intensity(img, pre_mask, post_mask)
        distance = synapse_distance(pre_mask, post_mask)
        coloc_metrics = synapse_colocalization(pre_mask, post_mask)
        morphology_metrics = synapse_morphology(pre_mask, post_mask)

        metrics_row = {
            'syn_id': syn_id,
            'synapse_area': size_metrics[0],
            'pre_area': size_metrics[1],
            'post_area': size_metrics[2],
            'pre_post_ratio': size_metrics[3],
            'perimeter': shape_metrics[0],
            'circularity': shape_metrics[1],
            'mean_intensity': intensity_metrics[0],
            'median_intensity': intensity_metrics[1],
            'std_intensity': intensity_metrics[2],
            'pre_post_distance': distance,
            'jaccard_index': coloc_metrics[0],
            'pre_overlap': coloc_metrics[1],
            'post_overlap': coloc_metrics[2],
            'major_axis_length': morphology_metrics[0],
            'minor_axis_length': morphology_metrics[1]
        }

        metrics_data.append(metrics_row)

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