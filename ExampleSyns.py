import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from metric_calc import (synapse_size, synapse_shape, synapse_intensity, 
                        synapse_distance, synapse_colocalization, synapse_morphology,
                        synapse_texture, synapse_intensity_distribution, 
                        synapse_boundary, ALL_FEATURES)
from sklearn.preprocessing import StandardScaler

def plot_synapse_comparison(synapses, path_, n_each=3):
    """Plot comparison of E/I synapses with features"""
    
    # Separate E and I synapses
    e_synapses = synapses[synapses['pre_syn_clf_type'].str.contains('E', case=False)]
    i_synapses = synapses[synapses['pre_syn_clf_type'].str.contains('I', case=False)]
    
    # Randomly sample n_each from each type
    e_sample = e_synapses.sample(n=n_each, random_state=42)
    i_sample = i_synapses.sample(n=n_each, random_state=42)
    sample_synapses = pd.concat([e_sample, i_sample])
    
    # Calculate features for all synapses
    all_features = []
    for _, row in sample_synapses.iterrows():
        syn_id = row['syn_id']
        img = np.load(f'{path_}{syn_id}_syn.npy')
        pre_mask = np.load(f'{path_}{syn_id}_pre_syn_n_mask.npy')
        post_mask = np.load(f'{path_}{syn_id}_post_syn_n_mask.npy')
        
        # Get all metrics (excluding intensity distribution)
        metrics = {
            **synapse_size(pre_mask, post_mask),
            **synapse_shape(pre_mask, post_mask),
            **synapse_intensity(img, pre_mask, post_mask),
            **synapse_distance(pre_mask, post_mask),
            **synapse_colocalization(pre_mask, post_mask),
            **synapse_morphology(pre_mask, post_mask),
            **synapse_texture(img, pre_mask, post_mask),
            **synapse_boundary(pre_mask, post_mask)
        }
        all_features.append(metrics)
    
    # Convert to DataFrame and Z-score the features
    features_df = pd.DataFrame(all_features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(6, 2*n_each), constrained_layout=True)
    gs = gridspec.GridSpec(6, 3, width_ratios=[1, 1, 4], figure=fig)   
    for idx, (_, row) in enumerate(sample_synapses.iterrows()):
        syn_id = row['syn_id']
        syn_type = 'Excitatory' if 'E' in row['pre_syn_clf_type'] else 'Inhibitory'
        
        # Load images
        img = np.load(f'{path_}{syn_id}_syn.npy')
        pre_mask = np.load(f'{path_}{syn_id}_pre_syn_n_mask.npy')
        post_mask = np.load(f'{path_}{syn_id}_post_syn_n_mask.npy')
        
        # Plot raw image
        ax1 = plt.subplot(gs[idx, 0])
        ax1.imshow(img.T, cmap='gray')
        ax1.set_title(syn_type, fontsize=8, pad=2)
        ax1.axis('off')
        
        # Plot masks
        ax2 = plt.subplot(gs[idx, 1])
        ax2.imshow(img.T, cmap='gray')
        ax2.imshow(pre_mask.T, alpha=0.5, cmap='Reds')
        ax2.imshow(post_mask.T, alpha=0.5, cmap='Blues')
        ax2.axis('off')
        
        # Plot features as bars
        ax3 = plt.subplot(gs[idx, 2])
        features = features_scaled.iloc[idx]
        x_pos = np.arange(len(features))
        colors = plt.cm.tab20(np.linspace(0, 1, len(features)))  # Generate unique colors
        ax3.bar(x_pos, features, color=colors, alpha=0.5)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Only show x-axis labels on bottom plot
        if idx == len(sample_synapses) - 1:
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(features.index, fontsize=6, rotation=90, ha='center')
        else:
            ax3.set_xticks([])
            
        ax3.set_title('Z-scored Features' if idx == 0 else '')
    
    # Remove tight_layout since we're using constrained_layout
    plt.savefig('figs/example_synapses.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load data
    path_ = 'data/synpase_raw_em/'
    synapses = pd.read_csv(f'{path_}synapse_data.csv', index_col=0)
    
    # Plot comparison
    plot_synapse_comparison(synapses, path_)