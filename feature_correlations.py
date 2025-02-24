import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_correlations(df):
    """Plot correlation matrix of features and cell type specific patterns"""
    
    # Get feature columns (excluding metadata columns)
    feature_cols = ['synapse_area', 'pre_area', 'post_area', 'pre_post_ratio',
                   'perimeter', 'circularity', 'mean_intensity', 'median_intensity',
                   'std_intensity', 'pre_post_distance', 'jaccard_index',
                   'pre_overlap', 'post_overlap', 'major_axis_length', 'minor_axis_length']
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = plt.GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1, 1])
    
    # Plot correlation matrix
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im1, ax=ax1)
    
    # Add feature labels
    ax1.set_xticks(np.arange(len(feature_cols)))
    ax1.set_yticks(np.arange(len(feature_cols)))
    ax1.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax1.set_yticklabels(feature_cols)
    ax1.set_title('Feature Correlations')
    
    # Calculate cell type specific patterns
    # First, z-score the features
    df_z = df[feature_cols].copy()
    df_z = (df_z - df_z.mean()) / df_z.std()
    
    # Calculate median values for pre-synaptic cell types
    pre_medians = df_z.groupby(df['pre_syn_cell_type'])[feature_cols].median()
    pre_E = df_z[df['pre_syn_clf_type'] == 'E'][feature_cols].median()
    pre_I = df_z[df['pre_syn_clf_type'] == 'I'][feature_cols].median()
    pre_medians['E'] = pre_E
    pre_medians['I'] = pre_I
    
    # Calculate median values for post-synaptic cell types
    post_medians = df_z.groupby(df['post_syn_cell_type'])[feature_cols].median()
    post_E = df_z[df['post_syn_clf_type'] == 'E'][feature_cols].median()
    post_I = df_z[df['post_syn_clf_type'] == 'I'][feature_cols].median()
    post_medians['E'] = post_E
    post_medians['I'] = post_I
    
    # Plot pre-synaptic cell type patterns
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(pre_medians.T, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks(np.arange(len(pre_medians.index)))
    ax2.set_yticks(np.arange(len(feature_cols)))
    ax2.set_xticklabels(pre_medians.index, rotation=45, ha='right')
    ax2.set_yticklabels(feature_cols)
    ax2.set_title('Pre-synaptic Cell Types')
    
    # Plot post-synaptic cell type patterns
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(post_medians.T, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xticks(np.arange(len(post_medians.index)))
    ax3.set_yticks(np.arange(len(feature_cols)))
    ax3.set_xticklabels(post_medians.index, rotation=45, ha='right')
    ax3.set_yticklabels(feature_cols)
    ax3.set_title('Post-synaptic Cell Types')
    
    plt.savefig('figs/feature_correlations.png', dpi=300)
    plt.show()

# Load data and create plots
df = pd.read_csv('data/synapse_data_with_metrics.csv')
plot_feature_correlations(df)