import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from metric_calc import ALL_FEATURES, FEATURE_NAMES

def load_metrics():
    """Load pre-calculated metrics from CSV"""
    metrics_df = pd.read_csv('data/synapse_data_with_metrics.csv')
    
    # Use standardized feature names from metric_calc
    feature_cols = [col for col in metrics_df.columns if col in ALL_FEATURES]
    
    # Group features by category for analysis
    feature_groups = {
        group: [col for col in metrics_df.columns if col in features]
        for group, features in FEATURE_NAMES.items()
    }
    
    print("\nFeature groups available:")
    for group, features in feature_groups.items():
        print(f"\n{group.title()}:")
        print(f"Found {len(features)} features: {', '.join(features)}")
    
    return metrics_df, feature_cols, feature_groups

def perform_clustering_analysis(metrics_df, feature_cols, n_clusters=3, method='pca'):
    """Perform dimensionality reduction and clustering on the metrics"""
    X = metrics_df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = reducer.explained_variance_ratio_
    else:  # umap
        reducer = umap.UMAP(random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = None
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return X_reduced, clusters, explained_var

def plot_clusters(X_reduced, clusters, variance_ratio=None, method='pca'):
    """Plot the clustering results"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis')
    
    if method == 'pca':
        plt.xlabel(f'PC1 ({variance_ratio[0]:.2%} variance explained)')
        plt.ylabel(f'PC2 ({variance_ratio[1]:.2%} variance explained)')
        title = 'Synapse Clusters (PCA)'
    else:
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        title = 'Synapse Clusters (UMAP)'
        
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def perform_umap(metrics_df, feature_cols):
    """Perform UMAP dimensionality reduction"""
    X = metrics_df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reducer = umap.UMAP(random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    return X_reduced

def plot_cell_types(metrics_df, X_umap):
    """Plot UMAP colored by different cell type classifications"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Define the subplots data
    plots = [
        ('pre_syn_cell_type', 'Pre-synaptic Cell Type', 0, 0),
        ('pre_syn_clf_type', 'Pre-synaptic Classification', 0, 1),
        ('post_syn_cell_type', 'Post-synaptic Cell Type', 1, 0),
        ('post_syn_clf_type', 'Post-synaptic Classification', 1, 1)
    ]
    
    for col, title, i, j in plots:
        categories = pd.Categorical(metrics_df[col])
        scatter = axes[i,j].scatter(X_umap[:, 0], X_umap[:, 1], 
                                  c=categories.codes,
                                  cmap='tab20')
        axes[i,j].set_title(title)
        axes[i,j].set_xlabel('UMAP1')
        axes[i,j].set_ylabel('UMAP2')
        
        # Add legend
        unique_categories = categories.categories
        n_categories = len(unique_categories)
        colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=colors[idx],
                                    label=cat, markersize=10)
                         for idx, cat in enumerate(unique_categories)]
        
        axes[i,j].legend(handles=legend_elements,
                        title=col,
                        bbox_to_anchor=(1.05, 1),
                        loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figs/unsupervised.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    metrics_df, feature_cols, feature_groups = load_metrics()
    
    # Generate UMAP embedding
    X_umap = perform_umap(metrics_df, feature_cols)
    
    # Plot cell type visualizations
    plot_cell_types(metrics_df, X_umap) 