import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from typing import List, Tuple

def generate_points(n: int, mode: str, stick_length: float = 1.0) -> np.ndarray:
    """Generate points on a stick using different distribution patterns."""
    if mode == 'uniform':
        return np.random.uniform(0, stick_length, n)
    
    elif mode == 'two_bunches':
        # Mix of two normal distributions centered at 0.25 and 0.75 of stick length
        points = np.concatenate([
            np.random.normal(0.25 * stick_length, 0.05 * stick_length, n//2),
            np.random.normal(0.75 * stick_length, 0.05 * stick_length, n//2 + n%2)
        ])
        return np.clip(points, 0, stick_length)
    
    elif mode == 'three_clusters':
        # Three clusters at 0.2, 0.5, and 0.8 of stick length
        points = np.concatenate([
            np.random.normal(0.2 * stick_length, 0.02 * stick_length, n//3),
            np.random.normal(0.5 * stick_length, 0.02 * stick_length, n//3),
            np.random.normal(0.8 * stick_length, 0.02 * stick_length, n//3 + n%3)
        ])
        return np.clip(points, 0, stick_length)
    
    raise ValueError(f"Unknown mode: {mode}")

def calculate_metrics(points: np.ndarray) -> dict:
    """Calculate various distribution metrics."""
    # Sort points for nearest neighbor calculation
    sorted_points = np.sort(points)
    
    # Calculate nearest neighbor distances
    nn_distances = np.diff(sorted_points)  # distances to next neighbor
    
    # Calculate pairwise distances
    pairwise_distances = np.abs(points[:, None] - points)
    np.fill_diagonal(pairwise_distances, np.inf)  # exclude self-distances
    min_distances = np.min(pairwise_distances, axis=1)  # minimum distance for each point
    
    return {
        'mean_nn_dist': np.mean(min_distances),
        'std_nn_dist': np.std(min_distances),
        'mean_pairwise': np.mean(pairwise_distances[pairwise_distances != np.inf]),
        'std_pairwise': np.std(pairwise_distances[pairwise_distances != np.inf])
    }

def plot_experiment(ax: plt.Axes = None, points: np.ndarray = None, 
                   metrics: dict = None, title: str = '') -> plt.Axes:
    """Plot points and their distribution on given axes."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 2))
    
    # Plot points as dots
    ax.scatter(points, np.ones_like(points) * 0.5, color='black', alpha=0.5, s=50)
    
    # Add kernel density estimate
    if len(points) > 1:
        kde = gaussian_kde(points)
        x = np.linspace(0, 1, 200)
        ax.plot(x, kde(x), 'r-', alpha=0.5)
    
    # Draw the stick
    ax.plot([0, 1], [0.5, 0.5], 'k-', linewidth=2, alpha=0.3)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_yticks([])
    
    return ax

def plot_metrics(ax: plt.Axes = None, metrics: dict = None, 
                title: str = '') -> plt.Axes:
    """Plot metrics as bar chart on given axes."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 2))
    
    x = np.arange(len(metrics))
    ax.bar(x, list(metrics.values()))
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), rotation=45)
    ax.set_title(title)
    
    return ax

def run_experiments():
    """Run all experiments and create visualization."""
    # Define base experiments (will be run with both stick lengths)
    base_experiments = [
        ('uniform', 10, 'Uniform\nN=10'),
        ('uniform', 100, 'Uniform\nN=100'),
        ('two_bunches', 10, 'Two Bunches\nN=10'),
        ('two_bunches', 100, 'Two Bunches\nN=100'),
        ('three_clusters', 10, 'Three Clusters\nN=10'),
        ('three_clusters', 100, 'Three Clusters\nN=100')
    ]
    
    # Color scheme for size-N combinations
    size_n_colors = {
        (0.1, 10): 'skyblue',
        (0.1, 100): 'lightgreen',
        (1.0, 10): 'salmon',
        (1.0, 100): 'plum'
    }
    
    # Create full experiment list with both stick lengths
    experiments = []
    for mode, n, base_title in base_experiments:
        # Short stick version
        experiments.append((mode, n, 0.1, f"{base_title}\nL=0.1"))
        # Long stick version
        experiments.append((mode, n, 1.0, f"{base_title}\nL=1.0"))
    
    # Reorder experiments to group by distribution type
    experiments = sorted(experiments, key=lambda x: (x[0], x[2], x[1]))
    
    # Create two figures
    fig_dist, axes_dist = plt.subplots(4, 3, figsize=(15, 20))
    fig_dist.suptitle('Distribution Experiments', y=0.95, fontsize=14)
    
    fig_summary, axes_summary = plt.subplots(2, 2, figsize=(15, 10))
    fig_summary.suptitle('Metric Summary', y=0.95, fontsize=14)
    
    # Store all metrics to determine common y-axis scale
    all_metrics = []
    
    # First generate all data and collect metrics
    results = []
    for mode, n, stick_length, title in experiments:
        points = generate_points(n, mode, stick_length)
        metrics = calculate_metrics(points)
        results.append((points, metrics, title, stick_length, mode, n))
        all_metrics.append(metrics)
    
    # Plot distribution experiments
    for i, (points, metrics, title, stick_length, mode, n) in enumerate(results):
        row = i // 3
        col = i % 3
        
        # Plot points distribution
        ax_dist = axes_dist[row, col]
        plot_experiment(ax_dist, points, metrics, title)
        ax_dist.set_xlim(-0.05 * stick_length, 1.05 * stick_length)
        ax_dist.tick_params(labelsize=8)
        ax_dist.set_title(title, fontsize=10, pad=5)
    
    # Create summary plots
    metric_names = {
        'mean_nn_dist': 'Mean NN',
        'std_nn_dist': 'Std NN',
        'mean_pairwise': 'Mean Pair',
        'std_pairwise': 'Std Pair'
    }
    
    for idx, (metric, label) in enumerate(metric_names.items()):
        ax = axes_summary[idx//2, idx%2]
        values = [m[metric] for m in all_metrics]
        x = np.arange(len(experiments))
        
        # Color bars based on size-N combination
        bar_colors = [size_n_colors[(result[3], result[5])] for result in results]
        
        ax.bar(x, values, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels([exp[3] for exp in experiments], rotation=45, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10, pad=10)
        ax.set_ylabel('Distance', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # Adjust layouts
    plt.figure(fig_dist.number)
    plt.tight_layout()
    
    plt.figure(fig_summary.number)
    plt.tight_layout()
    
    return fig_dist, fig_summary

if __name__ == '__main__':
    fig_dist, fig_summary = run_experiments()
    plt.show()
