import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

# At the top of the file, after imports
FEATURE_NAMES = {
    'size': [
        'synapse_area',
        'pre_area',
        'post_area',
        'pre_post_area_ratio'
    ],
    'shape': [
        'perimeter',
        'circularity'
    ],
    'intensity': [
        'mean_intensity',
        'median_intensity',
        'std_intensity'
    ],
    'distance': [
        'pre_post_distance'
    ],
    'colocalization': [
        'jaccard_index',
        'pre_overlap',
        'post_overlap'
    ],
    'morphology': [
        'major_axis_length',
        'minor_axis_length'
    ],
    'intensity_distribution': [
        'pre_25th',
        'pre_median',
        'pre_75th',
        'post_25th',
        'post_median',
        'post_75th',
        'pre_post_median_ratio',
        'pre_post_iqr_ratio',
        'pre_intensity_range',
        'post_intensity_range'
    ],
    'texture': [
        'pre_contrast',
        'pre_homogeneity',
        'pre_energy',
        'pre_correlation',
        'post_contrast',
        'post_homogeneity',
        'post_energy',
        'post_correlation'
    ],
    'boundary': [
        'pre_perimeter',
        'post_perimeter',
        'pre_solidity',
        'post_solidity',
        'pre_eccentricity',
        'post_eccentricity'
    ]
}

# Also add a flat list for convenience
ALL_FEATURES = [feature for group in FEATURE_NAMES.values() for feature in group]

def synapse_size(pre_mask, post_mask):
    """Calculate synapse size metrics."""
    synapse_mask = pre_mask | post_mask
    synapse_area = np.sum(synapse_mask)
    pre_area = np.sum(pre_mask)
    post_area = np.sum(post_mask)
    pre_post_ratio = pre_area / post_area
    
    return {
        'synapse_area': synapse_area,
        'pre_area': pre_area,
        'post_area': post_area,
        'pre_post_area_ratio': pre_post_ratio
    }

def synapse_shape(pre_mask, post_mask):
    """Calculate synapse shape metrics."""
    synapse_mask = pre_mask | post_mask
    perimeter = measure.perimeter(synapse_mask)
    circularity = 4 * np.pi * np.sum(synapse_mask) / (perimeter ** 2)
    
    return {
        'perimeter': perimeter,
        'circularity': circularity
    }

def synapse_intensity(image, pre_mask, post_mask):
    """Calculate synapse intensity metrics."""
    synapse_mask = pre_mask | post_mask
    synapse_intensities = image[synapse_mask]
    mean_intensity = np.mean(synapse_intensities)
    median_intensity = np.median(synapse_intensities)
    std_intensity = np.std(synapse_intensities)
    
    return {
        'mean_intensity': mean_intensity,
        'median_intensity': median_intensity,
        'std_intensity': std_intensity
    }

def synapse_distance(pre_mask, post_mask):
    """Calculate distance between pre- and post-synaptic regions."""
    pre_centroid = measure.centroid(pre_mask)
    post_centroid = measure.centroid(post_mask)
    distance = np.sqrt((pre_centroid[0] - post_centroid[0]) ** 2 +
                      (pre_centroid[1] - post_centroid[1]) ** 2)
    
    return {'pre_post_distance': distance}

def synapse_colocalization(pre_mask, post_mask):
    """Calculate synapse co-localization metrics."""
    intersection = pre_mask & post_mask
    union = pre_mask | post_mask
    jaccard_index = np.sum(intersection) / np.sum(union)
    pre_overlap = np.sum(intersection) / np.sum(pre_mask)
    post_overlap = np.sum(intersection) / np.sum(post_mask)
    
    return {
        'jaccard_index': jaccard_index,
        'pre_overlap': pre_overlap,
        'post_overlap': post_overlap
    }

def synapse_morphology(pre_mask, post_mask):
    """Calculate synapse morphology metrics."""
    synapse_mask = pre_mask | post_mask
    labeled_synapse = measure.label(synapse_mask)
    properties = measure.regionprops(labeled_synapse)
    
    if properties:
        major_axis_length = properties[0].major_axis_length
        minor_axis_length = properties[0].minor_axis_length
    else:
        major_axis_length = 0
        minor_axis_length = 0
    
    return {
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length
    }

def synapse_intensity_distribution(image, pre_mask, post_mask):
    """Calculate detailed intensity distribution metrics for pre and post regions."""
    # Get intensities for each region
    pre_intensities = image[pre_mask]
    post_intensities = image[post_mask]
    
    # Calculate percentiles
    pre_percentiles = np.percentile(pre_intensities, [25, 50, 75])
    post_percentiles = np.percentile(post_intensities, [25, 50, 75])
    
    # Calculate intensity ratios and differences
    pre_post_median_ratio = pre_percentiles[1] / post_percentiles[1]
    pre_post_iqr_ratio = (pre_percentiles[2] - pre_percentiles[0]) / (post_percentiles[2] - post_percentiles[0])
    
    # Calculate intensity gradients
    pre_gradient = np.max(pre_intensities) - np.min(pre_intensities)
    post_gradient = np.max(post_intensities) - np.min(post_intensities)
    
    return {
        'pre_25th': pre_percentiles[0],
        'pre_median': pre_percentiles[1],
        'pre_75th': pre_percentiles[2],
        'post_25th': post_percentiles[0],
        'post_median': post_percentiles[1],
        'post_75th': post_percentiles[2],
        'pre_post_median_ratio': pre_post_median_ratio,
        'pre_post_iqr_ratio': pre_post_iqr_ratio,
        'pre_intensity_range': pre_gradient,
        'post_intensity_range': post_gradient
    }

def synapse_texture(image, pre_mask, post_mask):
    """Calculate texture metrics that might distinguish E/I synapses."""
    # Get masked regions
    pre_region = image * pre_mask
    post_region = image * post_mask
    
    # Calculate GLCM features for both regions
    def get_texture_features(region):
        # Normalize to 8-bit range
        region_norm = ((region - region.min()) * (255.0 / (region.max() - region.min()))).astype(np.uint8)
        glcm = graycomatrix(region_norm, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        return contrast, homogeneity, energy, correlation
    
    pre_texture = get_texture_features(pre_region)
    post_texture = get_texture_features(post_region)
    
    return {
        'pre_contrast': pre_texture[0],
        'pre_homogeneity': pre_texture[1],
        'pre_energy': pre_texture[2],
        'pre_correlation': pre_texture[3],
        'post_contrast': post_texture[0],
        'post_homogeneity': post_texture[1],
        'post_energy': post_texture[2],
        'post_correlation': post_texture[3]
    }

def synapse_boundary(pre_mask, post_mask):
    """Calculate boundary characteristics that might distinguish E/I synapses."""
    # Get boundaries
    pre_boundary = measure.find_contours(pre_mask, 0.5)[0]
    post_boundary = measure.find_contours(post_mask, 0.5)[0]
    
    # Calculate boundary complexity
    pre_perimeter = len(pre_boundary)
    post_perimeter = len(post_boundary)
    
    # Calculate convex hull and solidity
    pre_props = measure.regionprops(pre_mask.astype(int))[0]
    post_props = measure.regionprops(post_mask.astype(int))[0]
    
    return {
        'pre_perimeter': pre_perimeter,
        'post_perimeter': post_perimeter,
        'pre_solidity': pre_props.solidity,
        'post_solidity': post_props.solidity,
        'pre_eccentricity': pre_props.eccentricity,
        'post_eccentricity': post_props.eccentricity
    } 