import numpy as np
from skimage import measure

def synapse_size(pre_mask, post_mask):
    """Calculate synapse size metrics."""
    synapse_mask = pre_mask | post_mask
    synapse_area = np.sum(synapse_mask)
    pre_area = np.sum(pre_mask)
    post_area = np.sum(post_mask)
    pre_post_ratio = pre_area / post_area
    return synapse_area, pre_area, post_area, pre_post_ratio

def synapse_shape(pre_mask, post_mask):
    """Calculate synapse shape metrics."""
    synapse_mask = pre_mask | post_mask
    perimeter = measure.perimeter(synapse_mask)
    circularity = 4 * np.pi * np.sum(synapse_mask) / (perimeter ** 2)
    return perimeter, circularity

def synapse_intensity(image, pre_mask, post_mask):
    """Calculate synapse intensity metrics."""
    synapse_mask = pre_mask | post_mask
    synapse_intensities = image[synapse_mask]
    mean_intensity = np.mean(synapse_intensities)
    median_intensity = np.median(synapse_intensities)
    std_intensity = np.std(synapse_intensities)
    return mean_intensity, median_intensity, std_intensity

def synapse_distance(pre_mask, post_mask):
    """Calculate distance between pre- and post-synaptic regions."""
    pre_centroid = measure.centroid(pre_mask)
    post_centroid = measure.centroid(post_mask)
    distance = np.sqrt((pre_centroid[0] - post_centroid[0]) ** 2 +
                       (pre_centroid[1] - post_centroid[1]) ** 2)
    return distance

def synapse_colocalization(pre_mask, post_mask):
    """Calculate synapse co-localization metrics."""
    intersection = pre_mask & post_mask
    union = pre_mask | post_mask
    jaccard_index = np.sum(intersection) / np.sum(union)
    pre_overlap = np.sum(intersection) / np.sum(pre_mask)
    post_overlap = np.sum(intersection) / np.sum(post_mask)
    return jaccard_index, pre_overlap, post_overlap

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
    return major_axis_length, minor_axis_length 