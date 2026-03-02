# Shared constants for the YOLO / OCTRON training pipeline

# Color codes for train-mode indicators (segment vs. detect)
TASK_COLORS = {
    'segment': '#7e56c2',
    'detect':  '#5f9bdb',
}

# Default region properties extracted from segmentation masks via regionprops.
# These are passed to skimage.measure.regionprops_table.
# 'centroid' and 'label' are always included internally and should not be listed here.
# See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
DEFAULT_REGION_PROPERTIES = (
    'area',
    'eccentricity',
    'solidity',
    'orientation',
)
