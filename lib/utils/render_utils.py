import numpy as np


def add_seal_vertex(vertex):
    circle_v_id = np.array(
        [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
        dtype=np.int32,
    )
    center = (vertex[circle_v_id, :]).mean(0)
    vertex = np.vstack([vertex, center])
    return vertex


# Make sure render only contains segmentation color
def replace_non_matching_pixels(image, specific_colors):
    image = image.copy()
    specific_colors = np.array(specific_colors, dtype=np.uint8)
    
    # Create a mask initialized to all False
    matches = np.zeros(image.shape[:2], dtype=bool)
    
    # Check if each pixel matches any of the specific colors
    for color in specific_colors:
        matches |= np.all(image == color, axis=-1)
    
    # Create a copy of the image to modify
    modified_image = image.copy()
    
    # Set non-matching pixels to black
    modified_image[~matches] = [0, 0, 0]
    
    return modified_image


# Make sure render only contains segmentation color
def replace_pixels_to_seg(image, color_dict):
    matches = np.zeros(image.shape[:2], dtype=bool)
    modified_image = image.copy()
    
    for color in [*color_dict]:
        matches = np.all(image == color, axis=-1)

        new_color = color_dict[color]
        modified_image[matches] = [new_color, new_color, new_color]
    
    return modified_image