"""
Landmark extraction and normalization functions.

Matches the processing pipeline used during data collection.
"""

import copy
import itertools


def calc_landmark_list(image, landmarks):
    """
    Extract landmark coordinates from MediaPipe hand landmarks.
    
    Args:
        image: Frame image (used for width/height)
        landmarks: MediaPipe hand landmarks object
        
    Returns:
        List of [x, y] coordinate pairs for 21 hand landmarks
    """
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point


def pre_process_landmark(landmark_list):
    """
    Normalize landmarks to relative coordinates.
    
    Normalization process:
    1. Translate relative to wrist (landmark 0)
    2. Flatten to 1D array
    3. Scale by max absolute value
    
    Args:
        landmark_list: List of [x, y] coordinate pairs (21 landmarks)
        
    Returns:
        List of 42 normalized coordinate values in range [-1, 1]
    """
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates (relative to wrist)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    # Flatten
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value > 0:
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    
    return temp_landmark_list