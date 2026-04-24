import math

def euclidean_distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def mouth_aspect_ratio(landmarks, mouth_indices):
    """
    mouth_indices:
    [left_corner, upper_lip_1, upper_lip_2, right_corner, lower_lip_1, lower_lip_2]
    """
    p1 = landmarks[mouth_indices[0]]
    p2 = landmarks[mouth_indices[1]]
    p3 = landmarks[mouth_indices[2]]
    p4 = landmarks[mouth_indices[3]]
    p5 = landmarks[mouth_indices[4]]
    p6 = landmarks[mouth_indices[5]]

    vertical_1 = euclidean_distance(p2, p5)
    vertical_2 = euclidean_distance(p3, p6)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)