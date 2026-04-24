import math

def compute_head_tilt_angle(landmarks):
    left = landmarks[234]
    right = landmarks[454]

    dx = right.x - left.x
    dy = right.y - left.y

    angle = math.degrees(math.atan2(dy, dx))
    return angle