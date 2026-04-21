import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def euclidean_distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def eye_aspect_ratio(landmarks, eye_indices):
    """
    EAR = (vertical_1 + vertical_2) / (2 * horizontal)
    eye_indices should contain 6 landmark indices:
    [left_corner, upper_1, upper_2, right_corner, lower_1, lower_2]
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = euclidean_distance(p2, p5)
    vertical_2 = euclidean_distance(p3, p6)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


# MediaPipe eye landmark indices
# These points work well enough for a first version
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 0, 255),  # red
    thickness=1,
    circle_radius=1     # small circles for landmarks
)

connection_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0)  # green
)

# Alert settings
EYE_CLOSED_THRESHOLD = 0.22
EYE_CLOSED_SECONDS = 2.0

eyes_closed_start_time = None

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    


    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(     #face landmarks
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=connection_spec

                    )       

            cv2.imshow("Step 2 - Face Mesh", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()