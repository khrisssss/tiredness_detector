import cv2
import mediapipe as mp
import time
from detectors.mouth_detector import mouth_aspect_ratio
from detectors.head_pose import compute_head_tilt_angle
from detectors.eye_detector import eye_aspect_ratio
from detectors.head_pose import compute_head_tilt_angle
from detectors.phone_detector import PhoneDetector
from detectors.hand_detector import HandDetector

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 13, 312, 291, 14, 17]

def main():
    cap = cv2.VideoCapture(0)
    phone_detector = PhoneDetector(model_path="yolov8n.pt", confidence_threshold=0.45)
    hand_detector = HandDetector(max_num_hands=2)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ---------- Thresholds ----------
    EYE_CLOSED_THRESHOLD = 0.45
    EYE_CLOSED_SECONDS = 2.0

    HEAD_TILT_THRESHOLD = 10
    HEAD_TILT_SECONDS = 3.0

    MOUTH_OPEN_THRESHOLD = 0.30
    MOUTH_OPEN_SECONDS = 3.0
    

    PHONE_DETECTED_SECONDS = 0.20
    PHONE_GRACE_PERIOD = 0.2
    
    HAND_DETECTED_SECONDS = 0.5

    # ---------- Timers ----------
    eyes_closed_start_time = None
    head_tilt_start_time = None
    mouth_open_start_time = None
    phone_detected_start_time = None
    phone_last_seen_time = None
    hand_detected_start_time = None

    #drawing_spec = mp_drawing.DrawingSpec(
    #    color=(0, 0, 255),
    #    thickness=1,
    #    circle_radius=1
    #)

    #connection_spec = mp_drawing.DrawingSpec(
    #    color=(0, 255, 0),
    #    thickness=1
    #)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            avg_ear = 0.0
            head_angle = 0.0
            
            eye_alert = ""
            head_alert = ""
            
            mouth_ratio = 0.0
            mouth_alert = ""

            phone_alert = ""
            phone_box = None
            
            hand_detected = False
            hand_message = ""
            hand_landmarks_list = []
            hand_visible_elapsed = 0.0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )

                    # ---------- EYE DETECTION ----------
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < EYE_CLOSED_THRESHOLD:
                        if eyes_closed_start_time is None:
                            eyes_closed_start_time = time.time()

                        eye_elapsed = time.time() - eyes_closed_start_time

                        cv2.putText(
                            frame,
                            f"Eyes closed time: {eye_elapsed:.2f}s",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                        if eye_elapsed >= EYE_CLOSED_SECONDS:
                            eye_alert = "ALERT!"
                    else:
                        eyes_closed_start_time = None

                    # ---------- HEAD TILT DETECTION ----------
                    head_angle = compute_head_tilt_angle(landmarks)

                    if abs(head_angle) > HEAD_TILT_THRESHOLD:
                        if head_tilt_start_time is None:
                            head_tilt_start_time = time.time()

                        head_elapsed = time.time() - head_tilt_start_time

                        cv2.putText(
                            frame,
                            f"Tilt time: {head_elapsed:.2f}s",
                            (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2
                        )

                        if head_elapsed >= HEAD_TILT_SECONDS:
                            head_alert = "ALERT!"
                    else:
                        head_tilt_start_time = None

                    # ---------- MOUTH OPEN DETECTION ----------
                    mouth_ratio = mouth_aspect_ratio(landmarks, MOUTH)

                    if mouth_ratio > MOUTH_OPEN_THRESHOLD:
                        if mouth_open_start_time is None:
                            mouth_open_start_time = time.time()

                        mouth_elapsed = time.time() - mouth_open_start_time

                        cv2.putText(
                            frame,
                            f"Mouth open time: {mouth_elapsed:.2f}s",
                            (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                        if mouth_elapsed >= MOUTH_OPEN_SECONDS:
                            mouth_alert = "ALERT !"
                    else:
                        mouth_open_start_time = None

                    # ---------- PHONE DETECTION ----------
                    phone_found, phone_box = phone_detector.detect_phone(frame)

                    current_time = time.time()

                    if phone_found:
                        phone_last_seen_time = current_time

                        if phone_detected_start_time is None:
                            phone_detected_start_time = current_time

                        phone_elapsed = current_time - phone_detected_start_time

                        if phone_elapsed >= PHONE_DETECTED_SECONDS: 
                            phone_alert = "ALERT !"

                        if phone_box is not None:
                            x1, y1, x2, y2, conf, label = phone_box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(
                                frame,
                                f"{label} {conf:.2f}",
                                (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2
                            )
                    else:
                        if phone_last_seen_time is not None:
                            if current_time - phone_last_seen_time > PHONE_GRACE_PERIOD:
                                phone_detected_start_time = None


                    # ---------- HAND DETECTION ----------
                    hand_detected, hand_landmarks_list = hand_detector.detect_hands(rgb_frame)
                    hand_visible_elapsed = 0.0

                    if hand_detected:
                        if hand_detected_start_time is None:
                            hand_detected_start_time = current_time

                        hand_visible_elapsed = current_time - hand_detected_start_time

                        hand_detector.draw_hands(frame, hand_landmarks_list)

                        if hand_visible_elapsed >= HAND_DETECTED_SECONDS:
                            hand_message = "ALERT hands on the wheel!"
                    else:
                        hand_detected_start_time = None

            else:
                eyes_closed_start_time = None
                head_tilt_start_time = None
                mouth_open_start_time = None
                phone_detected_start_time = None
                hand_detected_start_time = None
                hand_detected_start_time = None
            # ---------- DISPLAY INFO ----------
            


            

            if eye_alert:
                cv2.putText(
                    frame,
                    eye_alert,
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            if head_alert:
                cv2.putText(
                    frame,
                    head_alert,
                    (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )
            if mouth_alert:
                cv2.putText(
                    frame,
                    mouth_alert,
                    (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            if phone_alert:
                cv2.putText(
                    frame,
                    phone_alert,
                    (30, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

    
            if hand_detected and hand_visible_elapsed < HAND_DETECTED_SECONDS:
                cv2.putText(
                    frame,
                    f"Hand detected: {hand_visible_elapsed:.2f}s",
                    (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            if hand_message:
                cv2.putText(
                    frame,
                    hand_message,
                    (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    3
                )
            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()