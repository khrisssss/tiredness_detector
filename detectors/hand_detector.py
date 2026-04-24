import mediapipe as mp

class HandDetector:
    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_hands(self, frame_rgb):
        results = self.hands.process(frame_rgb)

        hand_detected = False
        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks_list = results.multi_hand_landmarks

        return hand_detected, hand_landmarks_list

    def draw_hands(self, frame, hand_landmarks_list):
        for hand_landmarks in hand_landmarks_list:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

    def close(self):
        self.hands.close()