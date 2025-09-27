import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import tensorflow as tf
import time

class GestureInferenceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        print(f"--- Using TensorFlow Version: {tf.__version__} ---")

        # --- Configuration ---
        self.VIDEO_WIDTH = 640
        self.VIDEO_HEIGHT = 480
        self.SEQUENCE_LENGTH = 75
        self.CAPTURE_DURATION = 3.0 # MODIFIED: Define capture duration in seconds
        self.MODEL_PATH = 'gesture_recognition_model.h5'
        self.DATA_PATH = "recorded_data"
        self.CONFIDENCE_THRESHOLD = 0.7

        # --- State Variables ---
        self.is_capturing = False
        self.capture_sequence = []
        self.capture_start_time = 0 # ADDED: For accurate timing

        # --- Load Model and Actions ---
        try:
            print("Loading gesture recognition model...")
            self.model = tf.keras.models.load_model(self.MODEL_PATH)
            self.actions = sorted([d for d in os.listdir(self.DATA_PATH) if os.path.isdir(os.path.join(self.DATA_PATH, d))])
            print(f"Model loaded successfully. Actions: {self.actions}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.window.destroy()
            return
        
        # --- MediaPipe Initialization ---
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # --- OpenCV Video Capture ---
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.VIDEO_WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.VIDEO_HEIGHT)

        # --- Tkinter UI Elements ---
        self.canvas = tk.Canvas(window, width=self.VIDEO_WIDTH, height=self.VIDEO_HEIGHT, bg="black")
        self.canvas.pack(pady=10)
        
        self.status_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.prediction_label = tk.Label(window, text="Press 'Capture' to start", font=self.status_font, fg="blue", height=2)
        self.prediction_label.pack(pady=10)

        self.btn_capture = tk.Button(window, text=f"Capture & Predict ({self.CAPTURE_DURATION:.1f} seconds)", width=30, command=self.start_capture, font=("Helvetica", 12))
        self.btn_capture.pack(pady=10)
        
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_capture(self):
        """Starts the capture process based on duration."""
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_sequence = []
            self.capture_start_time = time.time() # Start the timer
            self.btn_capture.config(state=tk.DISABLED)
            self.prediction_label.config(text="Capturing...", fg="red")

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def draw_styled_landmarks(self, image, results):
        # The model still USES the pose data for inference, we just don't draw it.
        #mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, ...)
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                     mp.solutions.drawing_utils.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                     mp.solutions.drawing_utils.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                     mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                     mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            self.draw_styled_landmarks(image, results)
            
            # --- MODIFIED: Capture logic now uses a precise timer ---
            if self.is_capturing:
                elapsed_time = time.time() - self.capture_start_time
                self.prediction_label.config(text=f"Capturing... {elapsed_time:.1f}s / {self.CAPTURE_DURATION:.1f}s")

                if elapsed_time < self.CAPTURE_DURATION:
                    # Continue capturing frames
                    keypoints = self.extract_keypoints(results)
                    self.capture_sequence.append(keypoints)
                else:
                    # --- Time is up, process the captured frames ---
                    self.is_capturing = False
                    self.btn_capture.config(state=tk.NORMAL)
                    
                    num_captured = len(self.capture_sequence)
                    if num_captured > 0:
                        # Resample the captured frames to exactly SEQUENCE_LENGTH
                        indices = np.linspace(0, num_captured - 1, self.SEQUENCE_LENGTH, dtype=int)
                        final_sequence = np.array(self.capture_sequence)[indices]

                        # Run inference
                        prediction_input = np.expand_dims(final_sequence, axis=0)
                        res = self.model.predict(prediction_input, verbose=0)[0]
                        
                        predicted_action = self.actions[np.argmax(res)]
                        confidence = res[np.argmax(res)]
                        
                        if confidence > self.CONFIDENCE_THRESHOLD:
                            final_text = f"{predicted_action} ({confidence*100:.1f}%)"
                            self.prediction_label.config(text=final_text, fg="green")
                        else:
                            self.prediction_label.config(text="Uncertain", fg="orange")
                    else:
                        self.prediction_label.config(text="Capture failed", fg="red")

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # MODIFIED: Reduced delay for more responsive frame capture
        self.window.after(10, self.update_frame)

    def on_closing(self):
        print("Closing application...")
        self.vid.release()
        self.window.destroy()

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureInferenceApp(root, "Live Gesture Recognition")
    root.mainloop()


