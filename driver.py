import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame

# ---------------------- CONFIGURATION ---------------------- #
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSEC_FRAMES = 15

BRIGHTNESS_THRESHOLD = 80  # Below this = night mode

COUNTER = 0
yawn_counter = 0
alarm_on = False

# ---------------------- ALARM SETUP ---------------------- #
pygame.mixer.init()

try:
    pygame.mixer.music.load("alarm.wav")  # Make sure alarm.wav exists
except pygame.error:
    print("[ERROR] Could not load alarm.wav! Place a valid alarm.wav file in the same folder.")
    exit()

def start_alarm():
    global alarm_on
    if not alarm_on:
        pygame.mixer.music.play(-1)  # Loop alarm
        alarm_on = True

def stop_alarm():
    global alarm_on
    if alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False

# ---------------------- ASPECT RATIO FUNCTIONS ---------------------- #
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[15], mouth[17])
    C = dist.euclidean(mouth[12], mouth[16])
    return (A + B) / (2.0 * C)

# ---------------------- LOAD MODELS ---------------------- #
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# ---------------------- VIDEO STREAM ---------------------- #
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect brightness
    brightness = np.mean(gray)

    # Night mode enhancement
    if brightness < BRIGHTNESS_THRESHOLD:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        cv2.putText(frame, "Night Mode: ON", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Night Mode: OFF", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        is_drowsy = False
        is_yawning = False

        # Eye aspect ratio check
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                is_drowsy = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        # Yawn check
        if mar > MOUTH_AR_THRESH:
            yawn_counter += 1
            if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                is_yawning = True
                cv2.putText(frame, "YAWNING!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            yawn_counter = 0

        # Trigger alarm
        if is_drowsy or is_yawning:
            start_alarm()
        else:
            stop_alarm()

        # Display values
        cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Driver Monitor", frame)

    # Quit on "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
