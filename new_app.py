import os
import time
import joblib
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="Neurological Eye Movement Screening",
    layout="centered"
)

st.title("Neurological Eye Movement Screening System")

st.write(
    "Tracks eye movement using webcam and predicts "
    "Normal or Abnormal gaze patterns using Logistic Regression."
)

st.info(
    "This is a screening demo and not a medical diagnosis."
)

# ==========================================
# MODEL FILES
# ==========================================
MODEL_PATH = "logistic_eye_movement_model.pkl"
SCALER_PATH = "scaler.pkl"

# ==========================================
# SETTINGS
# ==========================================
CAPTURE_SECONDS = 10
CALIBRATION_SECONDS = 2
MIN_POINTS = 25

# BALANCED SETTINGS
SMOOTH_WINDOW = 13
VELOCITY_THRESHOLD = 0.15

# ==========================================
# FEATURES USED IN TRAINING
# ==========================================
FEATURE_COLUMNS = [
    "Fixation_Count",
    "Saccade_Count",
    "MeanVelocity",
    "Velocity_STD"
]

# ==========================================
# LOAD MODEL + SCALER
# ==========================================
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"Scaler not found: {SCALER_PATH}")
    st.stop()

try:

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

except Exception as e:

    st.error(f"Error loading model/scaler: {e}")

    st.stop()

st.success("Logistic Regression model loaded successfully")

# ==========================================
# MEDIAPIPE SETUP
# ==========================================
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_point(landmarks, idx, w, h):

    lm = landmarks[idx]

    return np.array(
        [lm.x * w, lm.y * h],
        dtype=np.float32
    )


def get_iris_center(landmarks, indices, w, h):

    pts = np.array(
        [get_point(landmarks, i, w, h) for i in indices],
        dtype=np.float32
    )

    return pts.mean(axis=0)


def safe_distance(p1, p2):

    return float(np.linalg.norm(p1 - p2))


def normalize_gaze(
    left_center,
    right_center,
    left_outer,
    left_inner,
    right_outer,
    right_inner
):

    face_eye_center = (
        left_center + right_center
    ) / 2.0

    left_eye_width = max(
        safe_distance(left_outer, left_inner),
        1e-6
    )

    right_eye_width = max(
        safe_distance(right_outer, right_inner),
        1e-6
    )

    avg_eye_width = (
        left_eye_width + right_eye_width
    ) / 2.0

    norm_x = face_eye_center[0] / avg_eye_width

    norm_y = face_eye_center[1] / avg_eye_width

    return norm_x, norm_y


def apply_calibration(
    raw_points,
    baseline_x,
    baseline_y
):

    calibrated = []

    for t, x, y in raw_points:

        calibrated.append([
            t,
            x - baseline_x,
            y - baseline_y
        ])

    return calibrated


def smooth_series(series, window=13):

    return series.rolling(
        window=window,
        min_periods=1
    ).mean()

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def compute_features(gaze_points):

    df = pd.DataFrame(
        gaze_points,
        columns=["time", "x", "y"]
    ).copy()

    if len(df) < 2:
        return None

    # ======================================
    # SMOOTHING
    # ======================================
    df["x"] = smooth_series(
        df["x"],
        SMOOTH_WINDOW
    )

    df["y"] = smooth_series(
        df["y"],
        SMOOTH_WINDOW
    )

    # ======================================
    # PREVIOUS VALUES
    # ======================================
    df["prev_x"] = df["x"].shift(1)

    df["prev_y"] = df["y"].shift(1)

    df["prev_time"] = df["time"].shift(1)

    # ======================================
    # DISTANCE
    # ======================================
    df["distance"] = np.sqrt(
        (df["x"] - df["prev_x"]) ** 2 +
        (df["y"] - df["prev_y"]) ** 2
    )

    df["distance"] = df["distance"].fillna(0)

    # ======================================
    # TIME DIFFERENCE
    # ======================================
    df["time_diff"] = (
        df["time"] - df["prev_time"]
    )

    df["time_diff"] = df["time_diff"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ======================================
    # VELOCITY
    # ======================================
    df["velocity"] = np.where(
        (df["time_diff"].notna()) &
        (df["time_diff"] > 0),

        df["distance"] / df["time_diff"],

        0
    )

    df["velocity"] = df["velocity"].replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    # ======================================
    # SMOOTH VELOCITY
    # ======================================
    df["velocity"] = smooth_series(
        df["velocity"],
        SMOOTH_WINDOW
    )

    # ======================================
    # REMOVE EXTREME SPIKES
    # ======================================
    df["velocity"] = df["velocity"].clip(
        lower=0,
        upper=0.35
    )

    # ======================================
    # MOVEMENT CLASSIFICATION
    # ======================================
    df["movement_type"] = np.where(
        df["velocity"] <= VELOCITY_THRESHOLD,
        "Fixation",
        "Saccade"
    )

    # ======================================
    # COUNTS
    # ======================================
    fixation_count = int(
        (df["movement_type"] == "Fixation").sum()
    )

    saccade_count = int(
        (df["movement_type"] == "Saccade").sum()
    )

    # ======================================
    # VELOCITY FEATURES
    # ======================================
    mean_velocity = float(
        df["velocity"].mean()
    )

    vel_std = df["velocity"].std()

    velocity_std = float(
        vel_std
    ) if not np.isnan(vel_std) else 0.0

    # ======================================
    # FIXATION RATIO
    # ======================================
    total = fixation_count + saccade_count

    fixation_ratio = (
        fixation_count / total
        if total > 0 else 0
    )

    # ======================================
    # FINAL FEATURES
    # ======================================
    features = {
        "Fixation_Count": fixation_count,
        "Saccade_Count": saccade_count,
        "MeanVelocity": mean_velocity,
        "Velocity_STD": velocity_std,
        "Fixation_Ratio": fixation_ratio
    }

    return features

# ==========================================
# BUILD INPUT
# ==========================================
def build_input_df(features):

    input_df = pd.DataFrame(
        [[
            features["Fixation_Count"],
            features["Saccade_Count"],
            features["MeanVelocity"],
            features["Velocity_STD"]
        ]],
        columns=FEATURE_COLUMNS
    )

    input_scaled = scaler.transform(
        input_df
    )

    return input_scaled

# ==========================================
# LOGISTIC PREDICTION
# ==========================================
def predict_logistic(input_scaled):

    probs = model.predict_proba(
        input_scaled
    )[0]

    normal_prob = float(probs[0])

    abnormal_prob = float(probs[1])

    prediction = model.predict(
        input_scaled
    )[0]

    return prediction, normal_prob, abnormal_prob

# ==========================================
# FINAL DECISION
# ==========================================
def final_decision(
    prediction,
    normal_prob,
    abnormal_prob,
    features
):

    fixation_count = features["Fixation_Count"]

    saccade_count = features["Saccade_Count"]

    mean_velocity = features["MeanVelocity"]

    fixation_ratio = features["Fixation_Ratio"]

    # ======================================
    # STRONG ABNORMAL
    # ======================================
    if (
        fixation_ratio < 0.35
        and mean_velocity > 0.18
    ):

        return 1

    # ======================================
    # STRONG NORMAL
    # ======================================
    if (
        fixation_ratio > 0.55
        and mean_velocity < 0.20
    ):

        return 0

    # ======================================
    # SECONDARY NORMAL
    # ======================================
    if (
        fixation_count > saccade_count
        and mean_velocity < 0.15
    ):

        return 0

    # ======================================
    # SECONDARY ABNORMAL
    # ======================================
    if (
        saccade_count > fixation_count * 1.5
        and mean_velocity > 0.15
    ):

        return 1

    # ======================================
    # OTHERWISE MODEL
    # ======================================
    return prediction

# ==========================================
# MAIN BUTTON
# ==========================================
if st.button("Start Eye Tracking"):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error("Webcam not detected")

        st.stop()

    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    st.info(
        f"Calibration for {CALIBRATION_SECONDS} seconds. "
        f"Please look at the center."
    )

    frame_placeholder = st.empty()

    status_placeholder = st.empty()

    calibration_points = []

    tracking_points = []

    start_time = time.time()

    phase = "calibration"

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            st.error("Unable to read webcam frame")

            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        current_time = time.time()

        elapsed_total = current_time - start_time

        if results.multi_face_landmarks:

            landmarks = (
                results.multi_face_landmarks[0].landmark
            )

            left_center = get_iris_center(
                landmarks,
                LEFT_IRIS,
                w,
                h
            )

            right_center = get_iris_center(
                landmarks,
                RIGHT_IRIS,
                w,
                h
            )

            left_outer = get_point(
                landmarks,
                LEFT_EYE_OUTER,
                w,
                h
            )

            left_inner = get_point(
                landmarks,
                LEFT_EYE_INNER,
                w,
                h
            )

            right_outer = get_point(
                landmarks,
                RIGHT_EYE_OUTER,
                w,
                h
            )

            right_inner = get_point(
                landmarks,
                RIGHT_EYE_INNER,
                w,
                h
            )

            norm_x, norm_y = normalize_gaze(
                left_center,
                right_center,
                left_outer,
                left_inner,
                right_outer,
                right_inner
            )

            if elapsed_total < CALIBRATION_SECONDS:

                calibration_points.append([
                    current_time,
                    norm_x,
                    norm_y
                ])

                phase = "calibration"

                remaining = (
                    CALIBRATION_SECONDS -
                    elapsed_total
                )

                label = (
                    f"Calibration... {remaining:.1f}s"
                )

                color = (0, 255, 255)

            else:

                phase = "tracking"

                tracking_elapsed = (
                    elapsed_total -
                    CALIBRATION_SECONDS
                )

                remaining = (
                    CAPTURE_SECONDS -
                    tracking_elapsed
                )

                tracking_points.append([
                    current_time,
                    norm_x,
                    norm_y
                ])

                label = (
                    f"Tracking... {remaining:.1f}s"
                )

                color = (0, 255, 0)

            cv2.circle(
                frame,
                tuple(left_center.astype(int)),
                3,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                tuple(right_center.astype(int)),
                3,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        frame_placeholder.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            channels="RGB",
            use_container_width=True
        )

        status_placeholder.write(
            f"Phase: {phase} | "
            f"Calibration points: {len(calibration_points)} | "
            f"Tracking points: {len(tracking_points)}"
        )

        if elapsed_total >= (
            CALIBRATION_SECONDS +
            CAPTURE_SECONDS
        ):
            break

    cap.release()

    face_mesh.close()

    # ==========================================
    # VALIDATION
    # ==========================================
    if (
        len(tracking_points) < MIN_POINTS or
        len(calibration_points) < 5
    ):

        st.warning(
            "Not enough gaze data collected. Please try again."
        )

        st.stop()

    # ==========================================
    # CALIBRATION
    # ==========================================
    calib_df = pd.DataFrame(
        calibration_points,
        columns=["time", "x", "y"]
    )

    baseline_x = float(
        calib_df["x"].mean()
    )

    baseline_y = float(
        calib_df["y"].mean()
    )

    calibrated_tracking_points = apply_calibration(
        tracking_points,
        baseline_x,
        baseline_y
    )

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    features = compute_features(
        calibrated_tracking_points
    )

    if features is None:

        st.error("Feature extraction failed")

        st.stop()

    # ==========================================
    # MODEL INPUT
    # ==========================================
    input_scaled = build_input_df(
        features
    )

    # ==========================================
    # MODEL PREDICTION
    # ==========================================
    prediction, normal_prob, abnormal_prob = predict_logistic(
        input_scaled
    )

    # ==========================================
    # FINAL DECISION
    # ==========================================
    final_pred = final_decision(
        prediction,
        normal_prob,
        abnormal_prob,
        features
    )

    # ==========================================
    # OUTPUT
    # ==========================================
    st.subheader("Extracted Features")

    for key, value in features.items():

        if isinstance(value, float):

            st.write(
                f"{key}: {value:.5f}"
            )

        else:

            st.write(
                f"{key}: {value}"
            )

    

    

    st.subheader("Result")

    if final_pred == 1:

        st.error(
            "Abnormal Eye Movement Detected"
        )

    else:

        st.success(
            "Normal Eye Movement Detected"
        )