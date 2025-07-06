import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import time
import queue
from ultralytics import YOLO
import threading

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

# ✅ Config
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
IMAGE_SIZE = 416

# Load the YOLO model for gesture detection
model = YOLO(MODEL_PATH) 

# App Layout
st.set_page_config(layout="wide")
st.title("📷 Hand Gesture Calculator")

# --- Step 1: Create a thread-safe queue ---
# This queue will safely pass data from the video processing thread to the main Streamlit thread.
@st.cache_resource
def get_queue():
    return queue.Queue()

result_queue = get_queue()

# ---------- GLOBAL press FUNCTION ----------
# This function handles the calculator's logic.
def press(btn):
    if btn in ["÷", "×", "−", "＋"]:
        if st.session_state.get("previous_operation") in ["÷", "×", "−", "＋"]:
            return
    st.session_state.previous_operation = btn

    if btn == "C":
        st.session_state.calc_display = ""
    elif btn == "=":
        try:
            expression = st.session_state.calc_display.replace("×", "*").replace("÷", "/").replace("＋", "+").replace("−", "-")
            result = eval(expression)
            st.session_state.calc_display = str(result)
        except Exception:
            st.session_state.calc_display = "Error"
    else:
        st.session_state.calc_display += btn

# Initialize variables in Streamlit's session state
if "calc_display" not in st.session_state:
    st.session_state.calc_display = ""
if "previous_operation" not in st.session_state:
    st.session_state.previous_operation = None
if "last_gesture_display" not in st.session_state:
    st.session_state.last_gesture_display = "Click START to begin"


# ---------- UI LAYOUT ----------
left_col, right_col = st.columns(2)


# ---------- RIGHT SIDE: Calculator UI ----------
with right_col:
    st.header("Calculator")
    st.markdown(f"""
    <div style="
        font-size: 2rem; color: black; border: 1px solid gray; padding: 1rem;
        border-radius: 0.5rem; background-color: #f0f0f0; text-align: right;
        width: 100%; min-height: 3rem; line-height: 1.5rem;  
        padding: 0.5rem 1rem; overflow-wrap: break-word; white-space: pre-wrap;
        word-break: break-word; margin-bottom: 1rem; box-sizing: border-box;">
        {st.session_state.calc_display or "0"}
    </div>
    """, unsafe_allow_html=True)

    button_rows = [
       ["7", "8", "9", "÷"],
       ["4", "5", "6", "×"],
       ["1", "2", "3", "−"],
       ["C", "0", "=", "＋"],
    ]
    for row in button_rows:
        cols = st.columns(4)
        for i, btn in enumerate(row):
            cols[i].button(btn, key=f"btn-{btn}", on_click=press, args=(btn,), use_container_width=True)


# ---------- LEFT SIDE: Camera and Gesture Detection ----------
with left_col:
    st.header("Camera Feed")
    def yolo_processing(img):
        try:
            results = model.predict(source=img, imgsz=IMAGE_SIZE)
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                return None

            # Extract confidences and class ids
            confidences = r.boxes.conf.cpu().numpy().flatten()
            class_ids    = r.boxes.cls.cpu().numpy().flatten().astype(int)

            # Find highest-confidence detection
            top_idx = confidences.argmax()
            top_cls = class_ids[top_idx]
            top_conf = confidences[top_idx]

            # Map to label name
            label_name = model.names[top_cls]

            
            if label_name == "c":
                label_name="C"
            # Put the detected gesture into the queue to be read by the main thread
            gesture=label_name
            result_queue.put(gesture)
                
                # Draw visual feedback on the frame
            cv2.putText(img, f"Detected: {label_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            pass
    # The Video Processor class (runs in a background thread)
    class GestureProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            super().__init__()
            self.detection_interval = 3.0  # seconds
            self.last_detection_time = time.time()
        
            
            
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
            except Exception as e:
                return frame  # skip bad frames
            # time.sleep(0.5)  # Simulate processing delay
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_interval:
                self.last_detection_time = current_time
                yolo_thread = threading.Thread(target=yolo_processing, args=(img,),daemon=True)
                yolo_thread.start()
                
                
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="stream", 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=GestureProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

    gesture_placeholder = st.empty()
    gesture_placeholder.markdown(f"## {st.session_state.last_gesture_display}")

    # This is the main polling loop for our app
    if ctx.state.playing:
        # Reset display text when starting
        st.session_state.last_gesture_display = " "
        
        # Process all gestures that have accumulated in the queue
        while not result_queue.empty():
            gesture = result_queue.get_nowait()
            st.session_state.last_gesture_display = " "
            press(gesture)
        
        # This is the key to the fix:
        # We add a small delay and then force the script to rerun.
        # This creates a continuous loop that checks the queue for new gestures.
        time.sleep(0.1)
        st.rerun()



