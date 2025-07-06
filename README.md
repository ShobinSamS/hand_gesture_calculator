# 📷 Hand Gesture Calculator

This is a **Streamlit** app that transforms your laptop's camera into a **real-time hand gesture calculator**! It leverages a trained **YOLO object detection model** to interpret your hand movements as calculator inputs.

---

## ✨ Features

* **Real-time Gesture Detection:** Utilizes YOLO to detect hand gestures instantly from your live video feed.
* **Camera Integration:** Works seamlessly with your laptop's built-in camera.
* **Dynamic Calculator Interface:** The on-screen calculator updates in real-time based on your detected gestures.
* **Responsive UI:** A thread-safe design ensures the user interface remains smooth and responsive during operation.
* **Comprehensive Functionality:** Supports numbers `0-9`, common operators (`+`, `-`, `×`, `÷`), a clear button (`C`), and an equals button (`=`).

---

## 🚀 Get Started

Follow these steps to set up and run the Hand Gesture Calculator:

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/hand_gesture_calculator.git](https://github.com/yourusername/hand_gesture_calculator.git)
    cd hand_gesture_calculator
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

After installing the dependencies, run the Streamlit app:

```bash
streamlit run app.py
```

### 📌 Notes
Lighting: Ensure you have good lighting conditions for optimal gesture detection performance.

Model Path: Verify that your YOLO model file is placed in the correct directory as expected by the application.

Dependencies: Remember to install all required dependencies using pip install -r requirements.txt before attempting to run the app.
