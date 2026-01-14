import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GSAM_PATH = os.path.join(PROJECT_ROOT, "Grounded_SAM_2")
if GSAM_PATH not in sys.path:
    sys.path.insert(0, GSAM_PATH)

import cv2
from models.gdino_loader import load_grounding_dino
from models.sam2_loader import load_sam2_models
from tools.grounded_sam_inference import run_realtime_sam
import time

DEVICE = "cuda"
URL = "placeholder" 
TEXT_PROMPT = "placeholder"
INFERENCE_INTERVAL = 30


gdino_model = load_grounding_dino(DEVICE)
sam2_image_model, sam_image_predictor, sam_video_predictor = load_sam2_models()

cap = cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def frame_generator():
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame from video source.")
            break
        yield frame
from collections import deque
fps_history = deque(maxlen=30)


prev_time = time.perf_counter()

for annotated_frame in run_realtime_sam(
    frame_generator(),
    grounding_model=gdino_model,
    image_predictor=sam_image_predictor,
    video_predictor=sam_video_predictor,
    text_prompt=TEXT_PROMPT,
    device=DEVICE,
    inference_interval=INFERENCE_INTERVAL
):
    current_time = time.perf_counter()
    elapsed = current_time - prev_time
    prev_time = current_time
    
    if elapsed > 0:
        fps = 1 / elapsed
        fps_history.append(fps)
    
    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

    cv2.putText(
        annotated_frame, 
        f"FPS: {avg_fps:.1f}", 
        (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.2, (0, 0, 0), 4, cv2.LINE_AA 
    )
    cv2.putText(
        annotated_frame, 
        f"FPS: {avg_fps:.1f}", 
        (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.2, (0, 255, 0), 2, cv2.LINE_AA 
    )

    if len(fps_history) == 30:
        print(f"\rCurrent Throughput: {avg_fps:.2f} FPS", end="")

    cv2.imshow("Processed Output Only", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n[INFO] Closing stream...")
cap.release()
cv2.destroyAllWindows()