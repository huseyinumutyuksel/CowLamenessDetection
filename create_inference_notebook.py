
import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split("\n")]
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in content.split("\n")]
    })

add_markdown("""# 🐄 Cow Lameness Inference - Multi-Cow (v16)
**Production & Clinical Reporting**

## Objective
Apply the trained **Tri-Modal Lameness Model** to multi-cow videos and generate clinical reports.

## Pipeline
1. **DeepLabCut**: Extract pose from entire video
2. **YOLO + ByteTrack**: Detect and track individual cows
3. **Tri-Modal Classification**: Fuse Pose + VideoMAE + RAFT for each cow
4. **SAM Visualization**: Overlay colored masks (Red=Lame, Green=Healthy)
5. **Clinical Report**: Export CSV with diagnosis per cow
""")

add_markdown("## 1. Setup")
add_code("""
!pip install -q ultralytics supervision
!pip install -q timm einops transformers
!pip install -q "deeplabcut[tf]"
!pip install -q segment-anything
!pip install -q moviepy scikit-learn scipy
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

import os
from google.colab import drive
drive.mount('/content/drive')
""")

add_markdown("## 2. Phase 1: DeepLabCut Analysis")
add_code("""
import deeplabcut
import glob
import pandas as pd
import numpy as np
import shutil

DLC_PROJECT_NAME = "CowGaitAnalysis_Inf"
DLC_OWNER = "Researcher"
DLC_WORK_DIR = "/content/dlc_work"
os.makedirs(DLC_WORK_DIR, exist_ok=True)

try:
    config_path = deeplabcut.create_pretrained_project(
        DLC_PROJECT_NAME, DLC_OWNER, ["/content/drive/MyDrive/Raw_MultiCow_Videos/test_video.mp4"], 
        working_directory=DLC_WORK_DIR, copy_videos=True, analyzevideo=False, 
        model="superanimal_quadruped", videotype=".mp4"
    )
except:
    search = glob.glob(f"{DLC_WORK_DIR}/{DLC_PROJECT_NAME}*/config.yaml")
    config_path = search[0]

INPUT_VIDEO = "/content/drive/MyDrive/Raw_MultiCow_Videos/test_video.mp4"
TEMP_VIDEO = f"/content/temp_inf_{os.path.basename(INPUT_VIDEO)}"

if not os.path.exists(TEMP_VIDEO):
    shutil.copy(INPUT_VIDEO, TEMP_VIDEO)

print(f"Running DeepLabCut on {TEMP_VIDEO}...")
deeplabcut.analyze_videos(config_path, [TEMP_VIDEO], save_as_csv=False, destfolder="/content")
h5_files = glob.glob(f"/content/temp_inf_*.h5")
dlc_data_path = h5_files[0]
print(f"✅ DLC Complete: {dlc_data_path}")
""")

add_markdown("## 3. Load All Models")
add_code("""
import torch
import cv2
from ultralytics import YOLO
import supervision as sv
from transformers import VideoMAEImageProcessor, VideoMAEModel
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms.functional as F
from collections import deque
from segment_anything import sam_model_registry, SamPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# A. YOLO for Detection
yolo_model = YOLO("yolov8x.pt")
print("✅ YOLO Loaded")

# B. SAM for Segmentation
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
sam_predictor = SamPredictor(sam)
print("✅ SAM Loaded")

# C. VideoMAE
mae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
mae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
print("✅ VideoMAE Loaded")

# D. RAFT
raft_weights = Raft_Large_Weights.DEFAULT
raft_model = raft_large(weights=raft_weights).to(device).eval()
raft_tf = raft_weights.transforms()
print("✅ RAFT Loaded")

# E. Load DLC Data
df_dlc = pd.read_hdf(dlc_data_path)
scorer = df_dlc.columns.levels[0][0]
bodyparts = df_dlc.columns.levels[1]
dlc_matrix = df_dlc[scorer].values 
num_kpts = len(bodyparts)
dlc_matrix = dlc_matrix.reshape(-1, num_kpts, 3)
print(f"✅ DLC Data Loaded: {num_kpts} keypoints")

def get_dlc_points_for_box(frame_idx, box):
    if frame_idx >= len(dlc_matrix): return np.zeros(num_kpts*3)
    kpts = dlc_matrix[frame_idx] 
    valid = kpts[kpts[:,2] > 0.1]
    if len(valid) == 0: return np.zeros(num_kpts*3)
    avg_x, avg_y = np.mean(valid[:,0]), np.mean(valid[:,1])
    x1, y1, x2, y2 = box
    if x1 < avg_x < x2 and y1 < avg_y < y2:
        return kpts.flatten()
    return np.zeros(num_kpts*3)
""")

add_markdown("## 4. Load Trained Classification Model")
add_code("""
class TriModalAttention(torch.nn.Module):
    def __init__(self, pose_dim, hidden_dim=256):
        super().__init__()
        self.pose_proj = torch.nn.Linear(pose_dim, hidden_dim)
        self.mae_proj = torch.nn.Linear(768, hidden_dim)
        self.flow_proj = torch.nn.Linear(2, hidden_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim*3, nhead=4, batch_first=True, dropout=0.1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*3, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 2)
        )
    def forward(self, p, m, f):
        src = torch.cat([self.pose_proj(p), self.mae_proj(m), self.flow_proj(f)], dim=2)
        out = self.encoder_layer(src)
        return self.classifier(out.mean(dim=1))

POS_DIM = num_kpts * 3
gait_model = TriModalAttention(pose_dim=POS_DIM).to(device).eval()

# Load Trained Weights with Validation
WEIGHTS_PATH = "/content/drive/MyDrive/outputs_v16_academic/cow_gait_transformer_v16_final.pth"
if os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    
    # Check if checkpoint contains metadata (new format) or just state_dict (legacy)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with validation
        expected_pose_dim = checkpoint['pose_dim']
        
        if POS_DIM != expected_pose_dim:
            raise ValueError(
                f"❌ DIMENSION MISMATCH!\\n"
                f"Training used pose_dim={expected_pose_dim} (keypoints={expected_pose_dim//3})\\n"
                f"Inference has pose_dim={POS_DIM} (keypoints={num_kpts})\\n"
                f"Please ensure both notebooks use the same DeepLabCut model!"
            )
        
        gait_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model Loaded & Validated (pose_dim={expected_pose_dim})")
    else:
        # Legacy format (just state_dict)
        gait_model.load_state_dict(checkpoint)
        print("⚠️ Model loaded (legacy format, no validation)")
else:
    print("⚠️ WARNING: Model weights not found. Using untrained model!")
""")

add_markdown("## 5. Feature Extraction Functions")
add_code("""
def extract_videomae_features(frames_list):
    if not frames_list: return np.zeros(768)
    indices = np.linspace(0, len(frames_list)-1, 16).astype(int)
    sampled = [frames_list[i] for i in indices]
    inputs = mae_processor(list(sampled), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = mae_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

def extract_raft_flow(frame1, frame2):
    img1 = F.to_tensor(frame1).unsqueeze(0).to(device) * 255.0
    img2 = F.to_tensor(frame2).unsqueeze(0).to(device) * 255.0
    img1, img2 = raft_tf(img1, img2)
    with torch.no_grad():
        flow = raft_model(img1, img2)
    return flow[-1].mean(dim=[2,3]).cpu().numpy()[0]

class CowState:
    def __init__(self, cow_id):
        self.id = cow_id
        self.crop_buffer = deque(maxlen=30)
        self.pose_seq = deque(maxlen=30)
        self.flow_seq = deque(maxlen=30)
        self.predictions = []
        self.last_crop = None
        
    def update(self, crop, pose_vec):
        self.crop_buffer.append(crop)
        self.pose_seq.append(pose_vec)
        
        if self.last_crop is not None:
            h, w, _ = crop.shape
            prev = cv2.resize(self.last_crop, (w, h))
            f = extract_raft_flow(prev, crop)
        else:
            f = np.zeros(2)
        self.flow_seq.append(f)
        self.last_crop = crop
        
    def is_ready(self):
        return len(self.pose_seq) == 30
        
    def predict(self):
        if not self.is_ready(): return None
        
        m_vec = extract_videomae_features(list(self.crop_buffer))
        
        p_t = torch.tensor(np.array(self.pose_seq), dtype=torch.float32).unsqueeze(0).to(device)
        f_t = torch.tensor(np.array(self.flow_seq), dtype=torch.float32).unsqueeze(0).to(device)
        m_seq = np.tile(m_vec, (30, 1))
        m_t = torch.tensor(m_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = gait_model(p_t, m_t, f_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        self.predictions.append(probs)
        return probs

print("✅ Feature Extraction Ready")
""")

add_markdown("## 6. Tracking & Classification Loop")
add_code("""
import time
from tqdm import tqdm

tracker = sv.ByteTrack()
mask_annotator = sv.MaskAnnotator(opacity=0.5)
label_annotator = sv.LabelAnnotator()

OUTPUT_VIDEO = "/content/drive/MyDrive/outputs_v16_academic/inference_result_v16.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

cow_registry = {}
frame_count = 0
start_time = time.time()

print(f"\\nProcessing Video: {total_frames} frames @ {fps} FPS")
print("="*60)

# Progress bar
pbar = tqdm(total=total_frames, desc="Inference Progress")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    # Detection
    results = yolo_model(frame, classes=[19], verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker.update_with_detections(detections)
    
    # SAM Prep
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(frame_rgb)
    
    masks_list = []
    labels = []
    
    for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
        # SAM Mask
        sam_masks, _, _ = sam_predictor.predict(box=xyxy, multimask_output=False)
        mask = sam_masks[0]
        masks_list.append(mask)
        
        # Crop
        x1, y1, x2, y2 = map(int, xyxy)
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # DLC Pose
        pose_vec = get_dlc_points_for_box(frame_count-1, xyxy)
        
        # Update State
        if track_id not in cow_registry:
            cow_registry[track_id] = CowState(track_id)
        
        cow = cow_registry[track_id]
        cow.update(crop_rgb, pose_vec)
        
        # Classify
        status = "Analyzing..."
        if cow.is_ready() and frame_count % 5 == 0:
            cow.predict()
        
        if cow.predictions:
            avg = np.mean(cow.predictions[-10:], axis=0)
            is_lame = avg[1] > 0.5
            conf = max(avg)
            status = f"TOPAL {conf:.1%}" if is_lame else f"SAGLIKLI {conf:.1%}"
        
        labels.append(f"#{track_id} {status}")
    
    # Annotate
    if len(masks_list) > 0:
        detections.mask = np.array(masks_list)
    
    annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    
    out.write(annotated)
    pbar.update(1)

pbar.close()
cap.release()
out.release()

# Calculate Performance Metrics
elapsed_time = time.time() - start_time
processing_fps = frame_count / elapsed_time

print("="*60)
print("✅ Video Processing Complete")
print(f"📊 Performance Metrics:")
print(f"   Total Frames: {frame_count}")
print(f"   Processing Time: {elapsed_time:.2f} seconds")
print(f"   Processing FPS: {processing_fps:.2f}")
print(f"   Speedup: {processing_fps/fps:.2f}x realtime")
print(f"   Cows Tracked: {len(cow_registry)}")
print("="*60)
""")

add_markdown("## 7. Generate Clinical Report")
add_code("""
report_data = []

for cid, state in cow_registry.items():
    if not state.predictions:
        status = "Insufficient Data"
        score = 0.0
    else:
        avg = np.mean(state.predictions, axis=0)
        is_lame = avg[1] > 0.5
        status = "TOPAL (LAME)" if is_lame else "SAGLIKLI (HEALTHY)"
        score = avg[1] if is_lame else avg[0]
        
    report_data.append({
        'Cow_ID': cid,
        'Diagnosis': status,
        'Confidence': f"{score:.4f}",
        'Frames_Tracked': len(state.predictions),
        'Duration_Seconds': len(state.predictions) / fps
    })

df_report = pd.DataFrame(report_data)
df_report.to_csv("/content/drive/MyDrive/outputs_v16_academic/clinical_report_v16.csv", index=False)

print("\\n📊 Clinical Report:")
print(df_report.to_string())
print(f"\\n✅ Report saved to: clinical_report_v16.csv")
""")

file_name = "c:/Users/Umut/PycharmProjects/CowLamenessDetection/02_Cow_Lameness_Inference_Multi_v16.ipynb"
os.makedirs(os.path.dirname(file_name), exist_ok=True)

print(f"Saving inference notebook to {file_name}...")
with open(file_name, "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("Done.")
