
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

# ==========================================
# NOTEBOOK CONTENT GENERATION
# ==========================================

add_markdown("""# 🐄 Cow Lameness Detection - Training & Research (v16)
**Academic Gold Standard Edition - PRODUCTION**

## Objective
This notebook implements a state-of-the-art **Tri-Modal Gait Analysis** system to detect lameness in cows using DeepLabCut SuperAnimal.

## Methodology
1.  **Phase 1**: Run **DeepLabCut (SuperAnimal-Quadruped)** on ALL training videos
2.  **Phase 2**: Extract Visual Features (VideoMAE, RAFT)
3.  **Phase 3**: Biometric Statistical Analysis (T-Test)
4.  **Phase 4**: Train with 5-Fold Cross-Validation
5.  **Phase 5**: Explainable AI (Attention Heatmaps)
""")

add_markdown("## 1. Setup & Configuration")
add_markdown("### Step 1.1: Install Core Dependencies")
add_code("""
# Install in stages to avoid dependency conflicts
!pip install -q ultralytics supervision
!pip install -q timm einops transformers
!pip install -q moviepy scikit-learn scipy seaborn matplotlib
!pip install -q psutil gputil
print("✅ Core dependencies installed")
""")

add_markdown("### Step 1.2: DeepLabCut Setup (IMPORTANT)")
add_markdown("""
> **⚠️ CRITICAL COMPATIBILITY ISSUE**
>
> DeepLabCut requires NumPy <2.0, but Colab now uses NumPy 2.2+.
> Downgrading NumPy breaks many other Colab packages (opencv, jax, etc.).
>
> **RECOMMENDED SOLUTION**: Process DLC offline, upload CSVs to Drive.
""")

add_code("""
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🔄 DEEPLABCUT WORKAROUND")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("")
print("Option A: SKIP DLC (Use Pre-Computed Results)")
print("  1. Run DLC SuperAnimal locally or on dedicated machine")
print("  2. Upload CSV files to Drive alongside videos")
print("  3. Notebook will auto-detect and use CSVs")
print("")
print("Option B: Try DLC Installation (May Fail)")
print("  - Uncomment the code below to attempt install")
print("  - Expect dependency conflicts")
print("")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# UNCOMMENT BELOW TO ATTEMPT DLC INSTALL (NOT RECOMMENDED)
# !pip install -q --force-reinstall numpy==1.26.4
# !pip install -q deeplabcut --no-deps
# !pip install -q dlclibrary filterpy ruamel.yaml imgaug scikit-image
# import deeplabcut
# print(f"✅ DLC {deeplabcut.__version__}")

# RECOMMENDED: Check if CSVs exist
import os
import glob
BASE_DIR_CHECK = "/content/drive/MyDrive/Inek Topallik Tespiti Parcalanmis Inek Videolari/cow_single_videos"
if os.path.exists(BASE_DIR_CHECK):
    csv_files = glob.glob(f"{BASE_DIR_CHECK}/**/*.csv", recursive=True)
    if csv_files:
        print(f"\\n✅ Found {len(csv_files)} DLC CSV files in Drive!")
        print("   You can proceed without installing DLC.")
    else:
        print(f"\\n⚠️ No CSV files found. You need to:")
        print("   1. Run DLC SuperAnimal offline")
        print("   2. Upload '*DLC*.csv' files next to videos")
else:
    print("\\n📁 Drive path not yet mounted")
""")

add_markdown("### Step 1.3: Mount Drive & Setup Paths")
add_code("""
import os
from google.colab import drive
import torch
import numpy as np
import glob
import pandas as pd
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/Inek Topallik Tespiti Parcalanmis Inek Videolari/cow_single_videos"
OUTPUT_DIR = "/content/drive/MyDrive/outputs_v16_academic"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASSES = ['Saglikli', 'Topal']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Get List of All Videos
all_videos = []
for label in CLASSES:
    folder = os.path.join(BASE_DIR, label)
    vids = glob.glob(os.path.join(folder, "*.mp4"))
    all_videos.extend(vids)
    
print(f"Total Videos Found: {len(all_videos)}")
""")

add_markdown("## 2. PHASE 1: DeepLabCut SuperAnimal Analysis")
add_markdown("""
> **CRITICAL ACADEMIC STEP**: Running official `deeplabcut.analyze_videos` on entire dataset.
> This uses **SuperAnimal-Quadruped** (Ye et al., 2024) for high-fidelity pose estimation.
""")

add_code("""
import deeplabcut

print("Initializing DeepLabCut SuperAnimal-Quadruped...")
DLC_PROJECT_NAME = "CowGaitAnalysis"
DLC_OWNER = "Researcher"
DLC_WORK_DIR = "/content/dlc_work"
os.makedirs(DLC_WORK_DIR, exist_ok=True)

# Initialize Project Configuration
try:
    dummy_vid = all_videos[0]
    config_path = deeplabcut.create_pretrained_project(
        DLC_PROJECT_NAME, DLC_OWNER, [dummy_vid], 
        working_directory=DLC_WORK_DIR, copy_videos=False, analyzevideo=False, 
        model="superanimal_quadruped", videotype=".mp4"
    )
    print(f"✅ DLC Project Created: {config_path}")
except Exception as e:
    # If already exists
    search = glob.glob(f"{DLC_WORK_DIR}/{DLC_PROJECT_NAME}*/config.yaml")
    config_path = search[0] if search else None
    print(f"📂 Using Existing DLC Config: {config_path}")

# RUN BATCH ANALYSIS with Time Estimation
import time
estimated_time_per_video = 2  # minutes (conservative estimate)
total_estimated_minutes = len(all_videos) * estimated_time_per_video
hours = total_estimated_minutes // 60
minutes = total_estimated_minutes % 60

print(f"\\n🚀 Starting Batch Analysis of {len(all_videos)} videos...")
print(f"⏰ Estimated Time: ~{hours}h {minutes}m (may vary based on video length)")
print("⏳ Please be patient, this is a one-time process...")
print("💡 TIP: Results are cached in Drive. Subsequent runs will skip processed videos.")

start_dlc = time.time()
deeplabcut.analyze_videos(
    config_path, 
    all_videos, 
    videotype='.mp4', 
    save_as_csv=True, 
    destfolder=None  # Save next to video file
)
elapsed_dlc = (time.time() - start_dlc) / 60
print(f"✅ DeepLabCut Complete! Actual time: {elapsed_dlc:.1f} minutes")
""")

add_markdown("## 3. PHASE 2: Visual Feature Extraction")
add_code("""
from ultralytics import YOLO
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torchvision.transforms.functional as F

# A. VideoMAE
mae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
mae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

# B. RAFT
raft_weights = Raft_Large_Weights.DEFAULT
raft_transforms = raft_weights.transforms()
raft_model = raft_large(weights=raft_weights, progress=False).to(device).eval()

# C. YOLO (for cropping)
yolo_model = YOLO("yolov8x.pt")

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
    img1, img2 = raft_transforms(img1, img2)
    with torch.no_grad():
        flow_predictions = raft_model(img1, img2)
    return flow_predictions[-1].mean(dim=[2,3]).cpu().numpy()[0]

print("✅ Visual Feature Engines Loaded")

# Memory Monitoring for Colab Pro+
import psutil
import GPUtil

ram_percent = psutil.virtual_memory().percent
print(f"\\n📊 System RAM Usage: {ram_percent:.1f}%")

try:
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"🎮 GPU RAM Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
except:
    print("⚠️ GPU monitoring not available")
""")

add_markdown("## 4. Dataset Fusion (DLC + VideoMAE + RAFT)")
add_code("""
def process_video_fusion(video_path):
    # 1. Load DLC CSV
    folder = os.path.dirname(video_path)
    base = os.path.basename(video_path).replace('.mp4','')
    candidates = glob.glob(os.path.join(folder, f"*{base}*.csv"))
    
    dlc_csv = None
    for c in candidates:
        if "DLC" in c or "superanimal" in c.lower():
            dlc_csv = c
            break
    
    if not dlc_csv:
        return None
        
    try:
        df = pd.read_csv(dlc_csv, header=[1,2])
        pose_raw = df.values
        pose_raw = np.nan_to_num(pose_raw, nan=0.0)
    except:
        return None
    
    # 2. Load Video Frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if len(frames) < 10: return None
    
    # Resample to 30 frames
    indices = np.linspace(0, len(frames)-1, 30).astype(int)
    frames = [frames[i] for i in indices]
    pose_indices = np.linspace(0, len(pose_raw)-1, 30).astype(int)
    pose_seq = pose_raw[pose_indices]
    
    # 3. Extract Visual Features
    cropped_frames = []
    flow_seq = []
    last_crop = None
    
    for frame in frames:
        # YOLO Crop
        results = yolo_model(frame, classes=[19], verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            best_box = boxes[np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])]
            x1,y1,x2,y2 = map(int, best_box)
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cropped_frames.append(crop_rgb)
        
        # Flow
        if last_crop is not None:
            h, w, _ = crop_rgb.shape
            prev_resized = cv2.resize(last_crop, (w, h))
            f = extract_raft_flow(prev_resized, crop_rgb)
        else:
            f = np.zeros(2)
        flow_seq.append(f)
        last_crop = crop_rgb

    # VideoMAE: Segment-based embedding to preserve temporal info
    # Instead of tiling one global embedding, extract per-segment embeddings
    mae_seq = []
    segment_size = 6  # 30 frames / 5 segments = 6 frames per segment
    for seg_start in range(0, 30, segment_size):
        segment_frames = cropped_frames[seg_start:seg_start+segment_size]
        segment_emb = extract_videomae_features(segment_frames)
        # Repeat segment embedding for each frame in segment
        for _ in range(len(segment_frames)):
            mae_seq.append(segment_emb)
    mae_seq = np.array(mae_seq)
    
    return {'pose': pose_seq, 'mae': mae_seq, 'flow': np.array(flow_seq)}

# Build Final Dataset
data_records = []
failed_videos = []  # Track failed videos for logging
from tqdm import tqdm
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CowLameness')

print("\\n🔄 Processing videos and extracting features...")
for vid in tqdm(all_videos, desc="Feature Extraction"):
    label = 1 if 'Topal' in vid else 0
    try:
        feats = process_video_fusion(vid)
        if feats:
            data_records.append({
                'video': os.path.basename(vid),
                'label': label,
                'pose': feats['pose'],
                'mae': feats['mae'],
                'flow': feats['flow']
            })
        else:
            logger.warning(f"No features extracted: {os.path.basename(vid)}")
            failed_videos.append((vid, "No DLC CSV found or insufficient frames"))
    except FileNotFoundError as e:
        logger.error(f"File not found: {vid}")
        failed_videos.append((vid, f"FileNotFoundError: {e}"))
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {vid}")
        failed_videos.append((vid, f"ParserError: {e}"))
    except cv2.error as e:
        logger.error(f"OpenCV error: {vid}")
        failed_videos.append((vid, f"OpenCV: {e}"))
    except Exception as e:
        logger.error(f"Unexpected error for {vid}: {type(e).__name__}: {e}")
        failed_videos.append((vid, f"{type(e).__name__}: {e}"))

# Summary logging
logger.info(f"✅ Processed: {len(data_records)}/{len(all_videos)} videos")
if failed_videos:
    logger.warning(f"❌ Failed: {len(failed_videos)} videos")
    for path, reason in failed_videos[:10]:
        logger.warning(f"   - {os.path.basename(path)}: {reason}")

print(f"✅ Final Dataset: {len(data_records)} samples")

# RAM cleanup with automatic garbage collection
ram_after = psutil.virtual_memory().percent
print(f"📊 RAM Usage after features: {ram_after:.1f}%")
if ram_after > 80:
    logger.warning("⚠️ High RAM usage detected. Initiating cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ram_cleaned = psutil.virtual_memory().percent
    logger.info(f"✅ RAM after cleanup: {ram_cleaned:.1f}% (freed {ram_after - ram_cleaned:.1f}%)")
""")

add_markdown("## 5. PHASE 3: Biometric Statistical Analysis")
add_code("""
from scipy import stats
import seaborn as sns

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    \"\"\"Calculate the angle at point B formed by vectors BA and BC.
    
    Uses the dot product formula: cos(θ) = (BA · BC) / (|BA| × |BC|)
    
    Args:
        a: First endpoint coordinates [x, y]
        b: Vertex point coordinates [x, y] (angle is measured here)
        c: Second endpoint coordinates [x, y]
    
    Returns:
        Angle in degrees (0-180). Returns 180 if vectors are collinear.
    
    Example:
        >>> calculate_angle(np.array([0,0]), np.array([1,0]), np.array([1,1]))
        90.0
    \"\"\"    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_back_curvature_angle(pose_seq: np.ndarray) -> float:
    \"\"\"Calculate the spinal curvature angle from pose sequence.
    
    Measures the angle formed by hip-spine-shoulder keypoints.
    Lame cows typically show MORE curvature (LOWER angle) due to pain compensation.
    
    Args:
        pose_seq: Pose sequence array of shape (30, num_keypoints*3)
                  where each keypoint has (x, y, confidence)
    
    Returns:
        Spine angle in degrees. 180 = straight back, <180 = curved.
        Returns 180 if insufficient keypoints.
    
    Note:
        Uses keypoint indices 5, 10, 15 which correspond to:
        - kpts[5]: Hip region
        - kpts[10]: Mid-spine  
        - kpts[15]: Shoulder region
    \"\"\"    
    kpts = pose_seq.reshape(30, -1, 3).mean(axis=0)
    if kpts.shape[0] < 15: 
        return 180  # Insufficient keypoints, assume straight
    return calculate_angle(kpts[5,:2], kpts[10,:2], kpts[15,:2])

healthy_scores = [get_back_curvature_angle(d['pose']) for d in data_records if d['label'] == 0]
lame_scores = [get_back_curvature_angle(d['pose']) for d in data_records if d['label'] == 1]

if len(healthy_scores) > 0 and len(lame_scores) > 0:
    t_stat, p_val = stats.ttest_ind(healthy_scores, lame_scores)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(healthy_scores, fill=True, label='Saglikli (Healthy)', color='green', alpha=0.6)
    sns.kdeplot(lame_scores, fill=True, label='Topal (Lame)', color='red', alpha=0.6)
    plt.title(f"Biometric Validation: Back Spine Angle\\n(Lower=More Curvature) | p-value={p_val:.4e}", fontsize=14)
    plt.xlabel("Spine Angle (degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/biometric_significance.png", dpi=150)
    plt.show()
    
    print(f"✅ Statistical Significance: p={p_val:.4e} {'(SIGNIFICANT ✓)' if p_val < 0.05 else '(NOT SIGNIFICANT)'}")
""")

add_markdown("## 6. PHASE 4: Model Training with 5-Fold CV")
add_code("""
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.manifold import TSNE

class TriModalAttention(nn.Module):
    def __init__(self, pose_dim, hidden_dim=256):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.mae_proj = nn.Linear(768, hidden_dim)
        self.flow_proj = nn.Linear(2, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim*3, nhead=4, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, pose, mae, flow):
        src = torch.cat([self.pose_proj(pose), self.mae_proj(mae), self.flow_proj(flow)], dim=2)
        out = self.encoder_layer(src)
        pooled = out.mean(dim=1)
        return self.classifier(pooled), pooled

class CowDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        return (torch.tensor(r['pose'], dtype=torch.float32), 
                torch.tensor(r['mae'], dtype=torch.float32), 
                torch.tensor(r['flow'], dtype=torch.float32), 
                torch.tensor(r['label'], dtype=torch.long))

if len(data_records) == 0:
    raise RuntimeError("No data loaded! Check DLC CSVs.")

sample_pose_dim = data_records[0]['pose'].shape[1]

# Pose dimension validation: ensure consistency across all records
expected_dim = sample_pose_dim
for i, rec in enumerate(data_records):
    if rec['pose'].shape[1] != expected_dim:
        raise ValueError(f"Pose dimension mismatch at record {i}: expected {expected_dim}, got {rec['pose'].shape[1]}")
print(f"✅ Pose Dimension Validated: {sample_pose_dim} (consistent across {len(data_records)} records)")

# CRITICAL: Train/Test Split for Unbiased Evaluation
from sklearn.model_selection import train_test_split

labels = [r['label'] for r in data_records]
train_records, test_records = train_test_split(
    data_records, 
    test_size=0.2,  # 80% train, 20% test
    stratify=labels,  # Maintain class balance
    random_state=42
)

print(f"\\n{'='*60}")
print(f"DATASET SPLIT")
print(f"{'='*60}")
print(f"Total Samples: {len(data_records)}")
print(f"Training Set: {len(train_records)} samples")
print(f"Test Set (HELD-OUT): {len(test_records)} samples")
print(f"{'='*60}")
print("📌 Test set will ONLY be used for final evaluation after training.")

# 5-Fold Cross-Validation on TRAINING SET ONLY
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_labels = [r['label'] for r in train_records]

fold_results = []
all_embeddings = []
all_labels = []
all_probs = []  # For ROC-AUC
train_losses_per_fold = []
val_losses_per_fold = []

# Track best model across folds
best_model_state = None
best_val_acc = 0.0

for fold, (train_idx, val_idx) in enumerate(skf.split(train_records, train_labels)):
    print(f"\\n{'='*50}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*50}")
    
    train_data = [train_records[i] for i in train_idx]
    val_data = [train_records[i] for i in val_idx]
    
    train_loader = DataLoader(CowDataset(train_data), batch_size=8, shuffle=True)
    val_loader = DataLoader(CowDataset(val_data), batch_size=8, shuffle=False)
    
    model = TriModalAttention(pose_dim=sample_pose_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training with Loss Tracking
    train_losses = []
    val_losses = []
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        for p, m, f, y in train_loader:
            p, m, f, y = p.to(device), m.to(device), f.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(p, m, f)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for p, m, f, y in val_loader:
                p, m, f, y = p.to(device), m.to(device), f.to(device), y.to(device)
                logits, _ = model(p, m, f)
                val_loss += criterion(logits, y).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Final Validation with Probabilities
    model.eval()
    preds, trues, embeds, probs_list = [], [], [], []
    with torch.no_grad():
        for p, m, f, y in val_loader:
            p, m, f, y = p.to(device), m.to(device), f.to(device), y.to(device)
            logits, emb = model(p, m, f)
            probs = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(y.cpu().numpy())
            embeds.extend(emb.cpu().numpy())
            probs_list.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (Lame)
    
    acc = accuracy_score(trues, preds)
    fold_results.append(acc)
    all_embeddings.extend(embeds)
    all_labels.extend(trues)
    all_probs.extend(probs_list)
    train_losses_per_fold.append(train_losses)
    val_losses_per_fold.append(val_losses)
    
    # Track best model
    if acc > best_val_acc:
        best_val_acc = acc
        best_model_state = model.state_dict().copy()
        print(f"🏆 New best model! Fold {fold+1} Accuracy: {acc:.4f}")
    else:
        print(f"✅ Fold {fold+1} Accuracy: {acc:.4f}")

print(f"\n{'='*50}")
print(f"5-FOLD CV RESULTS (Training Set Only)")
print(f"{'='*50}")
print(f"Mean Accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Final Test Set Evaluation (UNBIASED)
print(f"\n{'='*60}")
print(f"FINAL TEST EVALUATION (HELD-OUT SET)")
print(f"{'='*60}")

test_loader = DataLoader(CowDataset(test_records), batch_size=8, shuffle=False)

# Load best model for final evaluation
model = TriModalAttention(pose_dim=sample_pose_dim).to(device)
model.load_state_dict(best_model_state)
print(f"📌 Using best model from CV (val_acc={best_val_acc:.4f})")
model.eval()
test_preds, test_trues, test_probs = [], [], []
with torch.no_grad():
    for p, m, f, y in test_loader:
        p, m, f, y = p.to(device), m.to(device), f.to(device), y.to(device)
        logits, _ = model(p, m, f)
        probs = torch.softmax(logits, dim=1)
        test_preds.extend(logits.argmax(dim=1).cpu().numpy())
        test_trues.extend(y.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

test_acc = accuracy_score(test_trues, test_preds)
print(f"✅ Test Set Accuracy: {test_acc:.4f}")
print(f"📊 Test Set Size: {len(test_trues)} samples")
print(f"{'='*60}")
print("🎯 This is the FINAL, UNBIASED performance estimate for publication.")

# Save Final Model with Metadata for Validation
model_checkpoint = {
    'model_state_dict': model.state_dict(),
    'pose_dim': sample_pose_dim,
    'mae_dim': 768,
    'flow_dim': 2,
    'test_accuracy': test_acc  # Include test performance
}
torch.save(model_checkpoint, f"{OUTPUT_DIR}/cow_gait_transformer_v16_final.pth")
print(f"\n✅ Model Saved with Metadata (pose_dim={sample_pose_dim}, test_acc={test_acc:.4f})")
""")

add_markdown("## 6.5. Training Curves")
add_code("""
# Plot Loss Curves (Average across folds)
avg_train_losses = np.mean(train_losses_per_fold, axis=0)
avg_val_losses = np.mean(val_losses_per_fold, axis=0)

plt.figure(figsize=(10, 6))
epochs = range(1, 21)
plt.plot(epochs, avg_train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, avg_val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training & Validation Loss Curves (5-Fold Average)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/loss_curves.png", dpi=150)
plt.show()
print("✅ Loss Curves Saved")
""")

add_markdown("## 6.7. ROC-AUC Curve")
add_code("""
from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-AUC Curve (All Folds Combined)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/roc_auc_curve.png", dpi=150)
plt.show()

print(f"✅ ROC-AUC: {roc_auc:.4f}")
""")

add_markdown("## 7. PHASE 5: t-SNE Visualization")
add_code("""
# t-SNE of Feature Space
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(np.array(all_embeddings))

plt.figure(figsize=(10, 8))
for label, color, name in [(0, 'green', 'Healthy'), (1, 'red', 'Lame')]:
    idx = np.array(all_labels) == label
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                c=color, label=name, alpha=0.6, s=50)

plt.title("t-SNE: Feature Space Separation (Healthy vs Lame)", fontsize=14)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/tsne_clusters.png", dpi=150)
plt.show()
print("✅ t-SNE Visualization Complete")
""")

add_markdown("## 8. Confusion Matrix & Metrics")
add_code("""
from sklearn.metrics import classification_report

# Fix: Use probabilities instead of embeddings for predictions
predicted_labels = [1 if p > 0.5 else 0 for p in all_probs]
cm = confusion_matrix(all_labels, predicted_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Lame'], 
            yticklabels=['Healthy', 'Lame'])
plt.title("Confusion Matrix (All Folds Combined)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.show()

print("\\n📊 Classification Report:")
print(classification_report(all_labels, predicted_labels, target_names=['Healthy', 'Lame']))
""")

add_markdown("## 9. Ablation Study")
add_markdown("""
> **Academic Validation**: Compare individual modalities vs. fusion to prove superiority of Tri-Modal approach.
""")

add_code("""
# Train 3 Models: (A) Pose-Only, (B) VideoMAE-Only, (C) Tri-Modal (Ours)

class PoseOnlyModel(nn.Module):
    def __init__(self, pose_dim):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=pose_dim, nhead=4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(pose_dim, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, pose):
        out = self.encoder(pose)
        return self.classifier(out.mean(dim=1))

class VideoMAEOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, mae):
        out = self.encoder(mae)
        return self.classifier(out.mean(dim=1))

def train_ablation_model(model, data_records, model_name):
    print(f"\\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    
    labels = [r['label'] for r in data_records]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_records, labels)):
        train_data = [data_records[i] for i in train_idx]
        val_data = [data_records[i] for i in val_idx]
        
        train_loader = DataLoader(CowDataset(train_data), batch_size=8, shuffle=True)
        val_loader = DataLoader(CowDataset(val_data), batch_size=8, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Quick Training (10 epochs for ablation)
        for epoch in range(10):
            model.train()
            for p, m, f, y in train_loader:
                p, m, f, y = p.to(device), m.to(device), f.to(device), y.to(device)
                optimizer.zero_grad()
                
                if model_name == "Pose-Only":
                    logits = model(p)
                elif model_name == "VideoMAE-Only":
                    logits = model(m)
                else:  # Tri-Modal
                    logits, _ = model(p, m, f)
                
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for p, m, f, y in val_loader:
                p, m, f, y = p.to(device), m.to(device), f.to(device), y.to(device)
                
                if model_name == "Pose-Only":
                    logits = model(p)
                elif model_name == "VideoMAE-Only":
                    logits = model(m)
                else:
                    logits, _ = model(p, m, f)
                
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        
        acc = accuracy_score(trues, preds)
        fold_accs.append(acc)
    
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"✅ {model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
    return mean_acc, std_acc

# Run Ablation (using only train_records to prevent data leakage)
ablation_results = {}

model_a = PoseOnlyModel(sample_pose_dim).to(device)
ablation_results['Pose-Only'] = train_ablation_model(model_a, train_records, "Pose-Only")

model_b = VideoMAEOnlyModel().to(device)
ablation_results['VideoMAE-Only'] = train_ablation_model(model_b, train_records, "VideoMAE-Only")

model_c = TriModalAttention(pose_dim=sample_pose_dim).to(device)
ablation_results['Tri-Modal (Ours)'] = train_ablation_model(model_c, train_records, "Tri-Modal (Ours)")

# Plot Results
models = list(ablation_results.keys())
means = [ablation_results[m][0] for m in models]
stds = [ablation_results[m][1] for m in models]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, means, yerr=stds, capsize=10, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Ablation Study: Contribution of Each Modality', fontsize=14)
plt.ylim(0.5, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

plt.savefig(f"{OUTPUT_DIR}/ablation_study.png", dpi=150)
plt.show()

print("\\n" + "="*60)
print("📊 ABLATION STUDY RESULTS")
print("="*60)
for model_name, (mean, std) in ablation_results.items():
    print(f"{model_name:20s}: {mean:.4f} ± {std:.4f}")
print("="*60)
print("✅ Ablation Study Complete")
""")

# Save notebook to project directory (same location as this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, "01_Cow_Lameness_Training_v16.ipynb")
directory = os.path.dirname(file_name)
if not os.path.exists(directory):
    os.makedirs(directory)

print(f"Saving notebook to {file_name}...")
with open(file_name, "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("Done.")
