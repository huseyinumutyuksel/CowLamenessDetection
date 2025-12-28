"""
DeepLabCut SuperAnimal - Local Batch Processing Script
Run this on your local machine to generate CSV files for 1168 videos.
Then upload CSVs to Google Drive for use in Colab.

Requirements:
    pip install deeplabcut
    
Usage:
    python run_dlc_local.py
"""

import deeplabcut
import os
import glob
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# ========================
# CONFIGURATION
# ========================

# Path to your cow videos (update this to your Google Drive folder path on local machine)
# If you have Google Drive Desktop, this might be:
# Windows: "C:/Users/YourName/Google Drive/My Drive/Inek Topallik..."
# Mac: "/Users/YourName/Google Drive/My Drive/Inek Topallik..."
BASE_VIDEO_DIR = "path/to/your/cow_single_videos"  # UPDATE THIS!

# Working directory for DLC project
DLC_WORK_DIR = "./dlc_project"

# SuperAnimal model
DLC_MODEL = "superanimal_quadruped"

# ========================
# MAIN SCRIPT
# ========================

def main():
    print("="*60)
    print("🐄 DeepLabCut SuperAnimal - Batch Processing")
    print("="*60)
    
    # Check if base directory exists
    if not os.path.exists(BASE_VIDEO_DIR):
        print(f"\n❌ ERROR: Video directory not found!")
        print(f"   Path: {BASE_VIDEO_DIR}")
        print(f"\n💡 Please update BASE_VIDEO_DIR in this script to point to:")
        print(f"   .../cow_single_videos/")
        return
    
    # Get all videos
    video_paths = []
    for label in ['Saglikli', 'Topal']:
        folder = os.path.join(BASE_VIDEO_DIR, label)
        if os.path.exists(folder):
            vids = glob.glob(os.path.join(folder, "*.mp4"))
            video_paths.extend(vids)
            print(f"✓ Found {len(vids)} videos in {label}/")
    
    if not video_paths:
        print("\n❌ No videos found! Check your folder structure:")
        print("   cow_single_videos/")
        print("   ├── Saglikli/")
        print("   │   └── *.mp4")
        print("   └── Topal/")
        print("       └── *.mp4")
        return
    
    print(f"\n📊 Total videos to process: {len(video_paths)}")
    
    # Check if CSVs already exist (resume capability)
    existing_csvs = []
    for vid in video_paths:
        csv_pattern = vid.replace('.mp4', '*DLC*.csv')
        csvs = glob.glob(csv_pattern)
        if csvs:
            existing_csvs.append(vid)
    
    if existing_csvs:
        print(f"\n✓ Found {len(existing_csvs)} videos already processed")
        user_input = input("Continue processing remaining videos? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            return
        # Filter out already processed
        video_paths = [v for v in video_paths if v not in existing_csvs]
        print(f"📌 Remaining: {len(video_paths)} videos")
    
    if not video_paths:
        print("\n✅ All videos already processed!")
        return
    
    # Create DLC project (or load existing)
    os.makedirs(DLC_WORK_DIR, exist_ok=True)
    
    config_file = f"{DLC_WORK_DIR}/config.yaml"
    if os.path.exists(config_file):
        print(f"\n📂 Using existing DLC project: {config_file}")
        config_path = config_file
    else:
        print(f"\n🆕 Creating new DLC project...")
        try:
            config_path = deeplabcut.create_pretrained_project(
                "CowGaitAnalysis",
                "Local",
                [video_paths[0]],  # Just one video to initialize
                working_directory=DLC_WORK_DIR,
                copy_videos=False,
                analyzevideo=False,
                model=DLC_MODEL,
                videotype=".mp4"
            )
            print(f"✓ Project created: {config_path}")
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            return
    
    # Estimate processing time
    estimated_minutes = len(video_paths) * 2  # ~2 min per video
    hours = estimated_minutes // 60
    minutes = estimated_minutes % 60
    print(f"\n⏰ Estimated time: ~{hours}h {minutes}m")
    print(f"   (based on 2 min/video average)")
    
    user_input = input("\nStart batch processing? (y/n): ")
    if user_input.lower() != 'y':
        print("Cancelled.")
        return
    
    # Batch processing with progress bar
    print(f"\n🚀 Starting batch analysis...")
    print("="*60)
    
    successful = 0
    failed = []
    
    for vid_path in tqdm(video_paths, desc="Processing videos"):
        try:
            # Analyze single video
            deeplabcut.analyze_videos(
                config_path,
                [vid_path],
                videotype=".mp4",
                save_as_csv=True,
                destfolder=None  # Save next to video
            )
            successful += 1
        except Exception as e:
            failed.append((vid_path, str(e)))
            print(f"\n⚠️ Failed: {os.path.basename(vid_path)} - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successful: {successful}/{len(video_paths)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print("\nFailed videos:")
        for vid, err in failed[:10]:  # Show first 10
            print(f"  - {os.path.basename(vid)}: {err[:50]}...")
    
    # Instructions for next step
    print("\n" + "="*60)
    print("📤 NEXT STEPS:")
    print("="*60)
    print("1. CSV files have been created alongside your videos")
    print("   Look for files like: *DLC_resnet50*.csv")
    print("")
    print("2. Upload these CSVs to Google Drive:")
    print("   - If using Google Drive Desktop: They'll sync automatically")
    print("   - Otherwise: Upload manually to same folders as videos")
    print("")
    print("3. In Colab, the training notebook will auto-detect CSVs")
    print("   and skip the DLC analysis phase")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
