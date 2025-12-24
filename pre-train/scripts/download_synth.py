"""
Download PleIAs/SYNTH dataset to local storage.

Dataset info:
- Size: ~236 GB
- Samples: 79,648,272
- Tokens: ~75 billion

Usage:
    python download_synth.py

Alternative CLI command:
    hf download PleIAs/SYNTH --repo-type dataset --local-dir E:\GPT_SANDBOX_STORAGE\SYNTH_DATASET
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Configuration
DATASET_NAME = "PleIAs/SYNTH"
OUTPUT_DIR = Path(r"E:\GPT_SANDBOX_STORAGE\SYNTH_DATASET")

def main():
    print("=" * 60)
    print("DOWNLOADING PleIAs/SYNTH DATASET")
    print("=" * 60)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Expected size: ~236 GB")
    print("\nThis will take a while depending on your connection speed...")
    print("The download supports resume - you can interrupt and restart safely.")
    print("=" * 60 + "\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset using official HuggingFace method
        # https://huggingface.co/docs/huggingface_hub/main/en/guides/download
        local_path = snapshot_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            local_dir=str(OUTPUT_DIR),
            resume_download=True,  # Resume if interrupted
            max_workers=4,  # Parallel downloads for speed
        )
        
        print("\n" + "=" * 60)
        print("[OK] Download complete!")
        print(f"Dataset saved to: {local_path}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n[!] Download interrupted by user.")
        print("    Run the script again to resume from where you left off.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("    Try running the script again - it will resume automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
