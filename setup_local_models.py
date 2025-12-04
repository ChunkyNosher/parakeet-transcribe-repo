#!/usr/bin/env python3
"""
Setup script to download and save NeMo ASR models as local .nemo files.

This script downloads models from HuggingFace once and saves them as .nemo files
in the local_models/ directory. After running this script, transcribe_ui.py will
load models from local files instead of HuggingFace, providing:
- Completely offline operation
- Faster loading times
- No re-download issues from HuggingFace revision changes

Usage:
    python setup_local_models.py

Requirements:
    - NVIDIA GPU with CUDA support
    - NeMo Framework installed
    - Internet connection (for initial download only)
"""

import os
import sys
import time
from pathlib import Path

def setup_models():
    """Download and save models as local .nemo files."""
    
    print("\n" + "="*80)
    print("🚀 NeMo Model Setup Script")
    print("="*80)
    print("\nThis script will download ASR models from HuggingFace and save them")
    print("as local .nemo files for offline use.\n")
    
    # Check for CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  WARNING: No CUDA GPU detected!")
            print("   Models will still download, but transcription requires a GPU.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        else:
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Please install PyTorch with CUDA support first.")
        sys.exit(1)
    
    # Check for NeMo
    try:
        import nemo.collections.asr as nemo_asr
        print("✅ NeMo Framework available")
    except ImportError:
        print("❌ NeMo Framework not installed!")
        print("   Please install NeMo first: pip install nemo_toolkit[asr]")
        sys.exit(1)
    
    # Create local_models directory
    script_dir = Path(__file__).parent.absolute()
    local_models_dir = script_dir / "local_models"
    
    if not local_models_dir.exists():
        print(f"\n📁 Creating directory: {local_models_dir}")
        local_models_dir.mkdir(parents=True)
    
    # Model configurations
    models = {
        "parakeet": {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v2",
            "local_path": local_models_dir / "parakeet.nemo",
            "display_name": "Parakeet-TDT-0.6B v2",
            "size_estimate": "~600 MB"
        },
        "canary": {
            "repo_id": "nvidia/canary-qwen-2.5b",
            "local_path": local_models_dir / "canary.nemo",
            "display_name": "Canary-Qwen-2.5B",
            "size_estimate": "~5.1 GB"
        }
    }
    
    print("\n" + "-"*80)
    print("Models to download:")
    for name, config in models.items():
        status = "✅ Already exists" if config["local_path"].exists() else f"⬇️  Will download ({config['size_estimate']})"
        print(f"  • {config['display_name']}: {status}")
    print("-"*80)
    
    # Check if all models already exist
    all_exist = all(config["local_path"].exists() for config in models.values())
    if all_exist:
        print("\n✅ All models already downloaded and saved!")
        print("   You can run transcribe_ui.py now.")
        return
    
    # Ask for confirmation
    print("\n⚠️  This will download large model files from HuggingFace.")
    print("   Total size: ~5.7 GB")
    print("   This only needs to be done once.\n")
    
    response = input("Continue with download? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Download and save each model
    for name, config in models.items():
        if config["local_path"].exists():
            print(f"\n✅ {config['display_name']}: Already exists, skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"⬇️  Downloading {config['display_name']}...")
        print(f"   Source: {config['repo_id']}")
        print(f"   Estimated size: {config['size_estimate']}")
        print("="*80)
        
        try:
            start_time = time.time()
            
            # Download from HuggingFace
            print("   Step 1/2: Downloading from HuggingFace...")
            model = nemo_asr.models.ASRModel.from_pretrained(config["repo_id"])
            download_time = time.time() - start_time
            
            # Save to local .nemo file
            print(f"   Step 2/2: Saving to {config['local_path']}...")
            save_start = time.time()
            model.save_to(str(config["local_path"]))
            save_time = time.time() - save_start
            
            total_time = time.time() - start_time
            file_size = config["local_path"].stat().st_size / (1024 * 1024)
            
            print(f"\n✅ {config['display_name']} saved successfully!")
            print(f"   File: {config['local_path']}")
            print(f"   Size: {file_size:.1f} MB")
            print(f"   Download time: {download_time:.1f}s")
            print(f"   Save time: {save_time:.1f}s")
            print(f"   Total time: {total_time:.1f}s")
            
            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n❌ Error downloading {config['display_name']}:")
            print(f"   {type(e).__name__}: {str(e)}")
            print("\n   Please check:")
            print("   1. Internet connection is available")
            print("   2. Sufficient disk space (~6 GB free)")
            print("   3. HuggingFace is accessible")
            sys.exit(1)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 Setup Complete!")
    print("="*80)
    print("\nAll models have been saved to local_models/:")
    for name, config in models.items():
        if config["local_path"].exists():
            size_mb = config["local_path"].stat().st_size / (1024 * 1024)
            print(f"  ✅ {config['local_path'].name} ({size_mb:.1f} MB)")
    
    print("\n📝 Next steps:")
    print("   1. Run: python transcribe_ui.py")
    print("   2. Open http://127.0.0.1:7860 in your browser")
    print("   3. Enjoy 100% offline transcription!")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    setup_models()
